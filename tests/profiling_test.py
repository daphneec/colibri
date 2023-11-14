#!/usr/bin/env python3
#   by daphneec
import logging
import functools
import numpy as np
import time
import torch
import torch.nn as nn
import crypten
import crypten.nn as cnn
import os
import subprocess
import sys
sys.path.append('/Users/daphneechabal/Library/CloudStorage/OneDrive-UvA/WORK/3.NAS TRANSFORMER SECURE PAPER/securehrnas') 
from utils import distributed as udist
from utils.fix_hook import fix_hook
from utils import optim
from utils import prune

import models.secure_transformer as secure_transformer
import models.secure_mobilenet_base as mb
import models.secure_hrnet as hr
import models.secure_hrnet_base as hrb
import secure_common as mc


def shrink_model(model_wrapper,
                 ema,
                 optimizer,
                 prune_info,
                 threshold=1e-3,
                 ema_only=False):
    r"""Dynamic network shrinkage to discard dead atomic blocks.

    Args:
        model_wrapper: model to be shrinked.
        ema: An instance of `ExponentialMovingAverage`, could be None.
        optimizer: Global optimizer.
        prune_info: An instance of `PruneInfo`, could be None.
        threshold: A small enough constant.
        ema_only: If `True`, regard an atomic block as dead only when
            `$$\hat{alpha} \le threshold$$`. Otherwise use both current value
            and momentum version.
    """
    model = mc.unwrap_model(model_wrapper)
    for block_name, block in model.get_named_block_list().items():  # inverted residual blocks
        assert isinstance(block, mb.InvertedResidualChannels)
        masks = [
            bn.weight.detach().abs() > threshold
            for bn in block.get_depthwise_bn()
        ]
        if ema is not None:
            masks_ema = [
                ema.average('{}.{}.weight'.format(
                    block_name, name)).detach().abs() > threshold
                for name in block.get_named_depthwise_bn().keys()
            ]
            if not ema_only:
                masks = [
                    mask0 | mask1 for mask0, mask1 in zip(masks, masks_ema)
                ]
            else:
                masks = masks_ema
        block.compress_by_mask(masks,
                               ema=ema,
                               optimizer=optimizer,
                               prune_info=prune_info,
                               prefix=block_name,
                               verbose=False)

    if optimizer is not None:
        assert set(optimizer.param_groups[0]['params']) == set(
            model.parameters())
    secure_model = cnn.from_pytorch(model, inp)
    mc.model_profiling(model,
                       28,
                       28,
                       num_forwards=0,
                       verbose=False)
    if udist.is_master():
        logging.info('Model Shrink to FLOPS: {}'.format(secure_model.n_macs))
        logging.info('Current model: {}'.format(mb.output_network(secure_model)))

model_profiling_hooks = []
model_profiling_speed_hooks = []

name_space = 95
params_space = 15
macs_space = 15
seconds_space = 15


def get_params(self):
    """get number of params in module"""
    return np.sum([np.prod(list(w.size())) for w in self.parameters()])


def run_forward(self, input, num_forwards=10):
    if num_forwards <= 0:
        return 0.0
    with Timer() as t:
        for _ in range(num_forwards):
            self.forward(*input)
            torch.cuda.synchronize()
    return int(t.time * 1e9 / num_forwards)


def conv_module_name_filter(name):
    """filter module name to have a short view"""
    filters = {
        'kernel_size': 'k',
        'stride': 's',
        'padding': 'pad',
        'bias': 'b',
        'groups': 'g',
    }
    for k in filters:
        name = name.replace(k, filters[k])
    return name


def module_profiling(self, input, output, num_forwards, verbose):
    def add_sub(m, sub_op):
        m.n_macs += getattr(sub_op, 'n_macs', 0)
        m.n_params += getattr(sub_op, 'n_params', 0)
        m.n_seconds += getattr(sub_op, 'n_seconds', 0)

    _run_forward = functools.partial(run_forward, num_forwards=num_forwards)
    # if isinstance(self, (hr.ParallelModule, hr.FuseModule, hr.HeadModule)) \
    #     or (isinstance(self, nn.Sequential) and isinstance(self[0], hr.ParallelModule)):
    if not input:
        return
    if isinstance(self, nn.MultiheadAttention) or isinstance(input[0], list) or isinstance(output, list):
        pass
    else:
        ins = input[0].size()
        outs = output.size()
        # NOTE: There are some difference between type and isinstance, thus please
        # be careful.
        t = type(self)
        self._profiling_input_size = ins
        self._profiling_output_size = outs
    if isinstance(self, cnn.Conv2d):
        self.n_macs = (ins[1] * outs[1] * self.kernel_size[0] *
                       self.kernel_size[1] * outs[2] * outs[3] //
                       self.groups) * outs[0]
        self.n_params = get_params(self)
        self.n_seconds = _run_forward(self, input)
        self.name = conv_module_name_filter(self.__repr__())
    elif isinstance(self, nn.ConvTranspose2d):
        self.n_macs = (ins[1] * outs[1] * self.kernel_size[0] *
                       self.kernel_size[1] * outs[2] * outs[3] //
                       self.groups) * outs[0]
        self.n_params = get_params(self)
        self.n_seconds = _run_forward(self, input)
        self.name = conv_module_name_filter(self.__repr__())
    elif isinstance(self, cnn.Linear):
        self.n_macs = ins[1] * outs[1] * outs[0]
        self.n_params = get_params(self)
        self.n_seconds = _run_forward(self, input)
        self.name = self.__repr__()
    elif isinstance(self, cnn.AvgPool2d):
        # NOTE: this function is correct only when stride == kernel size
        self.n_macs = ins[1] * ins[2] * ins[3] * ins[0]
        self.n_params = 0
        self.n_seconds = _run_forward(self, input)
        self.name = self.__repr__()
    elif isinstance(self, cnn.AdaptiveAvgPool2d):
        # NOTE: this function is correct only when stride == kernel size
        self.n_macs = ins[1] * ins[2] * ins[3] * ins[0]
        self.n_params = 0
        self.n_seconds = _run_forward(self, input)
        self.name = self.__repr__()
    elif isinstance(self, mb.SqueezeAndExcitation):
        self.n_macs = ins[1] * ins[2] * ins[3] * ins[0]
        self.n_params = 0
        self.n_seconds = 0
        add_sub(self, self.se_reduce)
        add_sub(self, self.se_expand)
        self.name = self.__repr__()
    elif isinstance(self, mb.InvertedResidualChannels):
        self.n_macs = 0
        self.n_params = 0
        self.n_seconds = 0
        for op in self.ops:
            add_sub(self, op)
        add_sub(self, self.se_op)
        if not self.use_res_connect:
            add_sub(self, self.residual)
        if self.use_transformer and self.use_res_connect:
            add_sub(self, self.transformer)
        if self.use_transformer and self.downsampling_transformer and not self.use_res_connect:
            add_sub(self, self.transformer)
        self.name = self.__repr__()
    elif isinstance(self, mb.InvertedResidualChannelsFused):
        self.n_macs = 0
        self.n_params = 0
        self.n_seconds = 0
        for op in self.depth_ops:
            add_sub(self, op)
        add_sub(self, self.expand_conv)
        add_sub(self, self.project_conv)
        add_sub(self, self.se_op)
        self.name = self.__repr__()
    elif isinstance(self, hr.ParallelModule):
        self.n_macs = 0
        self.n_params = 0
        self.n_seconds = 0
        for op in self.branches:
            add_sub(self, op)
        self.name = self.__repr__()
    elif isinstance(self, hr.FuseModule):
        self.n_macs = 0
        self.n_params = 0
        self.n_seconds = 0
        for ops in self.fuse_layers:
            for op in ops:
                add_sub(self, op)
        self.name = self.__repr__()
    elif isinstance(self, hr.HeadModule):
        self.n_macs = 0
        self.n_params = 0
        self.n_seconds = 0
        for op in self.incre_modules:
            add_sub(self, op)
        for op in self.downsamp_modules:
            add_sub(self, op)
        add_sub(self, self.final_layer)
        self.name = self.__repr__()
    elif isinstance(self, secure_transformer.secure_Transformer):
        self.n_macs = 0
        self.n_params = 0
        self.n_seconds = 0
        add_sub(self, self.input_proj)
        add_sub(self, self.reverse_proj)
        add_sub(self, self.encoder)
        add_sub(self, self.decoder)
        self.name = self.__repr__()
    elif isinstance(self, secure_transformer.secure_TransformerEncoderLayer):
        self.n_macs = 0
        self.n_params = 0
        self.n_seconds = 0
        add_sub(self, self.self_attn)
        add_sub(self, self.linear1)
        add_sub(self, self.linear2)
        self.name = self.__repr__()
    elif isinstance(self, secure_transformer.secure_TransformerDecoderLayer):
        self.n_macs = 0
        self.n_params = 0
        self.n_seconds = 0
        add_sub(self, self.multihead_attn)
        add_sub(self, self.linear1)
        add_sub(self, self.linear2)
        self.name = self.__repr__()
    elif isinstance(self, nn.MultiheadAttention):
        self.n_macs = 0
        self.n_params = 0
        self.n_seconds = 0
        add_sub(self, self.out_proj)
        self.n_macs += 2 * input[0].shape[0] * input[1].shape[0] * input[0].shape[2] + \
            4 * input[0].shape[0] * input[0].shape[2] * input[0].shape[2]
        self.name = self.__repr__()
    elif isinstance(self, hrb.HighResolutionModule):
        self.n_macs = 0
        self.n_params = 0
        self.n_seconds = 0
        for op in self.branches:
            add_sub(self, op)
        for op in self.fuse_layers:
            add_sub(self, op)
        self.name = self.__repr__()
    elif isinstance(self, hrb.BasicBlock):
        self.n_macs = 0
        self.n_params = 0
        self.n_seconds = 0
        add_sub(self, self.conv1)
        if self.downsample is not None:
            add_sub(self, self.downsample)
        self.name = self.__repr__()
    elif isinstance(self, hrb.Bottleneck):
        self.n_macs = 0
        self.n_params = 0
        self.n_seconds = 0
        add_sub(self, self.conv1)
        add_sub(self, self.conv2)
        add_sub(self, self.conv3)
        if self.downsample is not None:
            add_sub(self, self.downsample)
        self.name = self.__repr__()
    else:
        # This works only in depth-first travel of modules.
        self.n_macs = 0
        self.n_params = 0
        self.n_seconds = 0
        num_children = 0
        for m in self.children():
            self.n_macs += getattr(m, 'n_macs', 0)
            self.n_params += getattr(m, 'n_params', 0)
            self.n_seconds += getattr(m, 'n_seconds', 0)
            num_children += 1
        ignore_zeros_t = [
            cnn.BatchNorm2d,
            nn.LayerNorm,
            cnn.Dropout2d,
            nn.Dropout,
            nn.Sequential,
            nn.ReLU6,
            nn.ReLU,
            mb.Swish,
            mb.Narrow,
            mb.Identity,
            cnn.MaxPool2d,
            nn.modules.padding.ZeroPad2d,
            cnn.Sigmoid,
        ]
        if (not getattr(self, 'ignore_model_profiling', False) and
                self.n_macs == 0 and t not in ignore_zeros_t):
            if udist.is_master():
                logging.info('WARNING: leaf module {} has zero n_macs.'.format(
                    type(self)))
        return
    if verbose:
        if udist.is_master():
            logging.info(
                self.name.ljust(name_space, ' ') +
                '{:,}'.format(self.n_params).rjust(params_space, ' ') +
                '{:,}'.format(self.n_macs).rjust(macs_space, ' ') +
                '{:,}'.format(self.n_seconds).rjust(seconds_space, ' '))
    return


def add_profiling_hooks(m, num_forwards, verbose):
    global model_profiling_hooks
    model_profiling_hooks.append(
        m.register_forward_hook(lambda m, input, output: module_profiling(
            m, input, output, num_forwards, verbose=verbose)))


def remove_profiling_hooks():
    global model_profiling_hooks
    for h in model_profiling_hooks:
        h.remove()
    model_profiling_hooks = []


def model_profiling(model,
                    height,
                    width,
                    batch=1,
                    channel=3,
                    use_cuda=True,
                    num_forwards=10,
                    verbose=True):
    """ Pytorch model profiling with input image size
    (batch, channel, height, width).
    The function exams the number of multiply-accumulates (n_macs).

    Args:
        model: pytorch model
        height: int
        width: int
        batch: int
        channel: int
        use_cuda: bool

    Returns:
        macs: int
        params: int

    """
    model.eval()
    data = crypten.rand(batch, channel, height, width)
    origin_device = next(model.parameters()).device
    #device = torch.device('mps')#"cuda" if use_cuda else "cpu")
    #model = model.to(device)
    #data = data.to(device)
    model.apply(lambda m: add_profiling_hooks(m, num_forwards, verbose=verbose))
    if verbose:
        logging.info('Item'.ljust(name_space, ' ') +
                     'params'.rjust(macs_space, ' ') +
                     'macs'.rjust(macs_space, ' ') +
                     'nanosecs'.rjust(seconds_space, ' '))
        logging.info(''.center(
            name_space + params_space + macs_space + seconds_space, '-'))
    with crypten.no_grad():
        model(data)
    if verbose:
        logging.info(''.center(
            name_space + params_space + macs_space + seconds_space, '-'))
        logging.info('Total'.ljust(name_space, ' ') +
                     '{:,}'.format(model.n_params).rjust(params_space, ' ') +
                     '{:,}'.format(model.n_macs).rjust(macs_space, ' ') +
                     '{:,}'.format(model.n_seconds).rjust(seconds_space, ' '))
    remove_profiling_hooks()
    model = model.to(origin_device)
    return model.n_macs, model.n_params


def profiling(model, use_cuda):
    """Profiling on either gpu or cpu."""
    if udist.is_master():
        logging.info('Start model profiling, use_cuda:{}.'.format(use_cuda))
    model_profiling(model,
                    28,
                    28,
                    verbose=True)



class DummyNetwork(cnn.Module):

    def __init__(self):
        super(DummyNetwork, self).__init__()
        self.conv1 = cnn.Conv2d(1, 16, 3, padding=1)  # Increased output channels and adjusted kernel size
        self.conv2 = cnn.Conv2d(16, 32, 3, padding=1)  # Added another convolutional layer
        self.pool = cnn.MaxPool2d(2, 2)
        self.relu = cnn.ReLU()
        #self.act1 = mb.SqueezeAndExcitation(2,2, active_fn=cnn.ReLU)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        #x = self.act1(x)
        return x


if __name__ == "__main__":
    # Prepare crypten
    crypten.init()

    # Create the network and encrypt it
    fix_hook(cnn.Module)  
    dummy_network = DummyNetwork().encrypt()
    dummy_input = crypten.randn(1, 1, 28, 28)
    # Encrypt it and add the profiling thingies
    
    #dummy_network.forward(dummy_input)
    #a = cnn.from_pytorch(dummy_network, dummy_input)
    model_profiling(dummy_network, 28, 28)
