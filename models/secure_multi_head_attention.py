#!/usr/bin/env python
# SECURE MULTI HEAD ATTENTION.py
#   by Lut99
#
# Created:
#   11 Jan 2024, 15:41:39
# Last edited:
#   22 Jan 2024, 11:41:26
# Auto updated?
#   Yes
#
# Description:
#   Implements a Crypten-compatible multi-head attention module (hopefully).
#   
#   Code taken from MPC Former (https://github.com/DachengLi1/MPCFormer/blob/main/transformers/src/transformers/models/ctrl/modeling_ctrl.py)
#   and adapted to use with Crypten.
#

import typing

import crypten
import crypten.nn as cnn
import numpy as np
import torch


##### TESTS #####
def test_multi_head_attention():
    """
        Implements a unit test for the multi-head attention.
    """

    import torch.nn as nn
    crypten.init()

    # Setup some random layers
    gold_layer = nn.MultiheadAttention(6, 2, dropout=0.2)
    cryp_layer = MultiHeadAttention(6, 2, dropout=0.2).encrypt()

    # Let's compare modules
    for m in gold_layer.modules():
        print(f"gold : {m}")
    for m in cryp_layer.modules():
        print(f"cryp : {m}")

    # Initialze random input and a Crypten-shared counterpart
    query, key, value = torch.tensor(np.random.rand(6, 6), dtype=torch.float32), torch.tensor(np.random.rand(6, 6), dtype=torch.float32), torch.tensor(np.random.rand(6, 6), dtype=torch.float32)
    cquery, ckey, cvalue = crypten.cryptensor(query), crypten.cryptensor(key), crypten.cryptensor(value)

    # Run it thru the layers
    gold1, _ = gold_layer.forward(query, key, value)
    cryp1, _, _ = cryp_layer.forward(cquery, ckey, cvalue, output_attentions=True)
    cryp1 = cryp1.get_plain_text()

    # Print some comparisons
    print("Gold attention outputs:\n" + (80 * "-") + "\n" + str(gold1) + "\n" + (80 * "-") + "\n\nCrypten attention outputs\n" + (80 * "-") + "\n" + str(cryp1) + "\n" + (80 * "-") + "\n")





##### HELPER FUNCTIONS #####
def find_pruneable_heads_and_indices(
    heads: typing.List[int], n_heads: int, head_size: int, already_pruned_heads: typing.Set[int]
) -> typing.Tuple[typing.Set[int], typing.Union[torch.LongTensor, crypten.CrypTensor]]:
    """
    Finds the heads and their indices taking `already_pruned_heads` into account.

    Args:
        heads (`List[int]`): List of the indices of heads to prune.
        n_heads (`int`): The number of heads in the model.
        head_size (`int`): The size of each head.
        already_pruned_heads (`Set[int]`): A set of already pruned heads.

    Returns:
        `Tuple[Set[int], torch.LongTensor]`: A tuple with the remaining heads and their corresponding indices.
    """

    heads = set(heads) - already_pruned_heads  # Convert to set and remove already pruned heads
    # mask = crypten.ones_like(n_heads, head_size)
    mask = set([[1 for _ in range(head_size)] for _ in range(n_heads)])
    for head in heads:
        # Compute how many pruned heads are before the head and move the index accordingly
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index = crypten.arange(len(mask))[mask].long()
    return heads, index



def prune_linear_layer(layer: cnn.Linear, index: typing.Union[torch.LongTensor, crypten.CrypTensor], dim: int = 0) -> cnn.Linear:
    """
    Prune a linear layer to keep only entries in index.

    Used to remove heads.

    Args:
        layer (`crypten.nn.Linear`): The layer to prune.
        index (`crypten.CrypTensor` or `torch.Tensor`): The indices to keep in the layer.
        dim (`int`, *optional*, defaults to 0): The dimension on which to keep the indices.

    Returns:
        `crypten.nn.Linear`: The pruned layer as a new layer with `requires_grad=True`.
    """
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = cnn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer



def scaled_dot_product_attention(q, k, v, dropout_p=0.0, attention_mask=None, head_mask=None):
    # Decide if we're running Crypten in encrypted mode
    is_encrypted = isinstance(q, crypten.CrypTensor)

    # calculate attention
    if isinstance(q, crypten.CrypTensor):
        matmul_qk = q.matmul(k.permute([0, 1, 3, 2]))
    else:
        matmul_qk = torch.matmul(q, k.permute(0, 1, 3, 2))

    dk = k.shape[-1]
    scaled_attention_logits = matmul_qk / np.sqrt(dk)

    # if mask is not None:
    #     nd, ns = scaled_attention_logits.size(-2), scaled_attention_logits.size(-1)
    #     scaled_attention_logits += mask[ns - nd : ns, :ns] * -1e4

    if attention_mask is not None:
        # Apply the attention mask
        scaled_attention_logits = scaled_attention_logits + attention_mask

    softmax = cnn.Softmax(-1)
    if is_encrypted:
        softmax = softmax.encrypt()
    attention_weights = softmax.forward(scaled_attention_logits)

    # Mask heads if we want to
    # ...we can leave our friends behind, because if they don't dance and if they don't dance they are no friends of mine
    if head_mask is not None:
        attention_weights = attention_weights * head_mask

    # Take our chance and apply the dropout!
    dropout = cnn.Dropout(dropout_p, False)
    if is_encrypted:
        dropout = dropout.encrypt()
    attention_weights = dropout.forward(attention_weights)
    if isinstance(attention_weights, crypten.CrypTensor):
        output = attention_weights.matmul(v)
    else:
        output = crypten.matmul(attention_weights, v)

    return output, attention_weights





##### LIBRARY #####
class MultiHeadAttention(cnn.Module):
    def __init__(self, d_model_size, num_heads, dropout=0.0, training=True):
        super().__init__()
        self.num_heads = num_heads
        self.d_model_size = d_model_size

        self.depth = int(d_model_size / self.num_heads)

        self.Wq = cnn.Linear(d_model_size, d_model_size)
        self.Wk = cnn.Linear(d_model_size, d_model_size)
        self.Wv = cnn.Linear(d_model_size, d_model_size)

        self.dense = cnn.Linear(d_model_size, d_model_size)
        self.pruned_heads = set()

        self.dropout_p = dropout if training else 0.0

    def prune_heads(self, heads):
        attention_head_size = self.d_model_size // self.num_heads
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.num_heads, attention_head_size, self.pruned_heads)

        # Prune linear layers
        self.Wq = prune_linear_layer(self.Wq, index)
        self.Wk = prune_linear_layer(self.Wk, index)
        self.Wv = prune_linear_layer(self.Wv, index)
        self.dense = prune_linear_layer(self.dense, index, dim=1)

        # Update hyper params
        self.num_heads = self.num_heads - len(heads)
        self.d_model_size = attention_head_size * self.num_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def split_into_heads(self, x, batch_size):
        x = x.reshape(batch_size, -1, self.num_heads, self.depth)
        return x.permute([0, 2, 1, 3])

    def forward(
        self,
        q,
        k,
        value,
        layer_past=None,
        attn_mask=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        batch_size = q.shape[0]

        q = self.Wq(q)
        k = self.Wk(k)
        value = self.Wv(value)

        q = self.split_into_heads(q, batch_size)
        k = self.split_into_heads(k, batch_size)
        value = self.split_into_heads(value, batch_size)
        if layer_past is not None:
            past_key, past_value = layer_past[0], layer_past[1]
            k = crypten.cat((past_key, k), dim=-2)
            value = crypten.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = crypten.stack((k, value))
        else:
            present = (None,)

        output = scaled_dot_product_attention(q, k, value, self.dropout_p, attn_mask, head_mask)
        scaled_attention = output[0].permute([0, 2, 1, 3])
        attn = output[1]
        original_size_attention = scaled_attention.reshape(batch_size, -1, self.d_model_size)
        output = self.dense(original_size_attention)

        outputs = (output, present)
        if output_attentions:
            outputs = outputs + (attn,)

        return outputs





##### ENTRYPOINT #####
if __name__ == "__main__":
    test_multi_head_attention()
