import pandas as pd
from joblib import load

LR_folderpath = "./utils/LR_model/powerful/"
name = "client" # Only considering the client side estimation

def conv_time_cal(ins, outs, kernel_size):
    X = pd.DataFrame()

    # ins:[N, CI, HI, WI]
    # outs:[N, CO, HO, WO]
    # kernel_size: [FH, FW]
    
    # flops: 2 * FH * FW * CI * HO * WO * CO
    X['FLOPs'] = [(2 * kernel_size[0] * kernel_size[1] * ins[1]) * outs[2] * outs[3] *outs[1]]
    # In_macs: HI * WI * CI
    X['IN_MACs'] = [(ins[2] * ins[3] * ins[1]) * 8]
    # Par_macs: CI * FH * FW * CO
    X['PAR_MACs'] = [(ins[1] * kernel_size[0] * kernel_size[1] * outs[1]) * 8]
    # Out_macs: HO * WO * CO
    X['OUT_MACs'] = [(outs[2] * outs[3] * outs[1]) * 8]

    reg = load(LR_folderpath + 'conv_' + name + '_LR_model.joblib')
    scaler = load(LR_folderpath + 'conv_' + name + '_scaler.joblib')

    X_normalized = scaler.transform(X)
    predicted_time = reg.predict(X_normalized)

    return max(round(predicted_time[0]), 0)

def linear_time_cal(ins, outs):
    X = pd.DataFrame()

    # ins:[N, CI]
    # outs:[N, CO]
    
    # flops: (2 * CI * CO) + CO
    X['FLOPs'] = [(2 * ins[1] * outs[1]) + outs[1]]
    # In_macs: CI * 8
    X['IN_MACs'] = [ins[1] * 8]
    # Par_macs: (CI + 1) * CO * 8
    X['PAR_MACs'] = [(ins[1] + 1) * outs[1] * 8]
    # Out_macs: CO * 8
    X['OUT_MACs'] = [outs[1] * 8]

    reg = load(LR_folderpath + 'fc_' + name + '_LR_model.joblib')
    scaler = load(LR_folderpath + 'fc_' + name + '_scaler.joblib')

    X_normalized = scaler.transform(X)
    predicted_time = reg.predict(X_normalized)

    return max(round(predicted_time[0]), 0)

def relu_time_cal(ins, outs):
    X = pd.DataFrame()

    # ins:[N, CI, HI, WI]
    # outs:[N, CO, HO, WO]
    
    # flops: N * HI * WI * CI
    X['FLOPs'] = [ins[0] * ins[1] * ins[2] * ins[3]]

    reg = load(LR_folderpath + 'relu_' + name + '_LR_model.joblib')
    scaler = load(LR_folderpath + 'relu_' + name + '_scaler.joblib')

    X_normalized = scaler.transform(X)
    predicted_time = reg.predict(X_normalized)

    return max(round(predicted_time[0]), 0)

def bn_time_cal(ins, outs):
    X = pd.DataFrame()

    # ins:[N, CI, HI, WI]
    # outs:[N, CO, HO, WO]

    # flops: CI * HI * WI
    X['FLOPs'] = [ins[1] * ins[2] * ins[3]]
    # In_macs: CI * HI * WI
    X['IN_MACs'] = [ins[1] * ins[2] * ins[3]]
    # Par_macs: CI
    X['PAR_MACs'] = [ins[1]]
    # Out_macs: CI * HI * WI
    X['OUT_MACs'] = [ins[1] * ins[2] * ins[3]]

    reg = load(LR_folderpath + 'bn_' + name + '_LR_model.joblib')
    scaler = load(LR_folderpath + 'bn_' + name + '_scaler.joblib')

    X_normalized = scaler.transform(X)
    predicted_time = reg.predict(X_normalized)

    return max(round(predicted_time[0]), 0)

def avgpool_time_cal(ins, outs, kernel_size):
    X = pd.DataFrame()

    # ins:[N, CI, HI, WI]
    # outs:[N, CO, HO, WO]
    # kernel_size: F

    # flops: F * F * HO * WO * CO
    X['FLOPs'] = [(kernel_size * kernel_size) * (outs[2] * outs[3]) * outs[1]]
    # In_macs: HI * WI * CI * 8
    X['IN_MACs'] = [(ins[2] * ins[3]) * ins[1] * 8]
    # Out_macs: HO * WO * CO * 8
    X['OUT_MACs'] = [(outs[2] * outs[3]) * outs[1] * 8]

    reg = load(LR_folderpath + 'ap_' + name + '_LR_model.joblib')
    scaler = load(LR_folderpath + 'ap_' + name + '_scaler.joblib')

    X_normalized = scaler.transform(X)
    predicted_time = reg.predict(X_normalized)

    return max(round(predicted_time[0]), 0)