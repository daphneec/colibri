import pandas as pd
from joblib import load

LR_folderpath = "./utils/LR_model/"
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