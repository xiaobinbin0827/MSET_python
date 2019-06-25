import numpy as np
#模拟四种传感器故障
#卡死
def Stuck(Kobs,Var_number,Initial_moment,a):
    Kobs[Initial_moment-1:,Var_number-1]=a
    return Kobs
#恒增益
def ConstantGain(Kobs,Var_number,Initial_moment,G):
    Kobs[Initial_moment-1:, Var_number-1] = (1+G)*Kobs[Initial_moment-1:,Var_number-1]
    return Kobs
#固定偏差
def FixedDeviation(Kobs,Var_number,Initial_moment,b):
    Kobs[Initial_moment-1:, Var_number-1] = b+ Kobs[Initial_moment-1:, Var_number-1]
    return Kobs
#线性偏差
def Linear_deviation(Kobs,Var_number,Initial_moment,f,c):
    line=(f * (np.arange(0, Kobs.shape[0]-Initial_moment + 1, step=1)) + c)
    Kobs[Initial_moment-1:, Var_number-1]=line+Kobs[Initial_moment-1:, Var_number-1]
    return Kobs


