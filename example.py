from Model import *
from Fault_modify import *

N=60#每次输入60个观测向量
label1=('风粉混合物温度/℃','反作用力加载油压/MPa','加载油压/MPa',
       '磨煤机电流/A','一次风压力/kPa','密封风母管压力/kPa',
       '一次风与密封风差压/kPa', '出入口差压/kPa','油箱油温/℃',
       '一次风流量/t·h-1','轴承温度/℃','推力瓦温/℃',
       '油池油温/℃','实际功率/MW')
label2=('1风粉混合物温度','2反作用力加载油压','3加载油压',
       '4磨煤机电流','5一次风压力','6密封风母管压力',
       '7一次风与密封风差压', '8出入口差压','9油箱油温',
       '10一次风流量','11轴承温度','12推力瓦温',
       '13油池油温','14实际功率')

name_list=['ae_D0_temp','ae_D1_temp','ae_D2_temp']#数据文件名列表
np_D,np_Dmax,np_Dmin=Traindata(name_list,if_nor=True)#加载训练集
# memorymat=MemoryMat_train(np_D,'memorymat1.npy')#训练得到记忆矩阵
# Temp_MemMat(memorymat,'Temp1.npy')#保存MSET计算用的临时矩阵
# Kobs=Testdata('ae_Kobs3_temp',np_Dmax,np_Dmin,if_nor=True) #加载测试集
Kobs=Faultdata('ae_ver_temp',np_Dmax,np_Dmin,if_nor=True) #加载故障集
Kobs=Kobs[0:1140,:]#故障集共1185个点，取前1140个，被60整除
# 将观测值与估计值输入值循环输入得到
sim=np.zeros((Kobs.shape[0],1))
thres=np.zeros((Kobs.shape[0],1))
Kest=np.zeros((Kobs.shape[0],Kobs.shape[1]))
for i in range(int(Kobs.shape[0]/N)):
    # 加载记忆矩阵与临时矩阵，输入观测向量，计算对应估计向量
    Kest[i*N:(i+1)*N] = MSET(memorymat_name='memorymat1.npy',
                Kobs=Kobs[i*N:(i+1)*N], Temp_name='Temp1.npy')
    sim[i*N:(i+1)*N]=Cal_sim(Kobs[i*N:(i+1)*N],Kest[i*N:(i+1)*N])
    thres[i*N:(i+1)*N],warning_index=Cal_thres(sim[i*N:(i+1)*N])
    if any(warning_index):
        #如果故障索引存在值，打印该点编号并显示误差贡献率
        print('第%d次循环中的故障点：' % (i + 1), warning_index)
        error_contribution(Kobs[i*N:(i+1)*N],Kest[i*N:(i+1)*N],warning_index[0],label2)
        Accumu_errorContirbution(Kobs[i*N:(i+1)*N], Kest[i*N:(i+1)*N]
                                 ,warning_index[0], N-warning_index[0],label2)
 # 画图
pic_vars(label1,Kobs,Kest,np_Dmax,np_Dmin)#各变量的估计结果及误差
plt.ion()
plt.plot(sim,label='相似度曲线')
plt.plot(thres,label='动态阈值')
plt.ylim((0,1))
plt.legend()
plt.show()
