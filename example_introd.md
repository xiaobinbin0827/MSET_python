example.py介绍  
=====
* 该文件实现了MSET的全流程，包括加载数据、训练数据、计算观测向量的估计值、相似度  
  >部分代码：
  ```
  name_list=['ae_D0_temp','ae_D1_temp','ae_D2_temp']#数据文件名列表  
  np_D,np_Dmax,np_Dmin=Traindata(name_list,if_nor=True)#加载训练集  
  # memorymat=MemoryMat_train(np_D,'memorymat1.npy')#训练得到记忆矩阵  
  # Temp_MemMat(memorymat,'Temp1.npy')#保存MSET计算用的临时矩阵  
  # Kobs=Testdata('ae_Kobs3_temp',np_Dmax,np_Dmin,if_nor=True) #加载测试集  
  Kobs=Faultdata('ae_ver_temp',np_Dmax,np_Dmin,if_nor=True) #加载故障集  
  ```
  ```
  for i in range(int(Kobs.shape[0]/N)):
    Kest[i*N:(i+1)*N] = MSET(memorymat_name='memorymat1.npy',  
                  Kobs=Kobs[i*N:(i+1)*N], Temp_name='Temp1.npy')  
    sim[i*N:(i+1)*N]=Cal_sim(Kobs[i*N:(i+1)*N],Kest[i*N:(i+1)*N])  
    thres[i*N:(i+1)*N],warning_index=Cal_thres(sim[i*N:(i+1)*N])  
  ```
  ```
  pic_vars(label1,Kobs,Kest,np_Dmax,np_Dmin)#各变量的估计结果及误差  
  ```
  -----
* 对于故障数据，在监测到故障时给出对应的时刻点，并显示误差贡献图与累计误差贡献图  
  >部分代码：
  ```
  if any(warning_index):
        #如果故障索引存在值，打印该点编号并显示误差贡献率
        print('第%d次循环中的故障点：' % (i + 1), warning_index)
        error_contribution(Kobs[i*N:(i+1)*N],Kest[i*N:(i+1)*N],warning_index[0],label2)
        Accumu_errorContirbution(Kobs[i*N:(i+1)*N], Kest[i*N:(i+1)*N],warning_index[0], N-warning_index[0],label2)
  ```
  ------
* 输入磨煤机堵煤故障数据，部分结果如下：
  >![figure 1](https://github.com/xiaobinbin0827/MSET_python/blob/master/img-folder/sim.png)
  >![figure 2](https://github.com/xiaobinbin0827/MSET_python/blob/master/img-folder/error_contribution.png)
  >![figure 3](https://github.com/xiaobinbin0827/MSET_python/blob/master/img-folder/Acc_errorContribution.png)
