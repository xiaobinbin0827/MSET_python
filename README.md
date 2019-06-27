# MSET_python 
## Python implementation of multivariate state estimation technology 
该项目为多元状态估计技术的python实现，主要包含`训练与测试数据`（.mat文件）、`模型`（Model.py）以及`测试例子`（example.py）<br>
其中`ae_D_temp`为训练数据，`ae_Kobs3_temp`为正常测试数据，`ae_ver_temp`为磨煤机堵煤故障数据，数据集包含风粉混合物温度等14个变量。`Model.py`包含以下函数：<br>
* Traindata(name_list,if_nor=True)  
	>#输入文件名列表，加载训练数据，默认对数据进行归一化<br>
* Testdata(name_string,np_Dmax,np_Dmin,if_nor=True)  
	>#加载测试数据，默认进行归一化<br>
* Faultdata(name_string,np_Dmax,np_Dmin,if_nor=True)  
	>#加载故障数据，默认进行归一化<br>
* normalization(np_Kobs,np_Dmax,np_Dmin)  
	>#归一化模块，适合用于需对原始数据进行操作而未进行归一化的数据<br>
* MemoryMat_train(np_D,memorymat_name)  
	>#模型训练，返回记忆矩阵<br>
* MemoryMats_train(np_D)  
	>#多模型训练，分高中低负荷建立模型，返回三个记忆矩阵<br>
* MemoryMats_train(np_D)  
	>#计算保存记忆矩阵的Temp矩阵，在每次输入观测向量，计算估计值时，可直接加载Temp，减少运算量<br>
* MSET(memorymat_name,Kobs,Temp_name)  
	>#输入记忆矩阵、观测向量以及Temp矩阵，返回对应的估计向量<br>
* MSETs(memorymat1_name,memorymat2_name,memorymat3_name,Kobs)  
	>#对于输入的观测向量先划分负荷段，再传到相应记忆矩阵中，得到估计值<br>
* Cal_sim(Kobs,Kest)  
	>#基于融合距离的相似度计算<br>
* Cal_thres(sim)  
	>#根据区间统计的思想确定动态阈值，动态阈值的更新与当前时刻及前一时刻的相似度均值、方差相关<br>
	>PS.为了避免在故障发生时阈值随相似度一直下降发生漏报，在阈值更新的触发条件上引入新的限制，即：相似度大于动态阈值且大于0.8，更新动态阈值；相似度小于动态阈值或相似度大于动态阈值且小于0.8，则不更新   
* pic_vars(label,Kobs,Kest,np_Dmax,np_Dmin)  
	>#各变量及其误差的可视化<br>
* error_contribution(Kobs,Kest,momtent,label)  
	>#误差贡献率<br>
* Accumu_errorContirbution(Kobs,Kest,momtent,time_range,label)  
	>#累计误差贡献率<br>
* Mat_update(Kobs,sim,thres,memorymat_name,Temp_name)  
	>#更新记忆矩阵，目前该函数未可用<br>
