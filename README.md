# 警告
crn lip_k还是有问题，没有考虑到rnn后的线性层的影响


# 代码结构
* LoadData.py
    * 加载csv，以及一些格式的转换
* CRN_lightning_model.py, CRN_model.py, LipLSTM.py
    * crn部分的模型
* AStarSearch.py, SearchAdapter.py, SearchApp.py
    * 搜索部分的代码
* test.py
    * 主函数在这

# 可能存在的问题

## 效率上的问题
* SearchApp.py中getfunc一部分中，转换函数需要借助L.Trainer和Dataloader，但是目前的数据集的大小为1，感觉这样效率不高，或许直接用crn_model (torch.nn.Module)效率会更高一些
* AStarSearch.py 中是用python写的搜索，但是我感觉瓶颈不在这，因为“状态转换”（需要用网络来推理）的次数太多了
* A\*搜索的“状态转换”需要借助decoder，但是目前的decoder使用lstm来实现的，这样导致SearchApp.py中得到的lval太大了，使得对于A\*搜索中对未来的估计太不准了，感觉很影响效率，感觉可以修改lstm的实现，或者换成朴素的rnn，或者不用rnn（hidden state / balancing representation 保持不变），但是感觉这样可能缺少卖点了（？）
## 其他
* 对于数据的处理略微有点问题（哪些数据该当成treatment，哪些该当成检验）