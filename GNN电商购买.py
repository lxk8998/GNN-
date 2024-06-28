import torch.optim
import torch_geometric
from torch_geometric.datasets import KarateClub
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data

from torch_geometric.nn import TopKPooling,SAGEConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


url=r'E:\pytorch geometric\创建自己的数据集\创建自己的数据集'

df = pd.read_csv(f'{url}'+r'\yoochoose-clicks.dat', header=None)
df.columns=['session_id','timestamp','item_id','category']
#print(df.iloc[0:2])

buy_df = pd.read_csv(f'{url}'+r'\yoochoose-buys.dat', header=None)
buy_df.columns=['session_id','timestamp','item_id','price','quantity']

#随机采样部分数据
sampled_session_id = np.random.choice(df.session_id.unique(), 100000, replace=False)
df = df.loc[df.session_id.isin(sampled_session_id)]
#print(df.nunique())

#标签,将标签合并入df,每个图对应一个标签
df['label'] = df.session_id.isin(buy_df.session_id)

df_test=df[:50]
print(df_test)

#每个data是一张图
data_list=[]

#groupby('session_id'),按照session_id对表格进行分组
grouped = df_test.groupby('session_id')
for session_id, group in grouped:
    #print(session_id)
    #print(group)

    #返回一个列表，对item_id按照大小赋值为0，1...
    sess_item_id = LabelEncoder().fit_transform(group.item_id)
    #print('sess_item_id:', sess_item_id)

    #将group的索引重置为默认值
    group = group.reset_index(drop=True)

    #将新编码的sess_item_id列添加到group中
    group['sess_item_id'] = sess_item_id
    #print('group:', group)

    #drop_duplicates()去掉item_id重复项
    node_features = group.loc[:, ['sess_item_id', 'item_id']].sort_values(
        'sess_item_id').item_id.drop_duplicates().values

    node_features = torch.LongTensor(node_features).unsqueeze(1)  # unsqueeze:指定的位置插入一个维度
    #print(node_features)
    #取出target和source
    target_nodes = group.sess_item_id.values[1:]
    source_nodes = group.sess_item_id.values[:-1]
    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)

    x = node_features
    y = torch.FloatTensor([group.label.values[0]])
    #print(y)
    data = Data(x=x, edge_index=edge_index, y=y)
    print(data.x)
    print(data.y)
    print(data.batch)
    data_list.append(data)

embed_dim = 128

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SAGEConv(embed_dim,128)
        self.pool1 = TopKPooling(128,ratio=0.8) #只保留0.8比例的节点，向量的维度是不变的
        self.conv2 = SAGEConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.8)
        self.conv3 = SAGEConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=0.8)
        #num_embeddings表示输入tensor的one_hot编码的维度，以最大的id为维度
        self.item_embedding = torch.nn.Embedding(num_embeddings=df.item_id.max() + 10, embedding_dim=embed_dim)
        self.lin1 = torch.nn.Linear(128, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, 1)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()

    def forward(self,data):
        x, edge_index, batch = data.x, data.edge_index, data.batch  # x:n*1,其中每个图里点的个数是不同的,n为node_features的个数
        # print(x)
        x = self.item_embedding(x)  # n*1*128 特征编码后的结果(seqlen,batchsize,embeddingsize)
        # print('item_embedding',x.shape)
        x = x.squeeze(1)  # n*128，去掉了batchsize
        # print('squeeze',x.shape)
        x = F.relu(self.conv1(x, edge_index))  # n*128
        # print('conv1',x.shape)
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)  # pool之后得到 n*0.8个点
        # print('self.pool1',x.shape)
        # print('self.pool1',edge_index)
        # print('self.pool1',batch)
        # x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x1 = gap(x, batch) #针对一个图中的每个节点的128维向量加起来再平均,batch*128
        # print('gmp',gmp(x, batch).shape) # batch*128
        # print('cat',x1.shape) # batch*256
        x = F.relu(self.conv2(x, edge_index))
        # print('conv2',x.shape)
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        # print('pool2',x.shape)
        # print('pool2',edge_index)
        # print('pool2',batch)
        # x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x2 = gap(x, batch)
        # print('x2',x2.shape)
        x = F.relu(self.conv3(x, edge_index))
        # print('conv3',x.shape)
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        # print('pool3',x.shape)
        # x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x3 = gap(x, batch)
        # print('x3',x3.shape)# batch * 256
        x = x1 + x2 + x3  # 获取不同尺度的全局特征

        x = self.lin1(x)
        # print('lin1',x.shape)
        x = self.act1(x)
        x = self.lin2(x)
        # print('lin2',x.shape)
        x = self.act2(x)
        #self.training返回Module是否处于训练状态,也就是说在训练时training就是True,
        #直接将self.training传入函数，就可以在训练时应用dropout，评估时关闭dropout
        x = F.dropout(x, p=0.5, training=self.training) #training 为false的时候,dropout不起作用,默认情况下training是True
        #.squeeze(1)操作将结果的维度从(batch_size, 1)变为(batch_size,)
        x = torch.sigmoid(self.lin3(x)).squeeze(1)  # batch个结果,
        # print('sigmoid',x.shape)
        return x




model = Net()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.BCELoss() #二分类问题
