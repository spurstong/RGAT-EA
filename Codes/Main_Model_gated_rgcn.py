# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import sys
import dgl
from dgl import DGLGraph
from dgl.data import MiniGCDataset
import dgl.function as fn
from spacy.tokens import Doc

from functools import partial

import re
import numpy as np
import pandas as pd

from pytorch_pretrained_bert import BertTokenizer

import spacy
import pickle
import collections
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt

import os

# 设置当前使用的GPU设备为0，1号两个设备，名称依次为‘/gpu:0’, '/gpu:1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# 设置指定的GPU设备
torch.cuda.set_device(0)
device = torch.device("cuda:0")

# tokens of every sentence without padding
token_lst = pickle.load(open('token_lst_wto_padding.pkl', "rb"))
# list of outputs of bert for every sentence
bert_outputs_lst = pickle.load(open('bert_outputs.pkl', "rb"))
# the positions of A, B, and the pronoun  [语料数量，3]
offsets_lst = pickle.load(open('offsets_lst.pkl', "rb"))
# tokens of every sentence without padding
test_token_lst = pickle.load(open('test_token_lst_wto_padding.pkl', "rb"))
# list of outputs of bert for every sentence
test_bert_outputs_lst = pickle.load(open('test_bert_outputs.pkl', "rb"))
test_offsets_lst = pickle.load(open('test_offsets_lst.pkl', "rb"))
# [句子数量，1， 3, hidden_size]
others_bert_outputs = pickle.load(open('others_bert_outputs.pkl', "rb"))
test_others_bert_outputs = pickle.load(open('test_others_bert_outputs.pkl', "rb"))

train_df = pd.read_csv("timebank-dense-train.csv")

test_df = pd.read_csv("timebank-dense-test.csv")


class RGCNLayer(nn.Module):
    def __init__(self, feat_size, num_rels, activation=None, gated=True):

        super(RGCNLayer, self).__init__()
        # h_dim = 1024
        self.feat_size = feat_size
        # num_rels = 3 边类型数量
        self.num_rels = num_rels
        self.activation = activation
        self.gated = gated
        # 论文里的每一层的边类型权重矩阵，Wr
        # (3, hidden_size=1024, 256)
        #self.weight = nn.Parameter(torch.Tensor(self.num_rels, self.feat_size, 256).float())
        self.weight = nn.Parameter(torch.Tensor(self.num_rels, self.feat_size, 256).float())

        # init trainable parameters 均匀分布
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

        if self.gated:
            # (3, hidden_size=1024, 1)
            self.gate_weight = nn.Parameter(torch.Tensor(self.num_rels, self.feat_size, 1).float())
            nn.init.xavier_uniform_(self.gate_weight, gain=nn.init.calculate_gain('sigmoid'))

    def forward(self, g):

        weight = self.weight
        gate_weight = self.gate_weight

        # 节点信息更新
        def message_func(edges):
            w = weight[edges.data['rel_type']]
            # 状态h的初始向量维度为(1, hidden_size)
            # squeeze() 将所有为1的维度删掉 (256)
            # edges.src['h'].unsqueeze(1)是 [边的个数，1， hidden_size]
            # torch.bmm(edges.src['h'].unsqueeze(1), w) 是[边的个数，1， 256]
            # mag [边的个数, 256]
            msg = torch.bmm(edges.src['h'].unsqueeze(1), w).squeeze()
            # edges.data['norm'] [边的个数，1]
            # msg [边的个数，256]
            msg = msg * edges.data['norm']

            if self.gated:
                # gate_w -> [边的个数, hidden_size, 1]
                gate_w = gate_weight[edges.data['rel_type']]
                # edges.src['h'].unsqueeze(1) -> [边的个数, 1, hidden_size]
                # torch.bmm(edges.src['h'].unsqueeze(1), gate_w) -> [边的个数,1,1]
                # .squeeze() -> [边的个数]
                # .reshape(-1,1) -> [边的个数，1]
                gate = torch.bmm(edges.src['h'].unsqueeze(1), gate_w).squeeze().reshape(-1, 1)
                # [边的个数, 1] 将变量映射成0到1的值
                gate = torch.sigmoid(gate)
                # 【边的个数， 256】
                msg = msg * gate

            return {'msg': msg}


        def apply_func(nodes):
            h = nodes.data['h']
            h = self.activation(h)
            return {'h': h}

        g.update_all(message_func, fn.sum(msg='msg', out='h'), apply_func)



# ## Define Full RGCN Model

# In[ ]:


class RGCNModel(nn.Module):
    def __init__(self, h_dim, num_rels, num_hidden_layers=1, gated=False):
        super(RGCNModel, self).__init__()
        # h_dim = 1024
        self.h_dim = h_dim
        # num_rels = 3
        self.num_rels = num_rels
        self.num_hidden_layers = num_hidden_layers
        self.gated = gated

        # create rgcn layers
        self.build_model()

    def build_model(self):
        self.layers = nn.ModuleList()
        for _ in range(self.num_hidden_layers):
            rgcn_layer = RGCNLayer(self.h_dim, self.num_rels, activation=F.relu, gated=self.gated)
            self.layers.append(rgcn_layer)

    def forward(self, g):
        for layer in self.layers:
            layer(g)

        rst_hidden = []
        # Return the list of graphs in this batch.
        for sub_g in dgl.unbatch(g):
            # g.ndata 能够访问所有节点的状态
            rst_hidden.append(sub_g.ndata['h'])
        # （图的个数，node_num, 256）
        return rst_hidden

class Head(nn.Module):
    """The MLP submodule"""

    def __init__(self, gcn_out_size: int, bert_out_size: int):
        super().__init__()
        # 512
        self.bert_out_size = bert_out_size
        # 256
        self.gcn_out_size = gcn_out_size

        self.fc = nn.Sequential(
            nn.BatchNorm1d(bert_out_size * 3 + gcn_out_size * 3),
            nn.Dropout(0.5),
            nn.Linear(bert_out_size * 3 + gcn_out_size * 3, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            # 结果是长度为6的一维数组
            nn.Linear(256, 6),
        )
        for i, module in enumerate(self.fc):
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                if getattr(module, "weight_v", None) is not None:
                    nn.init.uniform_(module.weight_g, 0, 1)
                    nn.init.kaiming_normal_(module.weight_v)
                    assert model[i].weight_g is not None
                else:
                    nn.init.kaiming_normal_(module.weight)
                nn.init.constant_(module.bias, 0)

    def tensor_pool(self, h, mask, type='max'):
        if type == 'max':
            h = h.masked_fill(mask, -1e12)
            return torch.max(h, 1)[0]
        elif type == 'avg':
            h = h.masked_fill(mask, 0)
            return h.sum(1) / (mask.size(1) - mask.float().sum(1))
        else:
            h = h.masked_fill(mask, 0)
            return h.sum(1)

    def forward(self, gcn_outputs, offsets_gcn, bert_embeddings, style):
        gcn_extracted_outputs = []
        # gcn_outputs -> （图的个数，node_num, 256）
        # bert_embeddings -> (batch_size, 512 * 3)
        # offsets_gcn (batch_size, 3)
        # 其中，图的个数，batch_size都是处理的一批句子的数量
        for index, data in enumerate(gcn_outputs):
            event1_pos = offsets_gcn[index][0]
            event2_pos = offsets_gcn[index][1]
            event1_mask = [1] * len(data)
            token_mask = [0] * len(data)

            for event_index, event_data in enumerate(event1_pos):
                if (event_data != -1):
                    event1_mask[event_data] = 0
                    token_mask[event_data] = 1
            event2_mask = [1] * len(data)
            for event_index, event_data in enumerate(event2_pos):
                if (event_data != -1):
                    event2_mask[event_data] = 0
                    token_mask[event_data] = 1
            if style == "train":
                token_mask = torch.BoolTensor(token_mask).unsqueeze(0).unsqueeze(2).bool().cuda()
                event1_mask = torch.BoolTensor(event1_mask).unsqueeze(0).unsqueeze(2).bool().cuda()
                event2_mask = torch.BoolTensor(event2_mask).unsqueeze(0).unsqueeze(2).bool().cuda()
                cu_gcn_emb = data.unsqueeze(0)
                token_output = self.tensor_pool(cu_gcn_emb, token_mask, 'max').float().cuda()
                event1_output = self.tensor_pool(cu_gcn_emb, event1_mask, 'avg').float().cuda()
                event2_output = self.tensor_pool(cu_gcn_emb, event2_mask, 'avg').float().cuda()
                cu_g_outpus = torch.cat([token_output, event1_output, event2_output], dim=0)
                gcn_extracted_outputs.append(cu_g_outpus)
            else:
                token_mask = torch.BoolTensor(token_mask).unsqueeze(0).unsqueeze(2).bool()
                event1_mask = torch.BoolTensor(event1_mask).unsqueeze(0).unsqueeze(2).bool()
                event2_mask = torch.BoolTensor(event2_mask).unsqueeze(0).unsqueeze(2).bool()
                cu_gcn_emb = data.unsqueeze(0)
                token_output = self.tensor_pool(cu_gcn_emb, token_mask, 'max').float()
                event1_output = self.tensor_pool(cu_gcn_emb, event1_mask, 'avg').float()
                event2_output = self.tensor_pool(cu_gcn_emb, event2_mask, 'avg').float()
                cu_g_outpus = torch.cat([token_output, event1_output, event2_output], dim=0)
                gcn_extracted_outputs.append(cu_g_outpus)

        # gcn_extracted_outputs = [gcn_outputs[i].unsqueeze(0).gather(1, offsets_gcn[i].unsqueeze(0).unsqueeze(2)
        #                                .expand(-1, -1, gcn_outputs[i].unsqueeze(0).size(2))).view(gcn_outputs[i].unsqueeze(0).size(0), -1) for i in range(len(gcn_outputs))]
        # # 输出应该是【图的个数=batch_size, 3 * 256】
        gcn_extracted_outputs = torch.stack(gcn_extracted_outputs, dim=0)
        gcn_extracted_outputs = gcn_extracted_outputs.reshape(gcn_extracted_outputs.shape[0], -1)

        # [batch_size, 512*3 + 256 * 3]
        embeddings = torch.cat((gcn_extracted_outputs, bert_embeddings), 1)
        # [batch_size, 6]
        return self.fc(embeddings)


class BERT_Head(nn.Module):
    def __init__(self, bert_hidden_size: int):
        super().__init__()
        # nn.Sequential 一个有序的容器，神经网络模块按照在传入构造器的顺序依次被添加到图中计算
        # BatchNorm1d 归一化
        self.fc = nn.Sequential(
            # 归一化
            nn.BatchNorm1d(bert_hidden_size * 3),
            # 在不同的训练过程中随机丢掉一部分神经元，丢掉的那部分神经元不更新权值，但可能下一次输入的时候
            # 会被选中进行更新权值
            nn.Dropout(0.5),
            nn.Linear(bert_hidden_size * 3, 512 * 3),
            nn.ReLU(),
        )




        for i, module in enumerate(self.fc):
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                # 对于设定维度中的每一个位置上都设置为1,常数，固定值
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                if getattr(module, "weight_v", None) is not None:
                    nn.init.uniform_(module.weight_g, 0, 1)
                    nn.init.kaiming_normal_(module.weight_v)
                    assert model[i].weight_g is not None
                else:
                    nn.init.kaiming_normal_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, bert_embeddings):
        # 输入(batch_size, 3 * hidden_size)
        # print('BERT_Head bert_embeddings: ', bert_embeddings, bert_embeddings.view(bert_embeddings.shape[0],-1).shape)
        outputs = self.fc(bert_embeddings.view(bert_embeddings.shape[0], -1))
        # 输出 （batch_size, 512 * 3）
        return outputs


class GPRModel(nn.Module):
    """The main model."""

    def __init__(self):
        super().__init__()
        self.RGCN = RGCNModel(h_dim=1024, num_rels=3, gated=True)

        self.BERThead = BERT_Head(1024)  # bert output size

        self.head = Head(256, 512)  # gcn output   berthead output

    def forward(self, offsets_bert, offsets_gcn, bert_embeddings, g, style="train"):
        gcn_outputs = self.RGCN(g)
        bert_head_outputs = self.BERThead(bert_embeddings)
        head_outputs = self.head(gcn_outputs, offsets_gcn, bert_head_outputs, style)
        return head_outputs


class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(' ')
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)


parser = spacy.load('en')
parser.tokenizer = WhitespaceTokenizer(parser.vocab)

BERT_MODEL = 'bert-large-uncased'
# never_split -> List of tokens which will never be split during tokenization
tokenizer = BertTokenizer.from_pretrained(
    BERT_MODEL,
    never_split=(
    "[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]", "[EVENT1BEFORE]", "[EVENT1END]", "[EVENT2BEFORE]", "[EVENT2END]")
)

tokenizer.vocab["[EVENT1BEFORE]"] = 0
tokenizer.vocab["[EVENT1END]"] = 0
tokenizer.vocab["[EVENT2BEFORE]"] = 0
tokenizer.vocab["[EVENT2END]"] = 0


def is_target(i, target_offset_list):
    return i in target_offset_list


def transfer_n_e(nodes, edges):
    num_nodes = len(nodes)
    new_edges = []
    for e1, e2 in edges:
        new_edges.append([nodes[e1], nodes[e2]])
    return num_nodes, new_edges


all_graphs = []

gcn_offsets = []
index = 0
# 每一句话生成一个子图，
# token_lst -> tokens of every sentence without padding
for i, sent_token in enumerate(token_lst):
    cu_offsets = offsets_lst[i]
    sent_token = sent_token[1:cu_offsets[0]] + sent_token[cu_offsets[0] + 1:cu_offsets[1]] + sent_token[cu_offsets[1] + 1: cu_offsets[4]] + sent_token[cu_offsets[4] + 1: cu_offsets[2]] + sent_token[cu_offsets[2] + 1: cu_offsets[3]] + sent_token[cu_offsets[3] + 1:-1]
    # 将句子中的开始标签和结尾标签去除掉 sent_token[1:-1]
    # bert可以会把一个词分成多个词，分的词前部添加#，所以要去掉#
    event1_begin = cu_offsets[0] - 1
    event1_long = cu_offsets[1] - cu_offsets[0] - 1
    event2_begin = cu_offsets[2] - 4
    sent1_end = cu_offsets[4] - 4
    event2_long = cu_offsets[3] - cu_offsets[2] - 1
    sent = ' '.join([re.sub("[#]", "", token) for token in tokenizer.convert_ids_to_tokens(sent_token)])
    sent_id_mapping = np.arange(0, len(sent.split(' ')), 1)
    for index, data in enumerate(sent_id_mapping):
        if (index < event1_begin):
            sent_id_mapping[index] = sent_id_mapping[index] + 1
        elif (index < (event1_begin + event1_long)):
            sent_id_mapping[index] = sent_id_mapping[index] + 2
        elif (index <= sent1_end):
            sent_id_mapping[index] = sent_id_mapping[index] + 3
        elif (index < event2_begin):
            sent_id_mapping[index] = sent_id_mapping[index] + 4
        elif (index < (event2_begin + event2_long)):
            sent_id_mapping[index] = sent_id_mapping[index] + 5
        elif index >= (event2_begin + event2_long):
            sent_id_mapping[index] = sent_id_mapping[index] + 6
    doc = parser(sent)
    parse_rst = doc.to_json()

    # 原来
    # target_offset_list = [item - 1 for item in offsets_lst[i]]
    # offsets_lst记录的是利用gcn分的词的事件1开始下标，事件1结束下标，事件2开始下标，事件2结束下标，句子1结尾标签，句子2结尾标签
    # [event1_begin, event1_end, event2_begin, event2_end, first_sep_pos, second_sep_pos]

    event1_pos = np.arange(event1_begin, event1_begin + event1_long, 1).tolist()
    event2_pos = np.arange(event2_begin, event2_begin + event2_long, 1).tolist()

    target_offset_list = event1_pos + event2_pos

    nodes = collections.OrderedDict()
    edges = []
    edge_type = []

    extra_sencond_offset_list = []

    for i_word, word in enumerate(parse_rst['tokens']):
        if is_target(word['head'], target_offset_list):
            extra_sencond_offset_list.append(i_word)

    target_offset_list = target_offset_list + extra_sencond_offset_list
    # nodes是字典类型，key是在该句中的位置下标id,value是在定义在图中的节点id
    for i_word, word in enumerate(parse_rst['tokens']):
        # i_word实际是爱的当前分词的下标
        # 当前词是A，B或者指定代词，向下执行下一步 或者 当前词的依存关系的头结点是A,B或者指定代词 执行下一步
        if not (is_target(i_word, target_offset_list) or is_target(word['head'], target_offset_list)):
            continue
        # 添加当前节点的自反边  边类型为0
        #
        if i_word not in nodes:
            nodes[i_word] = len(nodes)
            edges.append([i_word, i_word])
            edge_type.append(0)
        # 添加当前节点的依存关系的头结点的自反边
        if word['head'] not in nodes:
            nodes[word['head']] = len(nodes)
            edges.append([word['head'], word['head']])
            edge_type.append(0)

        if not ([word['head'], word['id']] in edges):
            edges.append([word['head'], word['id']])
            edge_type.append(1)
        if not ([word['id'], word['head']] in edges):
            edges.append([word['id'], word['head']])
            edge_type.append(2)

    # 将边两边的端点转化为id
    # train_edges 二维数组 [边的个数，2]  [[边的头部在图中的id,边的尾部在图中的id]]
    num_nodes, tran_edges = transfer_n_e(nodes, edges)
    # 将事件1有关分词与事件2有关分词转化为相关节点下标
    event1_nodes = [-1] * 12
    # nodes 的key为在原输入句子中的位置，value为在图中的位置
    for index, data in enumerate(event1_pos):
        cu_nodes = nodes[data]
        event1_nodes[index] = cu_nodes
    event2_nodes = [-1] * 12
    for index, data in enumerate(event2_pos):
        cu_nodes = nodes[data]
        event2_nodes[index] = cu_nodes

    gcn_offset = [event1_nodes, event2_nodes]
    # gcn_offset = [nodes[offset] for offset in target_offset_list]
    # 句子个数就是图的个数，每个句子是一个子图
    # 二维数组，[句子个数,3]
    gcn_offsets.append(gcn_offset)

    G = dgl.DGLGraph()
    G.add_nodes(num_nodes)
    G.add_edges(list(zip(*tran_edges))[0], list(zip(*tran_edges))[1])

    for i_word, word in enumerate(parse_rst['tokens']):
        if not (is_target(i_word, target_offset_list) or is_target(word['head'], target_offset_list)):
            continue

        # others_bert_outputs[i] （batch_size, sequence_length, hidden_size）, 而batch_size 为1
        # others_bert_outputs[i][0] -》 （sequence_length, hidden_size）
        # others_bert_outputs[i][0][target_offset_list.index(i_word)] -> 一维数组 （hidden_size）
        # .unsqueeze(0) 将一维转化为二维 (1, hidden_size)
        # .cuda() 增加了对CUDA张量的支持，实现了与CPU张量相同的功能，但使用GPU进行计算
        G.nodes[[nodes[i_word]]].data['h'] = bert_outputs_lst[i][0][sent_id_mapping[i_word]].unsqueeze(0).float().to(
            device)
        G.nodes[[nodes[word['head']]]].data['h'] = bert_outputs_lst[i][0][sent_id_mapping[word['head']]].unsqueeze(
            0).float().to(device)

    edge_norm = []
    for e1, e2 in tran_edges:
        if e1 == e2:
            edge_norm.append(1)
        else:
            edge_norm.append(1 / (G.in_degree(e2) - 1))

    edge_type = torch.from_numpy(np.array(edge_type)).to(device)
    edge_norm = torch.from_numpy(np.array(edge_norm)).unsqueeze(1).float().to(device)

    G.edata.update({'rel_type': edge_type, })
    G.edata.update({'norm': edge_norm})
    all_graphs.append(G)

test_all_graphs = []
test_gcn_offsets = []
for i, sent_token in enumerate(test_token_lst):

    cu_test_offsets = test_offsets_lst[i]
    sent_token = sent_token[1:cu_test_offsets[0]] + sent_token[cu_test_offsets[0] + 1:cu_test_offsets[1]] + sent_token[cu_test_offsets[1] + 1: cu_test_offsets[4]] + sent_token[cu_test_offsets[4] + 1: cu_test_offsets[2]] + sent_token[cu_test_offsets[2] + 1: cu_test_offsets[3]] + sent_token[cu_test_offsets[3] + 1:-1]
    # 将句子中的开始标签和结尾标签去除掉 sent_token[1:-1]
    # bert可以会把一个词分成多个词，分的词前部添加#，所以要去掉#
    event1_begin = cu_test_offsets[0] - 1
    event1_long = cu_test_offsets[1] - cu_test_offsets[0] - 1
    event2_begin = cu_test_offsets[2] - 4
    event2_long = cu_test_offsets[3] - cu_test_offsets[2] - 1
    sent1_end = cu_test_offsets[4] - 4
    sent = ' '.join([re.sub("[#]", "", token) for token in tokenizer.convert_ids_to_tokens(sent_token)])
    sent_id_mapping = np.arange(0, len(sent.split(' ')), 1)
    for index, data in enumerate(sent_id_mapping):
        # 在第一个事件句之前
        if (index < event1_begin):
            sent_id_mapping[index] = sent_id_mapping[index] + 1
        # 在第一个事件句里面
        elif (index < (event1_begin + event1_long)):
            sent_id_mapping[index] = sent_id_mapping[index] + 2
        # 在第一句子结束之前
        elif (index <= sent1_end):
            sent_id_mapping[index] = sent_id_mapping[index] + 3
        # 在第二个事件句开始之前
        elif (index < event2_begin):
            sent_id_mapping[index] = sent_id_mapping[index] + 4
        elif (index < (event2_begin + event2_long)):
            sent_id_mapping[index] = sent_id_mapping[index] + 5
        elif index >= (event2_begin + event2_long):
            sent_id_mapping[index] = sent_id_mapping[index] + 6

    doc = parser(sent)
    parse_rst = doc.to_json()

    event1_pos = np.arange(event1_begin, event1_begin + event1_long, 1).tolist()
    event2_pos = np.arange(event2_begin, event2_begin + event2_long, 1).tolist()

    target_offset_list = event1_pos + event2_pos

    nodes = collections.OrderedDict()
    edges = []
    edge_type = []

    extra_sencond_offset_list = []

    for i_word, word in enumerate(parse_rst['tokens']):
        if is_target(word['head'], target_offset_list):
            extra_sencond_offset_list.append(i_word)

    target_offset_list = target_offset_list + extra_sencond_offset_list

    for i_word, word in enumerate(parse_rst['tokens']):
        if not (is_target(i_word, target_offset_list) or is_target(word['head'], target_offset_list)):
            continue

        if i_word not in nodes:
            nodes[i_word] = len(nodes)
            edges.append([i_word, i_word])
            edge_type.append(0)
        if word['head'] not in nodes:
            nodes[word['head']] = len(nodes)
            edges.append([word['head'], word['head']])
            edge_type.append(0)

        if not ([word['head'], word['id']] in edges):
            edges.append([word['head'], word['id']])
            edge_type.append(1)
        if not ([word['id'], word['head']] in edges):
            edges.append([word['id'], word['head']])
            edge_type.append(2)

        # if not (([word['head'], word['id']] in edges) and ([word['id'], word['head']])):
        #     edges.append([word['head'], word['id']])
        #     edge_type.append(1)
        #     edges.append([word['id'], word['head']])
        #     edge_type.append(2)

    num_nodes, tran_edges = transfer_n_e(nodes, edges)

    event1_nodes = [-1] * 12
    for index, data in enumerate(event1_pos):
        cu_nodes = nodes[data]
        event1_nodes[index] = cu_nodes
    event2_nodes = [-1] * 12
    for index, data in enumerate(event2_pos):
        cu_nodes = nodes[data]
        event2_nodes[index] = cu_nodes

    test_gcn_offset = [event1_nodes, event2_nodes]
    test_gcn_offsets.append(test_gcn_offset)

    G = dgl.DGLGraph()
    G.add_nodes(num_nodes)
    G.add_edges(list(zip(*tran_edges))[0], list(zip(*tran_edges))[1])

    for i_word, word in enumerate(parse_rst['tokens']):
        if not (is_target(i_word, target_offset_list) or is_target(word['head'], target_offset_list)):
            continue
        G.nodes[[nodes[i_word]]].data['h'] = test_bert_outputs_lst[i][0][sent_id_mapping[i_word]].unsqueeze(
            0).float().to(device)
        G.nodes[[nodes[word['head']]]].data['h'] = test_bert_outputs_lst[i][0][sent_id_mapping[word['head']]].unsqueeze(
            0).float().to(device)
    edge_norm = []
    for e1, e2 in tran_edges:
        if e1 == e2:
            edge_norm.append(1)
        else:
            edge_norm.append(1 / (G.in_degree(e2) - 1))

    edge_type = torch.from_numpy(np.array(edge_type))
    edge_norm = torch.from_numpy(np.array(edge_norm)).unsqueeze(1).float().to(device)

    G.edata.update({'rel_type': edge_type, })
    G.edata.update({'norm': edge_norm})
    test_all_graphs.append(G)


def find_data_index(temp, relation_list):
    return relation_list.index(temp)


class GPRDataset(Dataset):
    def __init__(self, original_df, graphs, bert_offsets, gcn_offsets, bert_embeddings):
        all_relation = ['a', 'b', 's', 'i', 'ii', 'v']
        tmp = original_df['relation'].values
        labels_id = []
        for cu_rela in tmp:
            cu_label_id = find_data_index(cu_rela, all_relation)
            labels_id.append(cu_label_id)
        self.y = torch.tensor(labels_id).long()
        self.graphs = graphs
        self.bert_offsets = bert_offsets  # 已经+1了
        self.bert_embeddings = bert_embeddings  # 有[CLS]
        self.gcn_offsets = gcn_offsets

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.bert_offsets[idx], self.gcn_offsets[idx], self.bert_embeddings[idx], self.y[idx]


# In[ ]:


def collate(samples):
    # bert_offsets [batch_size, 3]
    # labels [batch_size, 3]
    # bert_embeddings [batch_size, 1, 3, hidden_size]
    graphs, bert_offsets, gcn_offsets, bert_embeddings, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    offsets_bert = torch.stack([torch.LongTensor(x) for x in bert_offsets], dim=0)
    offsets_gcn = torch.stack([torch.LongTensor(x) for x in gcn_offsets], dim=0)
    labels = torch.stack(labels, dim=0)
    # one_hot_labels = torch.stack([torch.from_numpy(x.astype("uint8")) for x in labels], dim=0)
    # 返回每一行中最大值的索引下标 因为数组中不是1就是0，所以就是查找1的索引位置下标，返回的是一维数组[batch_size]
    # labels = one_hot_labels.max(dim=1)
    # 去掉维度为1的
    # 变成了[batch_size, 3, hidden_size]
    bert_embeddings = torch.stack(bert_embeddings, dim=0)

    return batched_graph, offsets_bert, offsets_gcn, bert_embeddings, labels

test_dataset = GPRDataset(original_df=test_df, graphs=test_all_graphs, bert_offsets=test_offsets_lst,
                          gcn_offsets=test_gcn_offsets, bert_embeddings=test_others_bert_outputs)
train_dataset = GPRDataset(original_df=train_df, graphs=all_graphs, bert_offsets=offsets_lst, gcn_offsets=gcn_offsets,
                           bert_embeddings=others_bert_outputs)

train_dataloarder = DataLoader(
    train_dataset,
    collate_fn=collate,
    batch_size=4,
    shuffle=True,
)

test_dataloarder = DataLoader(
    test_dataset,
    collate_fn=collate,
    batch_size=4,
)


def send_graph_to_cpu(g):
    # nodes
    labels = g.node_attr_schemes()
    for l in labels.keys():
        g.ndata[l] = g.ndata.pop(l).cpu()
    # edges
    labels = g.edge_attr_schemes()
    for l in labels.keys():
        g.edata[l] = g.edata.pop(l).cpu()
    return g


lr_value = 0.0001
total_epoch = 100

def adjust_learning_rate(optimizers, epoch):
    # warm up
    if epoch < 10:
        lr_tmp = 0.00001
    else:
        lr_tmp = lr_value * pow((1 - 1.0 * epoch / 100), 0.9)

    if epoch > 36:
        lr_tmp = 0.000015 * pow((1 - 1.0 * epoch / 100), 0.9)

    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_tmp

    return lr_tmp

all_relation = ['a', 'b', 's', 'i', 'ii', 'v']
tmp = train_df['relation'].values
labels_id = []
for cu_rela in tmp:
    cu_label_id = find_data_index(cu_rela, all_relation)
    labels_id.append(cu_label_id)
train_y = labels_id

from operator import itemgetter

kfold = StratifiedKFold(n_splits=5)
test_predict_lst = []  # the test output for every fold
cu_step = 0
for train_index, test_index in kfold.split(train_df, train_y):
    print("=" * 20)
    print(f"Fold {len(test_predict_lst) + 1}")
    print("=" * 20)
    # test_index是一个数组，可以获取该数组中显示位置的数据记录
    # itemgetter(*test_index) 返回的是一个函数
    # itemgetter(*test_index)(offsets_lst) 获取相应位置下的数据
    val_dataset = GPRDataset(original_df=train_df.iloc[test_index],
                             graphs=list(itemgetter(*test_index)(all_graphs)),
                             bert_offsets=list(itemgetter(*test_index)(offsets_lst)),
                             gcn_offsets=list(itemgetter(*test_index)(gcn_offsets)),
                             bert_embeddings=list(itemgetter(*test_index)(others_bert_outputs)))

    train_dataset = GPRDataset(original_df=train_df.iloc[train_index],
                               graphs=list(itemgetter(*train_index)(all_graphs)),
                               bert_offsets=list(itemgetter(*train_index)(offsets_lst)),
                               gcn_offsets=list(itemgetter(*train_index)(gcn_offsets)),
                               bert_embeddings=list(itemgetter(*train_index)(others_bert_outputs)))

    # 每批次有4个数据
    train_dataloarder = DataLoader(
        train_dataset,
        collate_fn=collate,
        batch_size=4,
        shuffle=True, )

    val_dataloarder = DataLoader(
        val_dataset,
        collate_fn=collate,
        batch_size=4)

    model = GPRModel().cuda()
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr_value)
    reg_lambda = 0.035

    print('Dataloader Success---------------------')

    best_val_loss = 11
    # ce_losses = []
    # epoch_losses = []
    # val_looses = []
    # total_epoch 循环100次
    for epoch in range(total_epoch):

        if epoch % 5 == 0:
            print('|', ">" * epoch, " " * (80 - epoch), '|')

        lr = adjust_learning_rate([optimizer], epoch)
        # print("Learning rate = %4f\n" % lr)
        model.train()
        # epoch_loss = 0
        # reg_loss = 0
        # ce_loss = 0
        # batched_graph , offsets_bert大体上都是一个长度为batch_size的数组，
        for iter, (batched_graph, offsets_bert, offsets_gcn, bert_embeddings, labels) in enumerate(train_dataloarder):

            bert_embeddings = bert_embeddings.to(device)
            labels = labels.to(device)
            offsets_gcn = offsets_gcn.to(device)
            # [batch_size, 3]
            prediction = model(offsets_bert, offsets_gcn, bert_embeddings, batched_graph, "train")
            # l2正则化 为防止模型处于过拟合状态，使用L2正则化来降低复杂度
            l2_reg = None
            for w in model.RGCN.parameters():
                if not l2_reg:
                    l2_reg = w.norm(2)
                else:
                    l2_reg = l2_reg + w.norm(2)
            for w in model.head.parameters():
                if not l2_reg:
                    l2_reg = w.norm(2)
                else:
                    l2_reg = l2_reg + w.norm(2)
            for w in model.BERThead.parameters():
                if not l2_reg:
                    l2_reg = w.norm(2)
                else:
                    l2_reg = l2_reg + w.norm(2)
            loss = loss_func(prediction, labels) + l2_reg * reg_lambda
            # loss = loss_func(prediction, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_loss = 0
        model.eval()
        val_labels = []
        val_predict = None

        with torch.no_grad():
            for iter, (batched_graph, offsets_bert, offsets_gcn, bert_embeddings, labels) in enumerate(val_dataloarder):
                bert_embeddings = bert_embeddings.to(device)
                labels = labels.to(device)
                offsets_gcn = offsets_gcn.to(device)
                val_lab = labels.cpu()
                cu_val_lab = [cu_label.numpy().tolist() for cu_label in val_lab]
                val_labels = val_labels + cu_val_lab

                prediction = model(offsets_bert, offsets_gcn, bert_embeddings, batched_graph, "train")
                loss = loss_func(prediction, labels)
                # 如果一个tensor只有一个元素，那么可以使用.item()方法取出这个元素作为普通的python数字
                if val_predict is None:
                    val_predict = prediction
                else:
                    val_predict = torch.cat((val_predict, prediction), 0)

                val_loss += loss.detach().item()
            val_loss = val_loss / (iter + 1)

        val_predictions = np.argmax(val_predict.data.cpu().numpy(), axis=1).tolist()
        m_f = f1_score(val_labels, val_predictions, labels=[0, 1, 3, 4, 5], average='micro')
        print('验证集 Epoch {}, micro-f1值 {}'.format(epoch, m_f))
        if epoch % 20 == 0:
            print('Epoch {}, val_loss {:.4f}'.format(epoch, val_loss))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if epoch > 20:
                torch.save(model.state_dict(), 'best_model.pth')
            if epoch > 36: print('Best val loss found: ', best_val_loss)

    print('This fold, the best val loss is: ', best_val_loss)

    test_loss = 0.
    test_predict = None

    model = GPRModel()
    model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))
    model.eval()
    test_labels = []
    # all_relation = ['a', 'b', 's', 'i', 'ii', 'v']
    id2labels = {0: 'a', 1: 'b', 2: 's', 3: 'i', 4: 'ii', 5: 'v'}
    with torch.no_grad():
        for iter, (batched_graph, offsets_bert, offsets_gcn, bert_embeddings, labels) in enumerate(test_dataloarder):

            offsets_gcn = offsets_gcn
            bert_embeddings = bert_embeddings.cpu()
            labels = [cu_label.numpy().tolist() for cu_label in labels]
            test_labels = test_labels + labels
            batched_graph = send_graph_to_cpu(batched_graph)
            # [batch_size, 3]
            prediction = model(offsets_bert, offsets_gcn, bert_embeddings, batched_graph, "test")
            if test_predict is None:
                test_predict = prediction
            else:
                test_predict = torch.cat((test_predict, prediction), 0)

    predictions = np.argmax(test_predict.data.cpu().numpy(), axis=1).tolist()
    # labels = [id2labels[pre_label] for pre_label in test_labels]
    p_class, r_class, f_class, support_micro = score(test_labels, predictions, labels=[0, 1, 2, 3, 4, 5])
    m_p = precision_score(test_labels, predictions, labels=[0, 1, 2, 3, 4, 5], average='micro')
    m_r = recall_score(test_labels, predictions, labels=[0, 1, 2, 3, 4, 5], average='micro')
    m_f = f1_score(test_labels, predictions, labels=[0, 1, 2, 3, 4, 5], average='micro')
    print("各类准确率:", p_class)
    print("各类回召率:", r_class)
    print("各类F1值:", f_class)
    print("micro 准确率:", m_p)
    print("micro 召回率：", m_r)
    print("micro F1: ", m_f)
    print("准确率宏平均值:", np.array(p_class).sum() / 5)
    print("召回率宏平均值：", np.array(r_class).sum() / 5)
    print("F1宏平均值:", np.array(f_class).sum() / 5)
    m_p = precision_score(test_labels, predictions, labels=[0, 1, 3, 4, 5], average='micro')
    m_r = recall_score(test_labels, predictions, labels=[0, 1, 3, 4, 5], average='micro')
    m_f = f1_score(test_labels, predictions, labels=[0, 1, 3, 4, 5], average='micro')
    print("去掉特殊 micro 准确率:", m_p)
    print("去掉特殊 micro 召回率：", m_r)
    print("取代哦特殊 micro F1: ", m_f)