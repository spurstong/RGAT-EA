#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import re

import os
import logging
import gc
from pathlib import Path
import pickle

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader

from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel



# In[ ]:


import pandas as pd

train_df = pd.read_csv("E:/pythonTemporalProjects/TemporalGNN-with-BERT/DealData/timebank-dense.csv")




# In[ ]:

# bert-large-uncased 的模型是24layer, 1024hidden, 16 heads
BERT_MODEL = 'bert-large-uncased'

tokenizer = BertTokenizer.from_pretrained(
    BERT_MODEL,
    never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]", "[EVENT1BEFORE]", "[EVENT1END]", "[EVENT2BEFORE]", "[EVENT2END]")
)



# In[ ]:


# ## Tokenize

# In[ ]:


tokenizer.vocab["[EVENT1BEFORE]"] = 0
tokenizer.vocab["[EVENT1END]"] = 0
tokenizer.vocab["[EVENT2BEFORE]"] = 0
tokenizer.vocab["[EVENT2END]"] = 0




def tokenize(row, tokenizer):
    event1_text = row["event1_sen"].split(" ")
    event1_offset = row["event1_offset"]
    event1_length = row["event1_length"]
    event1_text = event1_text[:event1_offset] + ["EVENT1BEFORE"] + event1_text[event1_offset:event1_offset+event1_length] + ["EVENT1END"] + event1_text[event1_offset+event1_length:]
    event1_text = " ".join(event1_text)

    event1_text = re.sub("[^a-zA-Z]", " ", event1_text)
    event1_text = re.sub(r"EVENT1BEFORE", r"[EVENT1BEFORE]", event1_text)
    event1_text = re.sub(r"EVENT1END", r"[EVENT1END]", event1_text)
    event1_text = " ".join(list(filter(None, event1_text.split(" "))))

    event2_text = row["event2_sen"].split(" ")
    event2_offset = row["event2_offset"]
    event2_length = row["event2_length"]
    event2_text = event2_text[:event2_offset] + ["EVENT2BEFORE"] + event2_text[
                                                                     event2_offset:event2_offset + event2_length] + [
                      "EVENT2END"] + event2_text[event2_offset + event2_length:]
    event2_text = " ".join(event2_text)
    event2_text = re.sub("[^a-zA-Z]", " ", event2_text)
    event2_text = re.sub(r"EVENT2BEFORE", r"[EVENT2BEFORE]", event2_text)
    event2_text = re.sub(r"EVENT2END", r"[EVENT2END]", event2_text)
    event2_text = " ".join(list(filter(None, event2_text.split(" "))))
    # 由于在bert要对数据进行格式化， 如果是两个句子合并 [CLS]+ 句子1 + [SEP] + 句子2 + [SEP]
    # event1_offset = event1_offset + 1
    # event2_offset = len(event1_text.split(" ")) + int(event2_offset) + 2
    # first_sep_pos = len(event1_text.split(' ')) + 1
    # second_sep_pos = len(event1_text.split(' ')) + len(event2_text.split(' ')) + 2
    event_text = '[CLS] ' + event1_text + ' [SEP] ' + event2_text + ' [SEP]'
    final_tokens = tokenizer.tokenize(event_text)
    print(final_tokens)
    event1_begin = final_tokens.index("[EVENT1BEFORE]")
    print(event1_begin)
    event1_end = final_tokens.index("[EVENT1END]")
    print(event1_end)
    event2_begin = final_tokens.index("[EVENT2BEFORE]")
    event2_end = final_tokens.index("[EVENT2END]")
    first_sep_pos = final_tokens.index("[SEP]")
    second_sep_pos = len(final_tokens) - 1
    return final_tokens, [event1_begin, event1_end, event2_begin, event2_end, first_sep_pos, second_sep_pos]


# In[ ]:


offsets_lst = []
# 分词在bert词典中的id
# 二维数组，[句子数量，每个句子的长度]
tokens_lst = []
for _, row in train_df.iterrows():
    tokens, offsets = tokenize(row, tokenizer)
    offsets_lst.append(offsets)
    tokens_lst.append(tokenizer.convert_tokens_to_ids(tokens))


# ## Pad the sequences

# In[ ]:


max((len(x) for x in tokens_lst))


# In[ ]:


# truncate each row to the size of max_len

max_len = 215
tokens = np.zeros((len(tokens_lst), max_len), dtype=np.int64)
for i, row in enumerate(tokens_lst):
    row = np.array(row[:215])
    tokens[i, :len(row)] = row


# All sentenses 二维数组
# 按照设置的句子的最长长度进行截取
# [句子数量, max_len]
token_tensor = torch.from_numpy(tokens)


# ## Generate Embedding

# In[ ]:


#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
#torch.cuda.set_device(1)
bert = BertModel.from_pretrained(BERT_MODEL)


# In[ ]:

# attention_mask: 传入每个实例的长度，用于attention的mask
# output_all_encoded_layers: 控制是否输出所有encoder层的结果
# token_tensor[i]变成一维数组，长度为max_len，即设定的句子最长长度
# unsqueeze 在第一维增加一个维度，即 （1， max_len）
# bert_output （1, sequence_length, hidden_size）
bert_outputs = []
with torch.no_grad():
    for i in range(len(token_tensor)):
        if i % 40 == 0:
            print(i)
        bert_output, _ =  bert(
                    token_tensor[i].unsqueeze(0),
                    # (token_tensor[i].unsqueeze(0) > 0).long 将数组中值大于0的变成1， 小于等于0的变成0
                    # 得到一个二维数组 即[1, max_len]
                    attention_mask=(token_tensor[i].unsqueeze(0) > 0).long(),
                    token_type_ids=None, 
                    output_all_encoded_layers=False) 

        bert_outputs.append(bert_output)

# bert_outputs 是[句子数量,1,max_len, hidden_size=1024]
# In[ ]:


pickle.dump(offsets_lst, open('offsets_lst.pkl', "wb"))
pickle.dump(tokens_lst, open('token_lst_wto_padding.pkl', "wb"))
pickle.dump(bert_outputs, open('bert_outputs.pkl', "wb"))

#pickle.dump(offsets_lst, open('test_offsets_lst.pkl', "wb"))
#pickle.dump(tokens_lst, open('test_token_lst_wto_padding.pkl', "wb"))
#pickle.dump(bert_outputs, open('test_bert_outputs.pkl', "wb"))

