#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import re
import numpy as np
import pandas as pd
import codecs

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from pytorch_pretrained_bert.tokenization import BertTokenizer, BasicTokenizer
from pytorch_pretrained_bert.modeling import BertModel
import pickle

df_train_val = pd.read_csv("E:/pythonTemporalProjects/TemporalGNN-with-BERT/DealData/timebank-dense.csv")


BERT_MODEL = 'bert-large-uncased'
CASED = True

# 转化为小写字母
tokenizer = BertTokenizer.from_pretrained(
    BERT_MODEL,
    do_lower_case=CASED,
    never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]", "[EVENT1BEFORE]", "[EVENT1END]", "[EVENT2BEFORE]", "[EVENT2END]")
)

tokenizer.convert_ids_to_tokens()

def find_data_index(temp, relation_list):
    return relation_list.index(temp)

tokenizer.vocab["[EVENT1BEFORE]"] = 0
tokenizer.vocab["[EVENT1END]"] = 0
tokenizer.vocab["[EVENT2BEFORE]"] = 0
tokenizer.vocab["[EVENT2END]"] = 0

# "[EVENT1BEFORE]", "[EVENT1END]", "[EVENT2BEFORE]", "[EVENT2END]"
def tokenize(row, tokenizer):
    event1_text = row["event1_sen"].split(" ")
    event1_offset = row["event1_offset"]
    event1_length = row["event1_length"]
    event1_text = event1_text[:event1_offset] + ["[EVENT1BEFORE]"] + event1_text[event1_offset:event1_offset+event1_length] + ["[EVENT1END]"] + event1_text[event1_offset+event1_length:]
    event1_text = " ".join(event1_text)

    event2_text = row["event2_sen"].split(" ")
    event2_offset = row["event2_offset"]
    event2_length = row["event2_length"]
    event2_text = event2_text[:event2_offset] + ["[EVENT2BEFORE]"] + event2_text[
                                                                     event2_offset:event2_offset + event2_length] + [
                      "[EVENT2END]"] + event2_text[event2_offset + event2_length:]
    event2_text = " ".join(event2_text)

    # 由于在bert要对数据进行格式化， 如果是两个句子合并 [CLS]+ 句子1 + [SEP] + 句子2 + [SEP]
    # event1_offset = event1_offset + 1
    # event2_offset = len(event1_text.split(" ")) + int(event2_offset) + 2
    # first_sep_pos = len(event1_text.split(' ')) + 1
    # second_sep_pos = len(event1_text.split(' ')) + len(event2_text.split(' ')) + 2
    event_text = '[CLS] ' + event1_text + ' [SEP] ' + event2_text + ' [SEP]'
    final_tokens = tokenizer.tokenize(event_text)
    event1_begin = final_tokens.index("[EVENT1BEFORE]")
    event1_end = final_tokens.index("[EVENT1END]")
    event2_begin = final_tokens.index("[EVENT2BEFORE]")
    event2_end = final_tokens.index("[EVENT2END]")
    first_sep_pos = final_tokens.index("[SEP]")
    second_sep_pos = len(final_tokens) - 1
    return final_tokens, [event1_begin, event1_end, event2_begin, event2_end, first_sep_pos, second_sep_pos]


class GAPDataset(Dataset):
    def __init__(self, df, tokenizer, labeled=True):
        self.labeled = labeled
        if labeled:
            all_relation = ['a', 'b', 's', 'i', 'ii', 'v']
            tmp = df['relation']
            labels_id = []
            for cu_rela in tmp:
                cu_label_id = find_data_index(cu_rela, all_relation)
                labels_id.append(cu_label_id)
            self.y = labels_id

        # Extracts the tokens and offsets(positions of A, B, and P)
        self.offsets = []
        self.tokens = []
        # 所有句子
        for _, row in df.iterrows():
            tokens, offsets = tokenize(row, tokenizer)
            self.offsets.append(offsets)
            self.tokens.append(tokenizer.convert_tokens_to_ids(tokens))

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        if self.labeled:
            return self.tokens[idx], self.offsets[idx], self.y[idx]

        return self.tokens[idx], self.offsets[idx], None


def collate_examples(batch, truncate_len=230):
    # 实际上是输入的是[(),(), ()] 第一个是分词id,第二个是A,B和代词的id, 第三个是lebel
    transposed = list(zip(*batch))

    # 获取分的词的个数，当超过500时就截取
    max_len = min(max((len(x) for x in transposed[0])), truncate_len)
    print(max_len)
    tokens = np.zeros((len(batch), max_len), dtype=np.int64)
    for i, row in enumerate(transposed[0]):
        row = np.array(row[:truncate_len])
        tokens[i, :len(row)] = row

    # 二维数组 [1, max_len]
    token_tensor = torch.from_numpy(tokens)
    offsets = transposed[1][0]
    token_mask = [0] * max_len
    token_mask[0] = 1
    token_mask[offsets[4]] = 1
    if ((offsets[5] + 1) >= max_len):
        token_mask[-1] = 1
    else:
        token_mask[offsets[5]:] = [1] * (max_len - offsets[5])
    token_mask[offsets[0]: offsets[1] + 1] = [1] * (offsets[1] - offsets[0] + 1)
    token_mask[offsets[2]: offsets[3] + 1] = [1] * (offsets[3] - offsets[2] + 1)

    # 输出维度 [1, max_len]
    token_mask = torch.tensor(token_mask).unsqueeze(0)

    event1_mask = [1] * max_len
    event1_mask[offsets[0] + 1: offsets[1]] = [0] * (offsets[1] - offsets[0] - 1)
    event1_mask = torch.tensor(event1_mask).unsqueeze(0)

    event2_mask = [1] * max_len
    event2_mask[offsets[2]+1:offsets[3]] = [0] * (offsets[3] - offsets[2] - 1)
    event2_mask = torch.tensor(event2_mask).unsqueeze(0)
    masks = torch.cat([token_mask, event1_mask, event2_mask], dim=0)

    # 二维数组 [1, 3]
    # offsets = torch.stack([torch.LongTensor(x) for x in transposed[1]], dim=0) + 1 # Account for the [CLS] token

    # abels = torch.tensor(transposed[2]).squeeze()
    return token_tensor, masks




# In[ ]:


train_val_ds = GAPDataset(df_train_val, tokenizer)

train_loader = DataLoader(
    train_val_ds,
    collate_fn = collate_examples,
    batch_size = 1,
    shuffle=False,
)




# In[ ]:


bert = BertModel.from_pretrained(BERT_MODEL)


# In[ ]:
def tensor_pool(h, mask, type='max'):
    if type == 'max':
        h = h.masked_fill(mask, -1e12)
        return torch.max(h, 1)[0]
    elif type == 'avg':
        h = h.masked_fill(mask, 0)
        return h.sum(1) / (mask.size(1) - mask.float().sum(1))
    else:
        h = h.masked_fill(mask, 0)
        return h.sum(1)

def deal_outputs(bert_outputs, offsets):
    # [1, max_len]
    token_mask = offsets[0].unsqueeze(0).unsqueeze(2).byte()
    event1_mask = offsets[1].unsqueeze(0).unsqueeze(2).byte()
    event2_mask = offsets[2].unsqueeze(0).unsqueeze(2).byte()
    # [1, hidden_size]
    token_output = tensor_pool(bert_outputs, token_mask, 'max').float()

    event1_output = tensor_pool(bert_outputs, event1_mask, 'avg').float()

    event2_output = tensor_pool(bert_outputs, event2_mask, 'avg').float()

    outpus = torch.cat([token_output, event1_output, event2_output], dim=0)

    return outpus


bert_outputs = []

with torch.no_grad():
    for token_tensor, offsets in train_loader:
        bert_output, _ = bert(
            token_tensor,
            # (token_tensor[i].unsqueeze(0) > 0).long 将数组中值大于0的变成1， 小于等于0的变成0
            # 得到一个二维数组 即[1, max_len]
            attention_mask=(token_tensor > 0).long(),
            token_type_ids=None,
            output_all_encoded_layers=False)

        outputs = deal_outputs(bert_output, offsets)
        # [句子数量，1， 3, hidden_size]
        bert_outputs.append(outputs)

print("成功")
# In[ ]:


pickle.dump(bert_outputs, codecs.open('others_bert_outputs.pkl', "wb"))
#pickle.dump(bert_outputs, codecs.open('test_others_bert_outputs.pkl', "wb"))
