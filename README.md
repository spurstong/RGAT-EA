# Incorporating Edge Attributes in Relational Graph Attention Networks for Event Temporal Relation Extraction 

# Introduction
Event temporal relation extraction is a challenging Natural Language Processing (NLP) task, 
and most of the previous work is the sequential modeling method (e.g., RNN). So, it’s difficult for them to adequately capture potential long-distance semantic information in the event sentence pairs. 
In this paper, we propose a novel Relational Graph ATtention Networks that incorporates Edge Attributes to slove above problem, 
called RGAT-EA. We first build a semantic dependency graph through dependency parsing, 
and then model semantic graph considering the attributes of edges and using top-k attention mechanism to learn hidden semantic 
contextual representations, and finally predict event temporal relation. 
We evaluate RGAT-EA on three datasets (i.e., TB-Dense, MATRES, TCR). 
Compared with the previous baselines, the micro-F1 scores obtained by our model are improved by 10.0\%, 2.8\%, and 2.7\%, respectively.

# Requirement
- python 3
- pytorch == 1.2.0
- spacy == 2.1.8 

# How to run the code?

First we need to run the code in the `DealData` directory to get the relevant token embedding vectors, 
and then run the `Main_Model_gated_rgcn.py` code in the `Codes` directory to implement the algorithm in the paper. 

# Corpus 

[TB-Dense](https://www.usna.edu/Users/cs/nchamber/caevo/#corpus)  
[TCR](https://github.com/qiangning/TemporalCausalReasoning)  
[MATRES](https://github.com/qiangning/MATRES)  
