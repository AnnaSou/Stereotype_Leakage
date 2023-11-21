# mark
from transformers import RobertaTokenizer, RobertaForMaskedLM
from transformers import BertTokenizer, BertForMaskedLM
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import AdamW

import torch
from torch.nn import functional as F

import numpy as np
from numpy import linalg as LA
import pandas as pd
import math
import statistics
from collections import defaultdict
import csv
import random
import os
from multiclassUpdate import multiclass_update

random.seed(10)



def equal(l1,l2):
    assert len(l1) == len(l2)
    for x,y in zip(l1,l2):
        if x!=y: return False
    return True


def get_lognorm_score(model, tokenizer, sentence, trait):
    # print(trait)
    # print(sentence)
    trait_ids = tokenizer.encode(trait, return_tensors='pt').squeeze().tolist()
    # print(trait_ids)
    trait_ids = trait_ids[1:len(trait_ids)-1]
    # print(trait_ids)
    scores = []
    input_ids = tokenizer.encode(sentence, return_tensors='pt')
    masked_position = (input_ids.squeeze() == tokenizer.mask_token_id).nonzero()
    masked_poss = [mask.item() for mask in masked_position]
    # print(masked_poss)
    for i in range(len(trait_ids)):
        # print(input_ids)
        # print(masked_pos)
        # print(i)
        first_masked_pos = masked_poss[i]
        target_id = trait_ids[i]
        # print("Masked position, target id: ", first_masked_pos, target_id)
        output = model(input_ids,output_hidden_states=True)
        logits = output.logits.squeeze()[first_masked_pos].cpu().detach().numpy() # y
        lhs = output.hidden_states[-1].squeeze()[first_masked_pos] 
        # print('X: ', lhs.shape)
        lhs = model.cls.predictions.transform(lhs).cpu().detach().numpy() # x
        # print('X: ', lhs.shape)
        # print('Y: ', logits.shape, logits)
        weights = model.cls.predictions.decoder.weight.squeeze().cpu().detach().numpy()
        bias = model.cls.predictions.decoder.bias.reshape(-1,1).cpu().detach().numpy()
        # print('A: ',weights.shape)
        # print('Bias: ',bias.shape, bias)
        lhs = np.append(lhs, 1)
        # print('Final X: ', lhs.shape)
        W = np.append(weights, bias, axis=1)
        # print('Final A: ', W.shape)
        newW = multiclass_update(W, lhs, target_id)
        score = np.log(1-LA.norm(W-newW))
        scores = np.append(scores, score)
        input_ids[0][first_masked_pos] = target_id
    # print('#',scores)
    fscore = np.sum(scores)
    # print(fscore)
    return fscore


def wordasso(model, tokenizer, groups, traits, prior_group, tmplt="The <group> person is <mask>.",traits_for_prior=None, tplt_for_prior=None):
    scores = {'traits': traits}
    for group in groups:
        group_scores = []
        if traits_for_prior == None:
            traits_for_prior = [None]*len(traits)
        for trait, trait_fp in zip(traits, traits_for_prior):
            if trait_fp == None and tplt_for_prior == None:
                trait_ids = tokenizer.encode(trait, return_tensors='pt').squeeze()
                trait_len = list(trait_ids.shape)[0]-2
                # print(trait,trait_ids, trait_len)
                input_txt = tmplt.replace(' <mask>', (' '+tokenizer.mask_token)*trait_len).replace('<group>',prior_group)
                # print(input_txt)
                prior = get_lognorm_score(model, tokenizer, input_txt, trait)
            elif trait_fp != None:
                trait_ids = tokenizer.encode(trait_fp, return_tensors='pt').squeeze()
                trait_len = list(trait_ids.shape)[0]-2
                # print(trait,trait_ids, trait_len)
                input_txt = tmplt.replace(' <mask>', (' '+tokenizer.mask_token)*trait_len).replace('<group>',prior_group)
                # print(input_txt)
                prior = get_lognorm_score(model, tokenizer, input_txt, trait_fp)
                trait_ids = tokenizer.encode(trait, return_tensors='pt').squeeze()
                trait_len = list(trait_ids.shape)[0]-2
                # print(trait,trait_ids, trait_len)
            elif tplt_for_prior != None:
                trait_ids = tokenizer.encode(trait, return_tensors='pt').squeeze()
                trait_len = list(trait_ids.shape)[0]-2
                # print(trait,trait_ids, trait_len)
                input_txt = tplt_for_prior.replace(' <mask>', (' '+tokenizer.mask_token)*trait_len).replace('<group>',prior_group)
                # print(input_txt)
                prior = get_lognorm_score(model, tokenizer, input_txt, trait)
            input_txt = tmplt.replace(' <mask>', (' '+tokenizer.mask_token)*trait_len).replace('<group>',group)
            # print(input_txt)
            target = get_lognorm_score(model, tokenizer, input_txt, trait)
            lps_score = target-prior
            # p_scores.append(prior)#
            # t_scores.append(target)#
            group_scores.append(lps_score)
            
        scores[group] = group_scores
        df = pd.DataFrame(data=scores)
        
    return df

# no space in sentences
def wordasso_zh(model, tokenizer, groups, traits, prior_group, tmplt="The <group> person is <mask>."):
    scores = {'traits': traits}
    for group in groups:
        group_scores = []
        for trait in traits:
            trait_ids = tokenizer.encode(trait, return_tensors='pt').squeeze()
            trait_len = list(trait_ids.shape)[0]-2
            # print(trait,trait_ids, trait_len)
            input_txt = tmplt.replace('<mask>', (tokenizer.mask_token)*trait_len).replace('<group>',prior_group)
            # print(input_txt)
            prior = get_lognorm_score(model, tokenizer, input_txt, trait)
            input_txt = tmplt.replace('<mask>', (tokenizer.mask_token)*trait_len).replace('<group>',group)
            # print(input_txt)
            target = get_lognorm_score(model, tokenizer, input_txt, trait)
            lps_score = target-prior
            # p_scores.append(prior)#
            # t_scores.append(target)#
            group_scores.append(lps_score)
            
        scores[group] = group_scores
        df = pd.DataFrame(data=scores)
        
    return df



def main():
   pass
    

if __name__ == "__main__":
	main()