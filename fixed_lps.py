from transformers import MT5Tokenizer, MT5ForConditionalGeneration

import torch
from torch.nn import functional as F

import numpy as np
import pandas as pd
import math
import statistics
import random

random.seed(10)

def equal(l1,l2):
    assert len(l1) == len(l2)
    for x,y in zip(l1,l2):
        if x!=y: return False
    return True

def get_log_prob(model, tokenizer, sentence, trait):
    # print("Input: ",sentence)
    input_ids = tokenizer(sentence, return_tensors="pt").input_ids
    target_output = "<extra_id_0>{} <extra_id_1>".format(trait)
    labels = tokenizer(target_output, return_tensors="pt").input_ids
    # print("Label: ", target_output)
    loss = model(input_ids=input_ids, labels=labels).loss
    log_prob = -1*loss.item() # cross entropy loss -- thus should be in log
    # print(log_prob)
    return log_prob

def mt5_wordasso(model, tokenizer, groups, traits, prior_group, tmplt="The <group> person is <mask>.", traits_for_prior=None, tplt_for_prior=None):
    scores = {'traits': traits}
    for group in groups:
        group_scores = []
        if traits_for_prior == None:
            traits_for_prior = [None]*len(traits)
        for trait, trait_fp in zip(traits, traits_for_prior):
            trait = ' '+trait
            input_txt = tmplt.replace('<mask>', '<extra_id_0>').replace('<group>',prior_group)
            # print(input_txt)
            if trait_fp == None and tplt_for_prior == None:
                prior = get_log_prob(model, tokenizer, input_txt, trait)
            elif trait_fp != None:
                prior = get_log_prob(model, tokenizer, input_txt, trait_fp)
            elif tplt_for_prior != None:
                input_txt = tplt_for_prior.replace('<mask>', '<extra_id_0>').replace('<group>',prior_group)
                prior = get_log_prob(model, tokenizer, input_txt, trait)
            input_txt = tmplt.replace('<mask>', '<extra_id_0>').replace('<group>',group)
            target = get_log_prob(model, tokenizer, input_txt, trait)
            lps_score = target-prior
            # p_scores.append(prior)#
            # t_scores.append(target)#
            group_scores.append(lps_score)
        scores[group] = group_scores
        df = pd.DataFrame(data=scores)
        
    return df

def mt5_wordasso_zh(model, tokenizer, groups, traits, prior_group, tmplt="The <group> person is <mask>."):
    scores = {'traits': traits}
    for group in groups:
        group_scores = []
        for trait in traits:
            input_txt = tmplt.replace('<mask>', ' <extra_id_0>').replace('<group>',prior_group)
            # print(input_txt)
            prior = get_log_prob(model, tokenizer, input_txt, trait)
            input_txt = tmplt.replace('<mask>', ' <extra_id_0>').replace('<group>',group)
            # print(input_txt)
            target = get_log_prob(model, tokenizer, input_txt, trait)
            lps_score = target-prior
            # p_scores.append(prior)#
            # t_scores.append(target)#
            group_scores.append(lps_score)
        scores[group] = group_scores
        df = pd.DataFrame(data=scores)
        
    return df



def main():
    model = MT5ForConditionalGeneration.from_pretrained('google/mt5-base',cache_dir="../../hf_cache_dir/")
    tokenizer = MT5Tokenizer.from_pretrained('google/mt5-base', cache_dir="../../hf_cache_dir/")
    # groups = ['single mother']
    # # traits = {'altruistic':None}
    # traits = ['dominated']
    # prior_group = 'person'
    # template = 'That <group> is <mask>.'
    # wordasso(model, tokenizer, groups, traits,prior_group, template)
    # groups = ['女人']
    # traits = ['弱小的','强大的'] 
    # prior_group = '人'
    # template = '我认为<group>是<mask>的。'
    # print(mt5_wordasso_zh(model, tokenizer, groups, traits,prior_group, template))
    
    # groups = ['Богатый человек']
    # traits = ['традиционным']
    # prior_group = 'Человек'
    # template = '<group> кажется <mask>.'
    # print(mt5_wordasso(model, tokenizer, groups, traits,prior_group, template))

    groups = {'अकेली माँ'} # single mother
    traits = ['शक्तिहीन','शक्तिशाली'] # powerless, powerful
    template = 'वह <group> <mask> है।'
    prior_tplt = 'वह <group> <mask> है।'
    print(mt5_wordasso(model, tokenizer, groups, traits, template, tplt_for_prior = prior_tplt))


if __name__ == "__main__":
	main()