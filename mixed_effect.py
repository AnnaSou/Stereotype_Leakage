import os
import csv
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as stats

shared_groups = ['woman', 'man','gay man','lesbian','housewife','single mother','software engineer','disabled person',
             'wealthy person','poor person','Asian person','Black person','Muslim person', 'immigrant','government official', 
             'civil servant','feminist','veteran']
non_shared_groups = {'EN':['Texan','Puerto Rican','Mormon person'], 
                     'RU':['muscovite','Chechenets','VDV soldier'],
                     'ZH':['Shanghainese person','Hui person', 'migrant worker'],
                     'HI':['Gujarati person', 'Brahmin person', 'Shudra person']}
pluralgroup_en2en = {'Texans':'Texan', 'gay men':'gay man', 'Shanghainese people':'Shanghainese person', 'software engineers':'software engineer', 
                    'VDV soldiers':'VDV soldier', 'Mormon people':'Mormon person','Puerto Ricans':'Puerto Rican', 'Asian people':'Asian person', 
                    'civil servants':'civil servant', 'men':'man', 'single mothers':'single mother', 'veterans':'veteran', 'Black people':'Black person', 
                    'goverment officials':'government official', 'Hui people':'Hui person', 'migrant workers':'migrant worker', 'wealthy people':'wealthy person', 
                    'women':'woman', 'Muscovites':'muscovite', 'feminists':'feminist', 'disabled people':'disabled person', 'poor people':'poor person', 
                    'Chechentsy':'Chechenets', 'housewives':'housewife', 'Muslim people':'Muslim person', 'immigrants':'immigrant', 'lesbians':'lesbian',
                    'Gujarati people':'Gujarati person', 'Brahmin people':'Brahmin person', 'Shudra people':'Shudra person'}
traits = ['powerless', 'powerful', 'low status', 'high status', 'dominated', 'dominant',
        'poor', 'wealthy', 'confident', 'unconfident', 'unassertive', 'competitive',
        'traditional', 'modern', 'religious', 'non religious', 'conservative',
        'liberal', 'untrustworthy' ,'trustworthy' ,'dishonest', 'sincere', 'cold',
        'warm' ,'threatening', 'benevolent', 'repellent', 'likable', 'egoistic',
        'altruistic', 'irrational', 'rational']



grouplist = shared_groups + non_shared_groups['EN'] + non_shared_groups['ZH'] + non_shared_groups['RU'] + non_shared_groups['HI']
# print(grouplist)

data = {}
for lan in ['EN', 'ZH', 'HI', 'RU']:
    mbert_df = pd.read_csv("aggregated_scores/"+"mbert-"+lan+"_scores.csv")
    bert_df = pd.read_csv("aggregated_scores/"+"bert-"+lan+"_scores.csv")
    mt5_df = pd.read_csv("aggregated_scores/"+"ILPS/"+"mt5-"+lan+"_scores.csv")
    human_df = pd.read_csv("dataset/aggregated_data_"+lan+".csv")
    human_df = human_df.replace([999],0.0) 
    gpt_df =  pd.read_csv("gpt/"+lan+".csv")
    
    for g in grouplist:
        mbert_df[g] = mbert_df[g].multiply(100)
        bert_df[g] = bert_df[g].multiply(100)
        mt5_df[g] = mt5_df[g].multiply(10)
        human_df[g] = human_df[g].abs() # negative scores for one side of the polar traits should be positive here
        gpt_df[g] = gpt_df[g].abs()
    
    mbert_col = []
    bert_col = []
    human_col = []
    mt5_col = []
    gpt_col = []
    gs = []
    ts = []
    for g in grouplist:
        for t in traits:
            mbert_col.append(float(mbert_df.loc[mbert_df['trait']==t][g].values[0]))
            human_col.append(float(human_df.loc[human_df['Traits']==t][g].values[0]))
            bert_col.append(float(bert_df.loc[bert_df['trait']==t][g].values[0]))
            mt5_col.append(float(mt5_df.loc[mt5_df['trait']==t][g].values[0]))
            gpt_col.append(float(gpt_df.loc[gpt_df['Traits']==t][g].values[0]))
            gs.append(g)
            ts.append(t)
    if "group" not in data:
        data["group"] = gs
        data["trait"] = ts
    data["mbert_"+lan] = mbert_col
    data["bert_"+lan] = bert_col
    data["mt5_"+lan] = mt5_col
    data["gpt_"+lan] = gpt_col
    data["human_"+lan] = human_col

    
df = pd.DataFrame(data)


# mBERT
model = smf.mixedlm("mbert_HI ~ human_EN + human_RU + human_ZH + human_HI + bert_HI ", df,
                    groups= "trait").fit()
print(model.summary())


# MT5
model = smf.mixedlm("mt5_RU ~ human_EN + human_RU + human_ZH + human_HI ", df,
                    groups= "trait").fit()
print(model.summary())

# # GPT-3.5
model = smf.mixedlm("gpt_RU ~ human_EN + human_RU + human_ZH + human_HI", df,
                    groups= "trait").fit()
print(model.summary())

