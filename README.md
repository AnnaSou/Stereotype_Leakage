# Multilingual large language models leak human stereotypes across language boundaries

This repository contains the code and dataset from our paper [].

## Dataset
We conduct human study in English, Russian, Chinese, and Hindi to collect human judgments on stereotypes towards 30 social groups within three categories, across all 16 pairs of traits from the ABC model. 
Please read the datasheet in the `dataset` folder for details of the dataset.

## Measure stereotypic associations in LLMs
Run `python asso.py` to get the group-trait associations scores with respect to each template.
Run `python aggre_tem.py` to get the aggregated group-trait association scores.

## Measure stereotype leakage
Run `python mixed_effect.py` to get results from the mixed effect analysis.

