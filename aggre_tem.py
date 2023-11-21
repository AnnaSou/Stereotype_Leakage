import pandas as pd
import csv
import os
import numpy as np

grouplist = ['Texan', 'gay man', 'Shanghainese person', 'software engineer', 'VDV soldier', 'Mormon person', 'Puerto Rican',
            'Asian person', 'civil servant', 'man', 'single mother', 'veteran', 'Black person', 'government official', 
            'Hui person', 'migrant worker', 'wealthy person', 'woman', 'muscovite', 'feminist', 'disabled person', 'poor person', 
            'Chechenets', 'housewife', 'Muslim person', 'immigrant', 'lesbian','Gujarati person', 'Brahmin person', 'Shudra person']

traitlist = ['powerless', 'powerful', 'low status', 'high status', 'dominated', 'dominant', 'poor', 'wealthy', 'confident', 
            'unconfident', 'unassertive', 'competitive', 'traditional', 'modern', 'religious', 'non religious', 
             'conservative', 'liberal', 'untrustworthy', 'trustworthy', 'dishonest', 'sincere', 'cold', 'warm', 
            'threatening', 'benevolent', 'repellent', 'likable', 'egotistic', 'altruistic', 'irrational', 'rational']

# GROUPS
group_zh2en = {'上海人':'Shanghainese person', '车臣人':'Chechenets', '黑人':'Black person', '家庭主妇':'housewife', '残障人士':'disabled person', 
            '单亲妈妈':'single mother', '女权主义者':'feminist', '穆斯林':'Muslim person', '德克萨斯人':'Texan', '俄罗斯空降军':'VDV soldier',
            '摩门教徒':'Mormon person', '公务员':'civil servant', '女人':'woman', '男人':'man', '退伍军人':'veteran', '移民':'immigrant', '莫斯科人':'muscovite', 
            '女同性恋':'lesbian', '穷人':'poor person', '农民工':'migrant worker', '富人':'wealthy person', '回民':'Hui person', '男同性恋':'gay man', 
            '波多黎各人':'Puerto Rican', '亚洲人':'Asian person', '软件工程师':'software engineer', '政府官员':'government official',
            '古吉拉特人':'Gujarati person','婆罗门':'Brahmin person','首陀罗':'Shudra person'}
csgroup_fem_ru2en = {'Домохозяйка':'housewife','Мать одиночка':'single mother', 'Женщина':'woman', 'Лесбиянка':'lesbian', 'Азиатка':'Asian person',
                    'Мусульманка':'Muslim person', 'Иммигрантка':'immigrant','Чиновница':'government official', 'Феминистка':'feminist',
                    'Госслужащая':'civil servant', 'Рабочая мигрантка':'migrant worker', 'Шанхайка':'Shanghainese person', 'Москвичка':'muscovite',
                    'Чеченка':'Chechenets', 'Техаска':'Texan', 'Пуэрто-риканка':'Puerto Rican','Гуджаратка':'Gujarati person','Брахманка':'Brahmin person','Шудрка':'Shudra person'}
csgroup_masc_ru2en ={ 'Мужчина':'man','Гей':'gay man','Программист':'software engineer','Богатый человек':'wealthy person','Бедный человек':'poor person',
                    'Инвалид':'disabled person','Азиат':'Asian person', 'Черный человек':'Black person', 'Мусульманин':'Muslim person', 'Иммигрант':'immigrant',
                    'Чиновник':'government official', 'Феминист':'feminist',
                    'Госслужащий':'civil servant', 'Ветеран':'veteran', 'Народ Хуэй':'Hui person', 'Рабочий мигрант':'migrant worker',
                    'Шанхайец':'Shanghainese person', 'Москвич':'muscovite', 'Чеченец':'Chechenets', 'Военослужащий ВДВ':'VDV soldier',
                    'Техасец':'Texan', 'Мормон':'Mormon person','Пуэрто-риканец':'Puerto Rican','Гуджартец':'Gujarati person','Брахманец':'Brahmin person','Шудр':'Shudra person'}
pluralgroup_pgroup_ru2en = {'домохозяйки':'housewife','матери одиночки':'single mother','мужчины':'man','женщины':'woman',
                            'геи':'gay man','лесбиянки':'lesbian','программисты':'software engineer' ,'богатые люди':'wealthy person',
                            'бедные люди':'poor person','инвалиды':'disabled person','азиаты':'Asian person','черные люди':'Black person',
                            'мусульмане':'Muslim person','иммигранты':'immigrant','чиновники':'government official','феминисты':'feminist',
                            'госслужащие':'civil servant','ветераны':'veteran','народ Хуэй':'Hui person','рабочие мигранты':'migrant worker',
                            'шанхайцы':'Shanghainese person','москвичи':'muscovite','чеченцы':'Chechenets','военослужащие ВДВ':'VDV soldier',
                            'техасцы':'Texan','мормоны':'Mormon person','пуэрто-риканцы':'Puerto Rican','гуджаратцы':'Gujarati person','брахманцы':'Brahmin person','шудры':'Shudra person' }
pluralgroup_cpgroup_ru2en = {'Домохозяйки':'housewife','Матери одиночки':'single mother','Мужчины':'man','Женщины':'woman',
                            'Геи':'gay man','Лесбиянки':'lesbian','Программисты':'software engineer','Богатые люди':'wealthy person','Бедные люди':'poor person',
                            'Инвалиды':'disabled person','Азиаты':'Asian person','Черные люди':'Black person','Мусульмане':'Muslim person',
                            'Иммигранты':'immigrant','Чиновники':'government official','Феминисты':'feminist','Госслужащие':'civil servant',
                            'Ветераны':'veteran','Народ Хуэй':'Hui person','Рабочие мигранты':'migrant worker',
                            'Шанхайцы':'Shanghainese person','Москвичи':'muscovite','Чеченцы':'Chechenets','Военослужащие ВДВ':'VDV soldier',
                            'Техасцы':'Texan','Мормоны':'Mormon person','Пуэрто-риканцы':'Puerto Rican', 'Гуджаратцы':'Gujarati person','Брахманцы':'Brahmin person','Шудры':'Shudra person'}

group_en2en = {'Texan':'Texan', 'gay man':'gay man', 'Shanghainese person':'Shanghainese person', 'software engineer':'software engineer', 
            'VDV soldier':'VDV soldier', 'Mormon person':'Mormon person','Puerto Rican':'Puerto Rican', 'Asian person':'Asian person', 
            'civil servant':'civil servant', 'man':'man', 'single mother':'single mother', 'veteran':'veteran', 'Black person':'Black person', 
            'government official':'government official', 'Hui person':'Hui person', 'migrant worker':'migrant worker', 'wealthy person':'wealthy person', 
            'woman':'woman', 'muscovite':'muscovite', 'feminist':'feminist', 'disabled person':'disabled person', 'poor person':'poor person', 
            'Chechenets':'Chechenets', 'housewife':'housewife', 'Muslim person':'Muslim person', 'immigrant':'immigrant', 'lesbian':'lesbian',
            'Gujarati person':'Gujarati person', 'Brahmin person':'Brahmin person', 'Shudra person':'Shudra person'}

sfgroup_hi2en = {'गृहिणी':'housewife','अकेली माँ':'single mother','महिला':'woman','समलैंगिक महिला':'lesbian'}
smgroup_hi2en = {'पुरुष':'man','समलैंगिक पुरुष':'gay man'}
pfgroup_hi2en = {'गृहिणियां':'housewife','अकेली माताएँ':'single mother','महिलाएं':'woman','समलैंगिक महिलाएं':'lesbian'}
pmgroup_hi2en = {'पुरुष':'man','समलैंगिक पुरुष':'gay man'}

sngroup_hi2en = {'सॉफ्टवेयर इंजीनियर':'software engineer','अमीर व्यक्ति':'wealthy person','गरीब व्यक्ति':'poor person',
                'अपंग व्यक्ति':'disabled person','एशियाई व्यक्ति':'Asian person','अश्वेत (ब्लैक) व्यक्ति':'Black person','मुसलमान व्यक्ति':'Muslim person',
            'अप्रवासी':'immigrant','सरकारी अधिकारी':'government official','नारीवादी':'feminist','लोक सेवक':'civil servant',
            'सेवानिवृत्त सैनिक':'veteran','हुई व्यक्ति':'Hui person','प्रवासी मजदूर':'migrant worker','शांगहैनी व्यक्ति':'Shanghainese person',
            'मास्कोवासी व्यक्ति':'muscovite','चेचन्या व्यक्ति':'Chechenets','रूसी हवाई सैनिक':'VDV soldier','टेक्सास के व्यक्ति':'Texan',
            'मॉर्मन व्यक्ति':'Mormon person','प्यूर्टो रिकान व्यक्ति':'Puerto Rican','गुजराती व्यक्ति':'Gujarati person','ब्राह्मण':'Brahmin person','दलित':'Shudra person'}
pngroup_hi2en = {'सॉफ्टवेयर इंजीनियर':'software engineer','अमीर लोग':'wealthy person','गरीब लोग':'poor person',
                'अपंग लोग':'disabled person','एशियाई लोग':'Asian person','अश्वेत (ब्लैक) लोग':'Black person','मुस्लिम लोग':'Muslim person','अप्रवासी':'immigrant',
            'सरकारी अधिकारी':'government official','नारीवादी':'feminist','लोक सेवक':'civil servant','सेवानिवृत्त सैनिक':'veteran','हुई लोग':'Hui person',
            'प्रवासी मजदूर':'migrant worker','शांगहैनी लोग':'Shanghainese person','मास्कोवासी लोग':'muscovite',
            'चेचन्या लोग':'Chechenets','रूसी हवाई सैनिक':'VDV soldier','टेक्सास के लोग':'Texan','मॉर्मन लोग':'Mormon person',
            'प्यूर्टो रिकान लोग':'Puerto Rican','गुजराती लोग':'Gujarati person','ब्राह्मण':'Brahmin person','दलित':'Shudra person'}

pgroup2_hi2en = {'गृहिणियों':'housewife','अकेली माओं':'single mother','पुरूषों':'man','महिलाओं':'woman','समलैंगिक पुरूषों':'gay man',
                'समलैंगिक महिलाओं':'lesbian','सॉफ्टवेयर इंजीनियरों':'software engineer','अमीर लोगों':'wealthy person',
            'गरीब लोगों':'poor person','अपंग लोगों':'disabled person','एशियाई लोगों':'Asian person',
            'अश्वेत (ब्लैक) लोगों':'Black person','मुस्लिम लोगों':'Muslim person','अप्रवासियों':'immigrant','सरकारी अधिकारीयों':'government official',
            'नारीवादियों':'feminist','लोक सेवकों':'civil servant','सेवानिवृत्त सैनिकों':'veteran','हुई लोगों':'Hui person',
            'प्रवासी मजदूरों':'migrant worker','शांगहैनी लोगों':'Shanghainese person','मास्कोवासी लोगों':'muscovite',
            'चेचन्या लोगों':'Chechenets','रूसी हवाई सैनिकों':'VDV soldier','टेक्सास के लोगों':'Texan','मॉर्मन लोगों':'Mormon person',
            'प्यूर्टो रिकान लोगों':'Puerto Rican','गुजराती लोगों':'Gujarati person','ब्राह्मणों':'Brahmin person','दलितों':'Shudra person'}

pluralgroup_en2en = {'Texans':'Texan', 'gay men':'gay man', 'Shanghainese people':'Shanghainese person', 'software engineers':'software engineer', 
                    'VDV soldiers':'VDV soldier', 'Mormon people':'Mormon person','Puerto Ricans':'Puerto Rican', 'Asian people':'Asian person', 
                    'civil servants':'civil servant', 'men':'man', 'single mothers':'single mother', 'veterans':'veteran', 'Black people':'Black person', 
                    'government officials':'government official', 'Hui people':'Hui person', 'migrant workers':'migrant worker', 'wealthy people':'wealthy person', 
                    'women':'woman', 'Muscovites':'muscovite', 'feminists':'feminist', 'disabled people':'disabled person', 'poor people':'poor person', 
                    'Chechentsy':'Chechenets', 'housewives':'housewife', 'Muslim people':'Muslim person', 'immigrants':'immigrant', 'lesbians':'lesbian',
                    'Gujarati people':'Gujarati person', 'Brahmin people':'Brahmin person', 'Shudra people':'Shudra person'}
pluralcapgroup_en2en = {'Texans':'Texan', 'Gay men':'gay man', 'Shanghainese people':'Shanghainese person', 'Software engineers':'software engineer', 
                    'VDV soldiers':'VDV soldier', 'Mormon people':'Mormon person','Puerto Ricans':'Puerto Rican', 'Asian people':'Asian person', 
                    'Civil servants':'civil servant', 'Men':'man', 'Single mothers':'single mother', 'Veterans':'veteran', 'Black people':'Black person', 
                    'Government officials':'government official', 'Hui people':'Hui person', 'Migrant workers':'migrant worker', 'Wealthy people':'wealthy person', 
                    'Women':'woman', 'Muscovites':'muscovite', 'Feminists':'feminist', 'Disabled people':'disabled person', 'Poor people':'poor person', 
                    'Chechentsy':'Chechenets', 'Housewives':'housewife', 'Muslim people':'Muslim person', 'Immigrants':'immigrant', 'Lesbians':'lesbian',
                    'Gujarati people':'Gujarati person', 'Brahmin people':'Brahmin person', 'Shudra people':'Shudra person'}

# TRAITS
trait_en2en = {'powerless':'powerless', 'powerful':'powerful', 'low-status':'low status', 'high-status':'high status', 'dominated':'dominated', 
            'dominant':'dominant', 'poor':'poor', 'wealthy':'wealthy', 'confident':'confident', 'unconfident':'unconfident', 'unassertive':'unassertive', 
            'competitive':'competitive', 'traditional':'traditional', 'modern':'modern', 'religious':'religious', 'non religious':'non religious', 
            'conservative':'conservative', 'liberal':'liberal', 'untrustworthy':'untrustworthy', 'trustworthy':'trustworthy', 'dishonest':'dishonest', 
            'sincere':'sincere', 'cold':'cold', 'warm':'warm', 'threatening':'threatening', 'benevolent':'benevolent', 'repellent':'repellent', 
            'likable':'likable', 'egotistic':'egotistic', 'altruistic':'altruistic', 'irrational':'irrational', 'rational':'rational'}
trait_zh2en = {'弱小的':'powerless','强大的':'powerful','低微的':'low status','显赫的':'high status','弱势的':'dominated','强势的':'dominant',
            '贫穷的':'poor','富有的':'wealthy','自信的':'confident','自卑的':'unconfident','谦逊的':'unassertive','争强好胜的':'competitive',
            '传统的':'traditional','新潮的':'modern','信教的':'religious','不信教的':'non religious',
            '保守的':'conservative','开明的':'liberal','不可信赖的':'untrustworthy','可靠的':'trustworthy','奸诈的':'dishonest','真诚的':'sincere',
            '冷酷的':'cold','温暖的':'warm','危险的':'threatening','友善的':'benevolent','讨厌的':'repellent','讨喜的':'likable','自私的':'egotistic',
            '无私的':'altruistic','非理性的':'irrational','理性的':'rational'}

trait_hi2en = {'शक्तिहीन':'powerless','शक्तिशाली':'powerful','निम्न स्थिति':'low status','उच्च स्थिति':'high status','अधीन':'dominated','प्रभावशाली':'dominant',
                'गरीब':'poor','अमीर':'wealthy','आत्मविश्वासहीन':'unconfident','आत्मविश्वासी':'confident','निष्क्रिय':'unassertive','प्रतिस्पर्द्धी':'competitive',
                'पारंपरिक':'traditional','आधुनिक':'modern','धार्मिक':'religious','गैर धार्मिक':'non religious','रूढ़िवादी':'conservative','उदारवादी (लिबरल)':'liberal',
                'बेईमान':'untrustworthy','भरोसेमंद':'trustworthy','बेईमान':'dishonest','ईमानदार':'sincere','कठोर':'cold','नेक':'warm',
                'भयंकर':'threatening','दयालु':'benevolent','घृणास्पद':'repellent','मनभावन':'likable','अहंकारी':'egotistic','परोपकारी':'altruistic',
                'तर्कहीन':'irrational','तर्कसंगत':'rational'}

#TRAITS
strait_fem_ru2en = {'не влиятельной':'powerless','влиятельной':'powerful','с низким социальным статусом':'low status',
                    'с высоким социальным статусом':'high status','подчиненной':'dominated','доминирующей':'dominant','бедной':'poor',
                    'богатой':'wealthy','неуверенной':'unconfident','уверенной':'confident','неконкурентноспособной':'unassertive',
                    'конкурентноспособной':'competitive','традиционной':'traditional','современной':'modern','религиозной':'religious',
                    'нерелигиозной':'non religious','консервативной':'conservative',
                    'либеральной':'liberal','ненадежной':'untrustworthy','надежной':'trustworthy','бесчестной':'dishonest','честной':'sincere',
                    'неприветливой':'cold','приветливой':'warm','опасной':'threatening','доброжелательной':'benevolent','отталкивающей':'repellent',
                    'привлекательной':'likable','эгоистичной':'egotistic','альтруистичной':'altruistic',
                    'иррациональной':'irrational','рациональной':'rational'}
strait_masc_ru2en ={'не влиятельным':'powerless','влиятельным':'powerful','с низким социальным статусом':'low status',
                    'с высоким социальным статусом':'high status','подчиненным':'dominated','доминирующим':'dominant','бедным':'poor',
                    'богатым':'wealthy','неуверенным':'unconfident','уверенным':'confident','неконкурентноспособным':'unassertive',
                    'конкурентноспособным':'competitive','традиционным':'traditional','современным':'modern','религиозным':'religious',
                    'нерелигиозным':'non religious','консервативным':'conservative',
                    'либеральным':'liberal','ненадежным':'untrustworthy','надежным':'trustworthy','бесчестным':'dishonest',
                    'честным':'sincere','неприветливым':'cold','приветливым':'warm','опасным':'threatening','доброжелательным':'benevolent',
                    'отталкивающим':'repellent','привлекательным':'likable','эгоистичным':'egotistic','альтруистичным':'altruistic',
                    'иррациональным':'irrational','рациональным':'rational'}
pluraltrait_ptrait_ru2en = {'не влиятельные':'powerless','влиятельные':'powerful','с низким социальным статусом':'low status',
                            'с высоким социальным статусом':'high status','подчиненные':'dominated','доминирующие':'dominant','бедные':'poor',
                            'богатые':'wealthy','неуверенные':'unconfident','уверенные':'confident','неконкурентноспособные':'unassertive',
                            'конкурентноспособные':'competitive','традиционные':'traditional','современные':'modern','религиозные':'religious',
                            'нерелигиозные':'non religious','консервативные':'conservative',
                            'либеральные':'liberal','ненадежные':'untrustworthy','надежные':'trustworthy','бесчестные':'dishonest','честные':'sincere',
                            'неприветливые':'cold','приветливые':'warm','опасные':'threatening','доброжелательные':'benevolent','отталкивающие':'repellent',
                            'привлекательные':'likable','эгоистичные':'egotistic','альтруистичные':'altruistic',
                            'иррациональные':'irrational','рациональные':'rational'}
pluraltrait_ptrait2_ru2en={'не влиятельными':'powerless','влиятельными':'powerful','с низким социальным статусом':'low status',
                            'с высоким социальным статусом':'high status','подчиненными':'dominated','доминирующими':'dominant','бедными':'poor',
                            'богатыми':'wealthy','неуверенными':'unconfident','уверенными':'confident','неконкурентноспособными':'unassertive',
                            'конкурентноспособными':'competitive','традиционными':'traditional','современными':'modern','религиозными':'religious',
                            'нерелигиозными':'non religious',
                            'консервативными':'conservative','либеральными':'liberal','ненадежными':'untrustworthy',
                            'надежными':'trustworthy','бесчестными':'dishonest','честными':'sincere','неприветливыми':'cold','приветливыми':'warm',
                            'опасными':'threatening','доброжелательными':'benevolent','отталкивающими':'repellent','привлекательными':'likable',
                            'эгоистичными':'egotistic','альтруистичными':'altruistic','иррациональными':'irrational','рациональными':'rational'}

def aggre_en(directory):
    result = {}
    for t in traitlist:
        result[t] = {}
        for g in grouplist:
            result[t][g] = []

    for fname in os.listdir(directory):
        fpath = os.path.join(directory, fname)
        print(fpath)
        group_dic = None
        if 'fact_belif' in fname:
            group_dic = pluralgroup_en2en
        elif fname in ['gt_tg0.csv','sing_plur21.csv','sing_plur22.csv','sing_plur23.csv','sing_plur24.csv','dec_int0.csv']:
            group_dic = pluralgroup_en2en
        elif fname in ['adv0.csv','adv1.csv','adv2.csv','comp0.csv','fact_socExpt0.csv','fact_socExpt1.csv','fact_socExpt2.csv','fact_socExpt3.csv','fact_socExpt4.csv','sing_plur20.csv']:
            group_dic = pluralcapgroup_en2en
        elif fname in ['sing_plur10.csv','sing_plur11.csv','sing_plur12.csv']:
            group_dic = group_en2en
        trait_dic = trait_en2en
        
        with open(fpath, "r") as infile:
            reader = csv.reader(infile, delimiter='\t')
            headers = next(reader)[2:]
            for row in reader:
                for key, value in zip(headers, row[2:]):
                    g = group_dic[key]
                    t = trait_dic[row[1]]
                    score = float(value)
                    result[t][g].append(score)
    return result

        
def aggre_zh(directory):
    result = {}
    for t in traitlist:
        result[t] = {}
        for g in grouplist:
            result[t][g] = []

    for fname in os.listdir(directory):
        fpath = os.path.join(directory, fname)
        print(fpath)
        group_dic = group_zh2en
        trait_dic = trait_zh2en
        
        with open(fpath, "r") as infile:
            reader = csv.reader(infile, delimiter='\t')
            headers = next(reader)[2:]
            for row in reader:
                for key, value in zip(headers, row[2:]):
                    g = group_dic[key]
                    t = trait_dic[row[1]]
                    score = float(value)
                    result[t][g].append(score)
    return result


def aggre_ru(directory):
    result = {}
    for t in traitlist:
        result[t] = {}
        for g in grouplist:
            result[t][g] = []

    for fname in os.listdir(directory):
        fpath = os.path.join(directory, fname)
        group_dic = None
        trait_dic = None
        if 'dec_int' in fname:
            group_dic = pluralgroup_pgroup_ru2en
        elif 'fact_belif' in fname:
            group_dic = pluralgroup_pgroup_ru2en
        elif fname in ['adv0.csv', 'comp0.csv', 'comp2.csv','fact_socExpt1.csv','fact_socExpt3.csv', 'gt_tg0.csv']:
            group_dic = pluralgroup_pgroup_ru2en
        elif 'sing_plur2' in fname:
            group_dic = pluralgroup_pgroup_ru2en
        elif fname in ['adv1.csv', 'comp1.csv','fact_socExpt0.csv','fact_socExpt2.csv']:
            group_dic = pluralgroup_cpgroup_ru2en
        elif "fem" in fname:
            group_dic = csgroup_fem_ru2en
        elif "masc" in fname:
            group_dic = csgroup_masc_ru2en
        
        if fname in ['fact_socExpt0.csv', 'fact_socExpt1.csv', 'gt_tg0.csv','sing_plur20.csv','sing_plur22.csv','sing_plur23.csv','sing_plur24.csv','sing_plur25.csv']:
            trait_dic = pluraltrait_ptrait_ru2en
        elif fname in ['fact_socExpt2.csv', 'fact_socExpt3.csv','sing_plur21.csv']:
            trait_dic = pluraltrait_ptrait2_ru2en
        elif 'adv' in fname or 'dec_int' in fname or 'fact_belif' in fname:
            trait_dic = pluraltrait_ptrait_ru2en
        elif 'comp' in fname:
            trait_dic = pluraltrait_ptrait2_ru2en
        elif 'fem' in fname:
            trait_dic = strait_fem_ru2en
        elif 'masc' in fname:
            trait_dic = strait_masc_ru2en
        
        
        with open(fpath, "r") as infile:
            reader = csv.reader(infile, delimiter='\t')
            headers = next(reader)[2:]
            for row in reader:
                for key, value in zip(headers, row[2:]):
                    g = group_dic[key]
                    t = trait_dic[row[1]]
                    score = float(value)
                    result[t][g].append(score)
    return result


def aggre_hi(directory):
    result = {}
    for t in traitlist:
        result[t] = {}
        for g in grouplist:
            result[t][g] = []

    for fname in os.listdir(directory):
        fpath = os.path.join(directory, fname)
        print(fpath)
        group_dic = None
        if "fact_socExpt" in fname or "gt_tg" in fname:
            group_dic = pgroup2_hi2en
        elif 'fact_belif' in fname or 'comp' in fname or "dec_int" in fname or 'adv' in fname:
            if '-masc' in fname:
                group_dic = pmgroup_hi2en
            elif '-fem' in fname:
                group_dic = pfgroup_hi2en
            elif '-neutral' in fname:
                group_dic = pngroup_hi2en
        elif 'sing_plur10-' in fname or 'sing_plur11-' in fname or 'sing_plur12-' in fname or 'sing_plur19-' in fname:
            if '-masc' in fname:
                group_dic = smgroup_hi2en
            elif '-fem' in fname:
                group_dic = sfgroup_hi2en
            elif '-neutral' in fname:
                group_dic = sngroup_hi2en
        else:
            if '-masc' in fname:
                group_dic = pmgroup_hi2en
            elif '-fem' in fname:
                group_dic = pfgroup_hi2en
            elif '-neutral' in fname:
                group_dic = pngroup_hi2en


        trait_dic = trait_hi2en
        
        with open(fpath, "r") as infile:
            reader = csv.reader(infile, delimiter='\t')
            headers = next(reader)[2:]
            for row in reader:
                for key, value in zip(headers, row[2:]):
                    g = group_dic[key]
                    t = trait_dic[row[1]]
                    score = float(value)
                    result[t][g].append(score)
    return result

def main():
    for lan in ["bert-EN",'mbert-EN',"mt5-EN"]:
        result = aggre_en(lan)
        outfname = "aggregated_scores/"
        if not os.path.exists(outfname):
            os.makedirs(outfname)
        outfname = outfname + lan+'_scores.csv'
        with open(outfname,'w') as f:
            writer = csv.writer(f)
            writer.writerow(['trait']+grouplist)
            for t in traitlist:
                row = [t]
                for g in grouplist:
                    scorelist = result[t][g]
                    score = sum(scorelist)/len(scorelist)
                    row.append(score)
                writer.writerow(row)
    
    for lan in ["bert-ZH",'mbert-ZH',"mt5-ZH"]:
        result = aggre_zh(lan)
        outfname = "aggregated_scores/"
        if not os.path.exists(outfname):
            os.makedirs(outfname)
        outfname = outfname + lan+'_scores.csv'
        with open(outfname,'w') as f:
            writer = csv.writer(f)
            writer.writerow(['trait']+grouplist)
            for t in traitlist:
                row = [t]
                for g in grouplist:
                    scorelist = result[t][g]
                    score = sum(scorelist)/len(scorelist)
                    row.append(score)
                writer.writerow(row)

    for lan in ["bert-RU",'mbert-RU',"mt5-RU"]:
        result = aggre_ru(lan)
        outfname = "aggregated_scores/"
        if not os.path.exists(outfname):
            os.makedirs(outfname)
        outfname = outfname + lan+'_scores.csv'
        with open(outfname,'w') as f:
            writer = csv.writer(f)
            writer.writerow(['trait']+grouplist)
            for t in traitlist:
                row = [t]
                for g in grouplist:
                    scorelist = result[t][g]
                    score = sum(scorelist)/len(scorelist)
                    row.append(score)
                writer.writerow(row)

    for lan in ["bert-HI","mbert-HI","mt5-HI"]:
        result = aggre_hi(lan)
        outfname = "aggregated_scores/"
        if not os.path.exists(outfname):
            os.makedirs(outfname)
        outfname = outfname + lan+'_scores.csv'
        with open(outfname,'w') as f:
            writer = csv.writer(f)
            writer.writerow(['trait']+grouplist)
            for t in traitlist:
                row = [t]
                for g in grouplist:
                    if t == 'untrustworthy':
                        scorelist = result['dishonest'][g]
                    else:
                        scorelist = result[t][g]
                    score = sum(scorelist)/len(scorelist)
                    row.append(score)
                writer.writerow(row)



if __name__ == "__main__":
	main()