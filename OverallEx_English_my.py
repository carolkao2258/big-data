# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 17:02:35 2022

@author: FJUSER211027A
"""

import pandas as pd

import os
os.chdir("C:\\Users\\uer\\Downloads")
file = "reddit_dataframe.pkl"
df1 = pd.read_pickle(file)
dfs=df1.copy()
dfs=dfs.iloc[0:100,:] #資料太大很耗時, 這裡我用小部分來做
###############################
###Part I: Identify the Nose###
###############################
import re

RE_SUSPICIOUS = re.compile(r'[&#<>{}\[\]\\]')

def impurity(text, min_len=10):
    """returns the share of suspicious characters in a text"""
    if text == None or len(text) < min_len:
        return 0
    else:
        return len(RE_SUSPICIOUS.findall(text))/len(text)
    
dfs['impurity'] = dfs['text'].apply(impurity, min_len=10)    


# get the top 3 records
dfs[['text', 'impurity']].sort_values(by='impurity', ascending=False).head(3)



#####################################################
###Part II: Removing Nose with Regular Expressions###
#####################################################
#remark: html.unescape
import html
p = '&lt;abc&gt;' #&lt; and &gt; are special simbles in html
#not showing in text example
txt= html.unescape(p)
print (txt)

import html

def clean(text):
    # convert html escapes like &amp; to characters.
    text = html.unescape(text) #in this example, this part does nothing
    # tags like <tab>
    text = re.sub(r'<[^<>]*>', ' ', text)
    # markdown URLs like [Some text](https://....)
    text = re.sub(r'\[([^\[\]]*)\]\([^\(\)]*\)', ' ', text)
    # text or code in brackets like [0]
    text = re.sub(r'\[[^\[\]]*\]', ' ', text)
    # standalone sequences of specials, matches &# but not #cool
    text = re.sub(r'(?:^|\s)[&#<>{}\[\]+|\\:-]{1,}(?:\s|$)', ' ', text)
    # standalone sequences of hyphens like --- or ==
    text = re.sub(r'(?:^|\s)[\-=\+]{2,}(?:\s|$)', ' ', text)
    # sequences of white spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


dfs['clean_text'] = dfs['text'].apply(clean)
dfs['impurity']   = dfs['clean_text'].apply(impurity, min_len=20)

dfs[['clean_text', 'impurity']].sort_values(by='impurity', ascending=False) \
                              .head(3)


####################################################
###Part III: Character Normalization with textacy###
####################################################
import textacy.preprocessing as tprep
#you need to install textacy
def normalize(text):
    text = tprep.normalize.hyphenated_words(text)
    text = tprep.normalize.quotation_marks(text)
    text = tprep.normalize.unicode(text)
    text = tprep.remove.accents(text)
    return text

dfs['clean_text'] = dfs['clean_text'].apply(normalize)


#############################################
###Part IV: Character Masking with textacy###
############################################# 

from textacy.preprocessing import replace
dfs['clean_text'] = dfs['clean_text'].apply(replace.urls)

##最後整理
dfs.rename(columns={'text': 'raw_text', 'clean_text': 'text'}, inplace=True)
dfs.drop(columns=['impurity'], inplace=True)


##########################
###Liguistic Processing###
##########################

#All steps in one by using spacy

import spacy
nlp = spacy.load('en_core_web_sm')

dfs['doc']=dfs['text'].apply(nlp)
ti=dfs['doc'][0]

import textacy

def extract_lemmas(doc, **kwargs):
    return [t.lemma_ for t in textacy.extract.basics.words(doc, **kwargs)]


dfs['lemmas'] = dfs['doc'].apply(extract_lemmas, include_pos=['ADJ', 'NOUN'])
dfs['lemmas']

#############
#Freq Charts#
#############

from collections import Counter
counter = Counter()#use a empty string first
dfs['lemmas'].map(counter.update)

print(counter.most_common(5))
# transform counter into data frame
min_freq=2
#transform dict into dataframe
freq_df = pd.DataFrame.from_dict(counter, orient='index', columns=['freq'])
freq_df = freq_df.query('freq >= @min_freq')
freq_df.index.name = 'token'
freq_df = freq_df.sort_values('freq', ascending=False)
freq_df.head(15)

ax = freq_df.head(15).plot(kind='barh', width=0.95, figsize=(8,3))
ax.invert_yaxis()
ax.set(xlabel='Frequency', ylabel='Token', title='Top Words')

###Creating Word Clouds
from matplotlib import pyplot as plt
from wordcloud import WordCloud ###
from collections import Counter ###

wordcloud = WordCloud(font_path="SimHei.ttf", background_color="white")
wordcloud.generate_from_frequencies(freq_df['freq'])
#plt.figure(figsize=(20,10)) 
plt.imshow(wordcloud)