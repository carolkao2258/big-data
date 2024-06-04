# -*- coding: utf-8 -*-
"""
Created on Mon May 24 10:08:27 2021

@author: Shawn
"""

import pandas as pd

import os
os.chdir("C:\Users\uer\Downloads\\Sec 02 Regular Expression and Coupus Cleaning")

file = "reddit_dataframe.pkl"
df = pd.read_pickle(file)

YahooNews=pd.read_csv('研究報告—個股09-12_utf8.csv')

text = """
After viewing the [PINKIEPOOL Trailer](https://www.youtu.be/watch?v=ieHRoHUg)
it got me thinking about the best match ups.
<lb>Here's my take:<lb><lb>[](/sp)[](/ppseesyou) Deadpool<lb>[](/sp)[](/ajsly)
Captain America<lb>"""

text1 = "聚焦國內、抗衰退、有派息！分析師：康卡斯特還能再漲20%,\
鉅亨網 ,20190906,13:40,康卡斯特 (Comcast) (CMCSA-US) 週四獲券商調高評級至等同買進，股價再添助力。 Oppenheimer 分析師 Timothy Horan 認為，寬頻營收成長、利潤擴大和相對低的估值都是買進 Comcast 股票的理由，儘管目前該股今年已上漲約 35%(含股利)，\
<em>Horan 估計該股仍遭低估近 20%。</em> <h6>背景</h6> 隨著多數媒體和電信產業發展，Comcast 去年也加入併購大戰，去年與迪士尼 (DIS) 競購 21 世紀福斯的娛樂資產，接著去年秋季又參與歐洲付費電視營運商 Sky 的拍賣，雖然不敵迪士尼，未能買下福斯資產，\
但卻成功以交易價格 461 億美元 (含債務) 贏得 Sky 的控制權，收購其位於歐洲 7 個國家 2300 萬個衛星電視用戶，以及其 Now TV 串流平台約 200 萬名訂閱戶。 投資者最初並未看好這些交易，因為他們擔心 Comcast 買貴了，且衛星電視趨勢呈負面，\
因此 Comcast 去年股價以下跌 15% 收場，直至今年才回彈。Comcast 今年因核心有線電視業務在 Xfinity 品牌的支撐下，成績斐然，因此穩定推升股價。 投資者對經濟成長趨緩、貿易戰、今年債券殖利率大跌的擔憂心裡凸顯了有線電視「聚焦國內」、「抗經濟衰退」的商業模式，\
且派息股在此時又更加迷人，因此今年迄今 Comcast 股價約上漲 35%，優於同期 S&amp;P 500 指數 18.8% 的收益表現，另外同業 Charter Communications (CHTR-US) 上漲 47.2%、Altice USA (ATUS-US) 上漲 75.5% 等。 \
<figure><img src=""https://s.yimg.com/uu/api/res/1.2/GgJDsAmVP.vyCqSfK5rEGg--/YXBwaWQ9eXRhY2h5b247cT03NTs-/https://media.zenfs.com/zh/cnyes.com/59458dcf186ed236719d9446e6762cf3"">\
</figure><h6>撐過剪線潮？</h6> 隨著有線電視訂閱人數不斷下降，網路與有線電視業務不再綁在一起出售，今年有線電視公司不斷發展寬頻網路事業，投資者也有了新的認識。 Oppenheimer 分析師 Horan 認為，其寬頻力道持續並支撐利潤上升，降低資本支出的要求。 \
「拆分經濟有助擴大利潤，因為顧客更傾向單獨購買 Comcast 的寬頻服務。由於營收混合方式轉向利潤更高的寬頻 (估計 70% EBITDA 利潤)，而非影視 (估計 20% 利潤)，有線電纜的利潤應該會持續擴大。」\
<figure><img src=""https://s1.yimg.com/uu/api/res/1.2/oo2yUJD0PVnhKFw.QeWs.w--/YXBwaWQ9eXRhY2h5b247cT03NTs-/https://media.zenfs.com/zh/cnyes.com/f67abbf18565746acab385220d1a4b14""></figure>Horan 看到，\
Comcast 持續將現行網路訂閱戶推向數據傳輸更快的服務，因其價位更高。同業 Charter 計畫明年起調漲價格，Horan 預期 Comcast 也將跟進，因此可望調升每月用戶的平均一年營收 4%。 <h6>前景</h6> Horan 估計 Comcast 今年 Ebitda 上升 2 個百分點，\
接著 2020 年再上升 0.7 個百分點達 32.2%。與此同時，他估計該公司資本支出穩定，且今年營收成長 3.5%，來年成長 4.5%。 其結果是 Comcast 2020 年自由現金流上升 15%，至估計近 190 億美元，該公司屆時可能用於派息、執行庫藏股或者清償債務。\
?Horan 對該公司目標價 54 美元，是基於 2020 年現金流收益 7.5% 來估計。"

import regex as re1

def tokenize(text):
    return re1.findall(r'[\w-]*\p{L}[\w-]*',text)

tokens = tokenize(text)
tokens[0]

from collections import Counter

tokens = tokenize('She likes my cats and my cats like my sofa')

counter = Counter(tokens)

more_tokens=tokenize('She likes dogs and cats')
counter.update(more_tokens)
counter

df['tokens']=df['text'].apply(tokenize)
df['tokens'][10]

counter = Counter()
df['tokens'].map(counter.update)

print(counter.most_common(5))

min_freq = 2
freq_df = pd.DataFrame.from_dict\
    (counter,orient='index',columns=['freq'])

freq_df.index.name = 'token'
freq_df = freq_df.sort_values('freq',ascending=False)
freq_df.head(10)

ax=freq_df.head(15).plot(kind='barh')
ax.invert_yaxis()

from matplotlib import pyplot as plt
from wordcloud import WordCloud

wordcloud = WordCloud(background_color='white')
wordcloud.generate_from_frequencies(freq_df['freq'])
plt.imshow(wordcloud)

#conda install cctbx202211::pillow

import re

RE_SUSPICIOUS = re.compile(r'[&#<>{}\[\]\\]')

def impurity(text, min_len=10):
    """returns the share of suspicious characters in a text"""
    if text == None or len(text) < min_len:
        return 0
    else:
        return len(RE_SUSPICIOUS.findall(text))/len(text)

print(impurity(text))

print(impurity(text1))

df['impurity'] = df['text'].apply(impurity, min_len=10)
df.columns
df['impurity'][:2]
df[['text', 'impurity']].sort_values(by='impurity', ascending=False).head(3)

YahooNews['Impurity']=YahooNews['Context'].apply(impurity, min_len=10)
YahooNews.columns
YahooNews[['Context', 'Impurity']].sort_values(by='Impurity', ascending=False).head(3)

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
    text = re.sub(r'\[([^\[\]]*)\]\([^\(\)]*\)', r'\1', text)
    # text or code in brackets like [0]
    text = re.sub(r'\[[^\[\]]*\]', ' ', text)
    # standalone sequences of specials, matches &# but not #cool
    text = re.sub(r'(?:^|\s)[&#<>{}\[\]+|\\:-]{1,}(?:\s|$)', ' ', text)
    # standalone sequences of hyphens like --- or ==
    text = re.sub(r'(?:^|\s)[\-=\+]{2,}(?:\s|$)', ' ', text)
    # sequences of white spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

clean_text = clean(text)
print(clean_text)
print("Impurity:", impurity(clean_text))

clean_text1 = clean(text1)
print(clean_text1)
print("Impurity:", impurity(clean_text1))


df['clean_text'] = df['text'].map(clean)
df['impurity']   = df['clean_text'].apply(impurity, min_len=20)

df[['clean_text', 'impurity']].sort_values(by='impurity', ascending=False) \
                              .head(3)
                              
                              
YahooNews['Clean_text'] = YahooNews['Context'].map(clean)
YahooNews['Impurity']   = YahooNews['Clean_text'].apply(impurity, min_len=20)

YahooNews[['Clean_text', 'Impurity']].sort_values(by='Impurity', ascending=False) \
                              .head(3) 


text = "The café “Saint-Raphaël” is loca-\nted on Côte dʼAzur."
#print(textacy.__version__)
import textacy.preprocessing as tprep
#you need to install textacy
def normalize(text):
    text = tprep.normalize.hyphenated_words(text)
    text = tprep.normalize.quotation_marks(text)
    text = tprep.normalize.unicode(text)
    text = tprep.remove.accents(text)
    return text

print(normalize(text))


from textacy.preprocessing import replace

text = "Check out https://spacy.io/usage/spacy-101"

# using default substitution _URL_
print(replace.urls(text))
print(replace.urls(text1))

#for df:

df['clean_text'] = df['clean_text'].map(replace.urls)
df['clean_text'] = df['clean_text'].map(normalize)

YahooNews['Clean_text']=YahooNews['Clean_text'].map(replace.urls)

df.rename(columns={'text': 'raw_text', 'clean_text': 'text'}, inplace=True)
df.drop(columns=['impurity'], inplace=True)

YahooNews.rename(columns={'Context': 'Raw_text', 'Clean_text': 'Context'}, inplace=True)
YahooNews.drop(columns=['Impurity'], inplace=True)






###Linguistic Processing

##Because this part is very complicated, I will demo the whole process first.
##Then I will explain each part later

###English###
dfs=df.iloc[:200,:]

import spacy
nlp = spacy.load('en_core_web_sm')

dfs['doc']=dfs['text'].apply(nlp)

import textacy

def extract_lemmas(doc, **kwargs):
    return [t.lemma_ for t in textacy.extract.basics.words(doc, **kwargs)]


dfs['lemmas'] = dfs['doc'].apply(extract_lemmas, include_pos=['ADJ', 'NOUN'])
dfs['lemmas']

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

wordcloud = WordCloud(background_color='white')
wordcloud.generate_from_frequencies(freq_df['freq'])
plt.imshow(wordcloud)


###Chinese###
#1加入繁體詞典
import jieba

jieba.set_dictionary('./結巴/dict.txt.big.txt')
stopwords1 = [line.strip() for line in open('./結巴/stopWords.txt', 'r', encoding='utf-8').readlines()]

def remove_stop(text):
    c1=[]
    for w in text:
        if w not in stopwords1:
            c1.append(w)
    c2=[i for i in c1 if i.strip() != '']
    return c2



YahooNews['tokens']=YahooNews['Context'].apply(jieba.cut)
YahooNews['tokens_new']=YahooNews['tokens'].apply(remove_stop)
YahooNews.iloc[0,:]


#Freq charts
from collections import Counter
counter = Counter()#use a empty string first
YahooNews['tokens_new'].apply(counter.update)
print(counter.most_common(15))

import seaborn as sns
sns.set(font="SimSun")
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

wordcloud = WordCloud(font_path="./結巴/SimHei.ttf", background_color="white")
wordcloud.generate_from_frequencies(freq_df['freq'])
#plt.figure(figsize=(20,10)) 
plt.imshow(wordcloud)




###Some Details

text = "The café “Saint-Raphaël” is loca-\nted on Côte dʼAzur."
#print(textacy.__version__)
import textacy.preprocessing as tprep
#you need to install textacy
def normalize(text):
    text = tprep.normalize.hyphenated_words(text)
    text = tprep.normalize.quotation_marks(text)
    text = tprep.normalize.unicode(text)
    text = tprep.remove.accents(text)
    return text

print(normalize(text))


from textacy.preprocessing import replace

text = "Check out https://spacy.io/usage/spacy-101"

# using default substitution _URL_
print(replace.urls(text))
print(replace.urls(text1))

#for df:

df['clean_text'] = df['clean_text'].map(replace.urls)
df['clean_text'] = df['clean_text'].map(normalize)

YahooNews['Clean_text']=YahooNews['Clean_text'].map(replace.urls)

df.rename(columns={'text': 'raw_text', 'clean_text': 'text'}, inplace=True)
df.drop(columns=['impurity'], inplace=True)

YahooNews.rename(columns={'Context': 'Raw_text', 'Clean_text': 'Context'}, inplace=True)
YahooNews.drop(columns=['Impurity'], inplace=True)

text = """
2019-08-10 23:32: @pete/@louis - I don't have a well-designed 
solution for today's problem. The code of module AC68 should be -1. 
Have to think a bit... #goodnight ;-) 😩😬"""



tokens = re.findall(r'\w\w+', text)
print(*tokens, sep='|')

RE_TOKEN = re.compile(r"""
               ( [#]?[@\w'’\.\-\:]*\w     # words, hash tags and email adresses
               | [:;<]\-?[\)\(3]          # coarse pattern for basic text emojis
               | [\U0001F100-\U0001FFFF]  # coarse code range for unicode emojis
               )
               """, re.VERBOSE)

def tokenize(text):
    return RE_TOKEN.findall(text)

tokens = tokenize(text)
print(*tokens, sep='|')



import nltk

nltk.download('punkt') ###
tokens = nltk.tokenize.word_tokenize(text)
print(*tokens, sep='|')


tokenizer = nltk.tokenize.RegexpTokenizer(RE_TOKEN.pattern, flags=re.VERBOSE)
tokens = tokenizer.tokenize(text)
print(*tokens, sep='|')


tokenizer = nltk.tokenize.TweetTokenizer()
tokens = tokenizer.tokenize(text)
print(*tokens, sep='|')


tokenizer = nltk.tokenize.ToktokTokenizer()
tokens = tokenizer.tokenize(text)
print(*tokens, sep='|')


import spacy
nlp = spacy.load('en_core_web_sm')

nlp.pipeline
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

nlp = spacy.load("en_core_web_sm")
text = "My best friend Ryan Peters likes fancy adventure games."
doc = nlp(text)

for token in doc:
    print(token, end="|")
    
    
def display_nlp(doc, include_punct=False):
    """Generate data frame for visualization of spaCy tokens."""
    rows = []
    for i, t in enumerate(doc):
        if not t.is_punct or include_punct:
            row = {'token': i,  'text': t.text, 'lemma_': t.lemma_, 
                   'is_stop': t.is_stop, 'is_alpha': t.is_alpha,
                   'pos_': t.pos_, 'dep_': t.dep_, 
                   'ent_type_': t.ent_type_, 'ent_iob_': t.ent_iob_}
            rows.append(row)
    
    df = pd.DataFrame(rows).set_index('token')
    df.index.name = None
    return df    
    
display_nlp(doc)  
    
  
    
    
   
nlp = spacy.load('en_core_web_sm') ###
text = "Dear Ryan, we need to sit down and talk. Regards, Pete"
doc = nlp(text)

non_stop = [t for t in doc if not t.is_stop and not t.is_punct]
print(non_stop)    
    
nlp = spacy.load('en_core_web_sm')
nlp.vocab['down'].is_stop = False
nlp.vocab['Dear'].is_stop = True
nlp.vocab['Regards'].is_stop = True


nlp = spacy.load('en_core_web_sm')



text = "My best friend Ryan Peters likes fancy adventure games."
doc = nlp(text)

print(*[t.lemma_ for t in doc], sep='|')

text = "My best friend Ryan Peters likes fancy adventure games."
doc = nlp(text)

nouns = [t for t in doc if t.pos_ in ['NOUN', 'PROPN']]
print(nouns)

import textacy

tokens = textacy.extract.words(doc, 
            filter_stops = True,           # default True, no stopwords
            filter_punct = True,           # default True, no punctuation
            filter_nums = True,            # default False, no numbers
            include_pos = ['ADJ', 'NOUN'], # default None = include all
            exclude_pos = None,            # default None = exclude none
            min_freq = 1)                  # minimum frequency of words

print(*[t for t in tokens], sep='|')


def extract_lemmas(doc, **kwargs):
    return [t.lemma_ for t in textacy.extract.words(doc, **kwargs)]

lemmas = extract_lemmas(doc, include_pos=['ADJ', 'NOUN'])
print(*lemmas, sep='|')



text = "My best friend Ryan Peters likes fancy adventure games."
doc = nlp(text)

patterns = ["POS:ADJ POS:NOUN:+"]
spans = textacy.extract.matches.token_matches(doc, patterns=patterns)
print(*[s.lemma_ for s in spans], sep='|')


print(*doc.noun_chunks, sep='|')

def extract_noun_phrases(doc, preceding_pos=['NOUN'], sep='_'):
    patterns = []
    for pos in preceding_pos:
        patterns.append(f"POS:{pos} POS:NOUN:+")
    spans = textacy.extract.matches.token_matches(doc, patterns=patterns)
    return [sep.join([t.lemma_ for t in s]) for s in spans]

print(*extract_noun_phrases(doc, ['ADJ', 'NOUN']), sep='|')




text = "James O'Neill, chairman of World Cargo Inc, lives in San Francisco."
doc = nlp(text)

for ent in doc.ents:
    print(f"({ent.text}, {ent.label_})", end=" ")


from spacy import displacy

displacy.serve(doc, style='ent')
displacy.serve(doc, style='dep')

def extract_entities(doc, include_types=None, sep='_'):

    ents = textacy.extract.entities(doc, 
             include_types=include_types, 
             exclude_types=None, 
             drop_determiners=True, 
             min_freq=1)
    
    return [sep.join([t.lemma_ for t in e])+'/'+e.label_ for e in ents]

print(extract_entities(doc, ['PERSON', 'GPE']))




