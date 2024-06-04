# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 12:51:52 2021

@author: Shawn
"""


dfs1=dfs.copy()




def list_to_string(org_list, seperator=' '):
    return seperator.join(org_list)

dfs1['C_lemmas']=dfs1['lemmas'].apply(list_to_string)
dfs1['C_lemmas'][0]

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(decode_error='ignore', min_df=2) 

dt01 = cv.fit_transform(dfs1['C_lemmas'])
print(cv.get_feature_names_out())
fn=cv.get_feature_names()


import pandas as pd
dtmatrix=pd.DataFrame(dt01.toarray(), columns=cv.get_feature_names_out())


#利用cos算出文字的相似
from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity(dt01[30], dt01[63])

sm = pd.DataFrame(cosine_similarity(dt01, dt01))


dfs1.iloc[30,:].text
dfs1.iloc[63,:].text


from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer()


tfidf_dt = tfidf.fit_transform(dt01)
tfidfmatrix = pd.DataFrame(tfidf_dt.toarray(), columns=cv.get_feature_names())

sm1 =pd.DataFrame(cosine_similarity(tfidf_dt, tfidf_dt))


from matplotlib import pyplot as plt
from wordcloud import WordCloud ###
from collections import Counter ###

tfidfsum=tfidfmatrix.T.sum(axis=1)

wordcloud = WordCloud(font_path="SimHei.ttf", background_color="white")
wordcloud.generate_from_frequencies(tfidfsum)
#plt.figure(figsize=(20,10)) 
plt.imshow(wordcloud)

from sklearn.cluster import KMeans




from sklearn import preprocessing 
distortions = []
for i in range(1, 31):
    km = KMeans(
        n_clusters=i, init='random',
        n_init=10, max_iter=300,
        tol=1e-04, random_state=0
    )
    km.fit(preprocessing.normalize(tfidf_dt))
    distortions.append(km.inertia_)

# plot
from matplotlib import pyplot as plt
plt.plot(range(1, 31), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()


km = KMeans(
    n_clusters=5, init='random',
    n_init=10, max_iter=300, 
    tol=1e-04, random_state=0
)
y_km = km.fit_predict(preprocessing.normalize(tfidf_dt))

g0 = dfs1['text'][y_km==0]
g0.head()
g1 = dfs1['text'][y_km==1]
g1.head()
g2 = dfs1['text'][y_km==2]
g2.head()
g3 = dfs1['text'][y_km==3]
g3.head()
g4 = dfs1['text'][y_km==4]
g4.head()





