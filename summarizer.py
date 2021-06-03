from sklearn.decomposition import LatentDirichletAllocation as LDiA
from nltk.tokenize import casual_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from nltk.tokenize import TreebankWordTokenizer
import nltk
from nltk.tokenize import sent_tokenize
counter = CountVectorizer()

with open('corpus2.txt', 'r+', encoding="utf-8") as sth:
    text = sth.read()


#corpus = 'This is the first document. This document is the second document. And this is the third one. Is this the first document?'
corpus = text.replace("—", '').replace(',', '').replace("'", '').replace('"', '')
corpus = sent_tokenize(corpus)

bow_docs = pd.DataFrame(counter.fit_transform(corpus).toarray())
print(bow_docs)
ldia = LDiA(n_components=5, learning_method='batch')
ldia = ldia.fit(bow_docs)
ldia = pd.DataFrame(ldia.transform(bow_docs))
print(ldia)
sentences = ldia.idxmax()
print(sentences)
print(sentences[0])
for i in range(5):
    idx = sentences[i]
    print(corpus[idx])
    with open ('summarization.txt', 'a', encoding='utf-8') as sth:
        if (i==0):
            sth.write('\n')
        sth.write(corpus[idx] + '\n')
        sth.close
