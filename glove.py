import num2words
import nltk
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
from nltk.stem.porter import *
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english')) 
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
stemmer=PorterStemmer()
from gensim.models import Word2Vec
from gensim.models import FastText
from glove import Corpus,Glove
from pyemd import wmd


import pandas as pd
df=pd.read_csv('Training.csv',encoding='unicode_escape')
df_combined = df


def preprocess_text(text):
    lemmatizer=WordNetLemmatizer()
    text=text.lower()
    token1=tokenizer.tokenize(text)
    token=[]
    for x in token1:
        if x not in stop_words:
            token.append(x)
            #stemmed = [stemmer.stem(tokens) ]
    lemmatiz=[lemmatizer.lemmatize(tokens) for tokens in token]
    return lemmatiz

def Answer_Pre_Processing(file):
    token_files=[]
    tokenizer = RegexpTokenizer(r'\w+')
    token_files=(tokenizer.tokenize(str(file)))
    for i in range(len(token_files)):
        if(token_files[i]=='u' or token_files[i]=='U'):
            token_files[i]='you'
        elif(token_files[i]=='d' ):
            token_files[i]='the'
        elif(token_files[i]=='n' or token_files[i]=='nd'):
            token_files[i]='and'
        elif(token_files[i]=='hv' ):
            token_files[i]='have'
        elif(token_files[i]=='bcoz' or token_files[i]=='becoz' or token_files=='bcz'):
            token_files[i]='because'
        elif(token_files[i]=='ur' or token_files[i]=='Ur'):
            token_files[i]='your'
        elif(token_files[i]=='thru'):
            token_files[i]='through'
    str1=""
    for word in token_files:
        str1=str1+word+" "
    return str1

topics=[[] for i in range(len(df))]
para=[[] for i in range(len(df))]
topics1=[[] for i in range(len(df))]
para1=[[] for i in range(len(df))]
for i in range(len(df_combined)):
    text=df_combined.iloc[i][0]
    text=str(text)
    topics[i]=preprocess_text(text)
    text=df_combined.iloc[i][1]
    text=str(text)
    para[i]=preprocess_text(text)

corpus = Corpus()
corpus.fit(para, window=10)
glove = Glove(no_components=5, learning_rate=0.05)
glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
glove.add_dictionary(corpus.dictionary)
# glove.save('glove.model')

corpus1 = Corpus()
corpus1.fit(topics, window=10)
glove1 = Glove(no_components=5, learning_rate=0.05)
glove1.fit(corpus1.matrix, epochs=30, no_threads=4, verbose=True)
glove1.add_dictionary(corpus1.dictionary)


import numpy as np
def get_answers(df_combined,query1):
    query=Answer_Pre_Processing(query1)
    q=preprocess_text(query)

    count=0
    max1=-1000
    result=""
    for i in range(len(df_combined)):
        sim=0
#         distance=glove.wmdistance(q,para[i])
        for j in range(len(para[i])):
            for k in range(len(q)):
                if para[i][k] in glove.dictionary:
                    a=glove.word_vectors[glove.dictionary[para[i][j]]]
                else:
                    a=np.zeros(5)
                if q[j] in glove.dictionary:
                    b=glove.word_vectors[glove.dictionary[q[k]]]
                else:
                    b=np.zeros(5)
                res1=np.dot(a,b)
                
                sim=sim+res1
        #print(sim)
        if sim>max1:
            max1=sim
            result=df_combined.iloc[i][2]
            if result=='nan':
                print(df_combined.iloc[i][1])
#             print(result)
        count=count+1
        #print ('distance = %.3f' % distance)
    print(count)
    return result,max1

print("Hello user")
print("How may I help you")
print("For exiting from the chatbot,press 0")
x=2
while x!=0:
    print('\n \n')
    print('Enter query')
    query=input()
    q=preprocess_text(query)
    
    q1=""
    for d in q:
        q1=q1+d+" "
    res,r=get_answers(df,q1)
    #print(r)
    #print(res)
    if r>0:
        print(res)
    else:
        print("Sorry I don't have the answer. Can you please rephrase the query")
    print("If you have any more queries then press 1 else press 0")
    x=int(input())
    if x==0:
        print("Thank you for using the chatbot.I hope you had a great time")

