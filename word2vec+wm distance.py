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


# ## preprocessing
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


# ## applying model
model_para=Word2Vec(para, min_count=1)
model_para.init_sims(replace=True)
model_topic=Word2Vec(topics,min_count=1)
model_topic.init_sims(replace=True)

def get_answers(df_combined,query):
# query="Is it compulsory to have a project?And does it act as an disadvantage in the interview process if one does not?"
    q=preprocess_text(query)

    count=0
    min1=1000
    result=""
    for i in range(len(df_combined)):

        distance=model_para.wmdistance(q,para[i])

        if distance<min1:
            min1=distance
            result=df.iloc[i][2]
        count=count+1
        #print ('distance = %.3f' % distance)
    return result,distance

# ## initiating chat process
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
    print(r)
    if r<2:
        print(res)
    else:
        print("Sorry I don't have the answer. Can you please rephrase the query")
    print("If you have any more queries then press 1 else press 0")
    x=int(input())
    if x==0:
        print("Thank you for using the chatbot.I hope you had a great time")

