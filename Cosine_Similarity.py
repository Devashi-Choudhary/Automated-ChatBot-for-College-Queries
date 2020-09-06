import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import base64
import string
import re
from collections import Counter
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
all_data=pd.read_csv("final_data_ir.csv",encoding='utf8')

# # Preprocessing
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
wordnet_lemmatizer = WordNetLemmatizer()
from nltk.stem.porter import PorterStemmer
porter_stemmer  = PorterStemmer()
import re
import inflect
def Pre_Processing(file):
    print(file)
    token_files=[]
    after_lower=[]
    after_lemmatizer=[]
    after_removing_stopwords=[]
    after_stemming=[]
    tokenizer = RegexpTokenizer(r'\w+')
    tokens=(tokenizer.tokenize(file))
    p = inflect.engine()
    token_files=[]
    for i in range(len(tokens)):
        if tokens[i].isnumeric() and len(tokens[i])<36:
            tem=p.number_to_words((tokens[i]))
            tokenizer = RegexpTokenizer(r'\w+')
            temp=(tokenizer.tokenize(tem))
            for x in temp:
                token_files.append(x)
        elif tokens[i].isnumeric() and len(tokens[i])>36:
            for j in range(len(tokens[i])):
                token_files.append(p.number_to_words((tokens[i][j])))
        else:
            token_files.append(tokens[i])
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
    
    for i in range(len(token_files)):
        after_lower.append(token_files[i].lower())
    for r in after_lower:
            if r not in stop_words:
                after_removing_stopwords.append(r)
    for i in range(len(after_removing_stopwords)):
        after_stemming.append(porter_stemmer.stem(after_removing_stopwords[i]))
    return after_stemming


# # TF-IDF Vector for a Term

x=[]
from sklearn.feature_extraction.text import TfidfVectorizer
k=0
while(k<len(all_data)):
    if(all_data['title'][k] != "nan"):
        preprossed_file=Pre_Processing(all_data['title'][k])
        text=""
        for c in preprossed_file:
            text=text+" "+c
        x.append(text) 
    k=k+1
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(x)
words=vectorizer.get_feature_names()


df = pd.DataFrame(X.toarray(),columns = vectorizer.get_feature_names())
List1=[]
for term in df.columns:
    c=0
    for i in range(len(df)):
        if(df[term][i]!=0):
            c+=df[term][i]
    List1.append(c)
dict1={}
for i in range(len(df.columns)):
    dict1[df.columns[i]]=List1[i]
print(len(dict1))

import operator
sorted_x = sorted(dict1.items(), key=operator.itemgetter(1),reverse=True)


# # Top 20 most Frequent words

df1=pd.DataFrame(sorted_x)
df2=df1.head(20)
xaxis=list(df2[0])
yaxis=list(df2[1])
fig = plt.figure(figsize=(20,6))
sns.barplot(x = xaxis, y=yaxis)
plt.show()

# # Cosine Similarity

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
def get_answers(all_data,query):
   
    queryTFIDF = TfidfVectorizer().fit(words)
    queryTFIDF = queryTFIDF.transform([query])
    maxval=-100
    for i in range(len(all_data)):
        cosine_similarities = cosine_similarity(queryTFIDF, X[i]).flatten()
        related_product_indices = cosine_similarities.argsort()[:-11:-1]
        if cosine_similarities>maxval:
            maxval=cosine_similarities
            answer=all_data['Answer1'][i]
    return (answer)


# # ChatBot
print("Hello user")
print("How may I help you")
print("For exiting from the chatbot,press 0")
x=2
inc=[0 for i in range(10)]
while x!=0:
    print('\n \n')
    print('Enter query')
    query=input()
    preprocessed_query= Pre_Processing(query)
    que=[]
    text=""
    for c in preprocessed_query:
        text=text+" "+c
    que=text
    res=get_answers(all_data,que)
    print(res)
    print("Enter 1 for relevant answer else return 0")
    s=int(input())
    inc[s-1]=inc[s-1]+1
    
    print("If you have any more queries then press 1 else press 0")
    x=int(input())
    if x==0:
        print("Thank you for using the chatbot.I hope you had a great time")

