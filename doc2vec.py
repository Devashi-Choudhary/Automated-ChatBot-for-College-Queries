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
lemmatizer=WordNetLemmatizer()
import pandas as pd

df=pd.read_csv('Training.csv',encoding='unicode_escape')
df_combined = df

# ## functions to preprocess file
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

def preprocess_text(text):
    
    text=text.lower()
    token1=tokenizer.tokenize(text)
    token=[]
    for x in token1:
        if x not in stop_words:
            token.append(x)
            #stemmed = [stemmer.stem(tokens) ]
    lemmatiz=[lemmatizer.lemmatize(tokens) for tokens in token1]
    return lemmatiz

# ## applying model

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


# ## training model using doc2vec hyperparameters
arr=df_combined['Question'].tolist()


from gensim.models.doc2vec import TaggedDocument
tagged_data = [TaggedDocument(words=tokenizer.tokenize(str(_d).lower()),tags=[str(i)]) for i, _d in enumerate(arr)]

from gensim.models.doc2vec import Doc2Vec
max_epochs = 500
vec_size = 5
alpha = 0.025
model = Doc2Vec(size=vec_size,
                alpha=alpha, 
                min_alpha=0.00025,
                min_count=5,
                dm =1)
model.build_vocab(tagged_data)
for epoch in range(max_epochs):
    #print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

model.save("d2v.model")
print("Model Saved")


# ## calculating similarity
from gensim.models.doc2vec import Doc2Vec
def get_answers(df,x,query1):
    model= Doc2Vec.load(x)
    #to find the vector of a document which is not in training data
    query=Answer_Pre_Processing(query1)
    test_data=preprocess_text(query)
    
    v1 = model.infer_vector(test_data)
    #print("V1_infer", v1)

    # to find most similar doc using tags
    similar_doc = model.docvecs.most_similar([v1])
    return df.iloc[int(similar_doc[0][0])][1]


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
    res=get_answers(df,"d2v.model",q1)
    print(res)
    print("If you have any more queries then press 1 else press 0")
    x=int(input())
    if x==0:
        print("Thank you for using the chatbot.I hope you had a great time")

