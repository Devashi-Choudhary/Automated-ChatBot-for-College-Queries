# Note 

Hi, 

Thanks for approaching me for Dataset related information, as it belongs to IIIT Delhi and we have collected the data from various resources. The dataset is private and it's not good practice to share it outside.

Most of you have asked about it's format, so that format I can share.

It has basically in pair of question answer and we are running our models on that only. 

I hope, this information will be helpful. Still you have doubts, please mail me over devashi882@gmail.com

Small request, if this repo is helpful please hit the ⭐

Thanks,

Devashi


# Automated-ChatBot-for-College-Queries
We are building an automatic chatbot for answering college-related queries that are frequently asked by students. Students have a lot of queries and the queries are quite varied so, the aim of our chatbot is tp serve this purpose efficiently. 

# Dependencies 
As the dataset is text, most of the NLP Libraries are required like [NLTK](https://pypi.org/project/nltk/), [gensim](https://pypi.org/project/gensim/), [glove](https://pypi.org/project/glove/), [pyemd](https://pypi.org/project/pyemd/), [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html), [cosine_similarity](https://pypi.org/project/strsim/), [pandas](https://pypi.org/project/pandas/), [numpy](https://pypi.org/project/numpy/), [seaborn](https://pypi.org/project/seaborn/), string, counter, inflect, operator.

# Dataset

As we are aiming to build an automated chatbot for college queries, the dataset is not publicly available so, we collected dataset (w.r.t IIITD College). The following are the steps for Dataset Collection.


**Dataset Collection**

The dataset has been acquired by scrapping various sites and pages such as the admissions page of IIITD, the Reddit page dedicated to IIITD, and all the questions related to IIITD from Quora and assembling them. Also, paragraphs related to all domains like admissions, fees, faculty, etc. The dataset was built in this format for applying generative models that require the dataset in such a format Dataset is in the form of the text question and answers i.e. for every question; we have the most relevant answers scrapped from the popular social networking sites — Facebook, Quora, and Reddit. We also scrapped the data from the college website for some factual answers such as details of the faculty and the courses. In the training dataset, there are 3400 questions and answers scrapped using the tool — FacePager, OctoParser, and ScrapeStorm. Also, we have added tags for the questions as well as the link of the website from where we have obtained all the dataset.

**Data Cleaning**

Data obtained from the different social networking sites was highly anomalous and inconsistent to use. So, to make the data appropriate for training we processed the data in the following steps —
1. We removed the ”nan” values from the data-set by filling it with an appropriate answer.
2. We substituted the slang words with the original words else the similarity scores will decrease considerably.
3. We removed the non-alphanumeric characters such as emojis and ambiguous symbols to remove the inconsistency.

# How to execute the code:

1. You will first have to download the repository and then extract the contents into a folder.
2. Make sure you have the correct version of Python installed on your machine. This code runs on Python 3.6 above.
3. Now, you need to install the libraries required.
5. Now, you can run the python script on terminal using :
`python filename.py`

# Results

[![](http://img.youtube.com/vi/3UbeZPRI3C8/0.jpg)](http://www.youtube.com/watch?v=3UbeZPRI3C8 "Automated ChatBot for College Queries")

For more information about the implementation, please refer [Automated ChatBot for College Queries](https://medium.com/@Devashi_Choudhary/automated-chatbot-for-college-queries-19b03d72e3c8).

# Contributions

1. [Gitika Chhabra](https://github.com/ChhabraGitika)
2. [Ritwik Mitra](https://github.com/ritwik18010)

# Acknowledgement

The project was done as a part of College project. As our model is trained on the basis of similarity and semantic meaning of the words. If the question is asked out of our dataset contains, then the model doesn't work fine and also dataset is small in size. The feedbacks are welcomed and for dataset access please contant on  devashi882@gmail.com.
