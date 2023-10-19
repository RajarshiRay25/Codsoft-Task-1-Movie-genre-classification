# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 15:07:47 2023

@author: Rajarshi Ray
@Internship : CodSoft - Task - 1
"""
# Data input and textual data preprocessing

## 1. Importing Libraries 

import nltk
# nltk.download()

import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns
## 2. Dataset input

df = pd.read_csv('D:/CodSoft Internship/1. TASK_1 Genre Classification Dataset/train_data.txt',sep=':::',names=['ID','TITLE','GENRE','DESCRIPTION'])
df.head()

## 3. Merge Title and Description columns

df['movie_details'] = df['TITLE'] + '' + df['DESCRIPTION']
df['movie_details']

## 4. View Column Names

df.columns

## 5. Textual data processing and cleaning with regular expression and NLTK

stemmer = PorterStemmer() # Declaring the stemmer object for text stemming
lemmatizer = WordNetLemmatizer() # Declaring the lemmatizer for text lemmatization

def nlp_processing(df):
    
    text_data = [] # in this list all the processed clean text data will be there
    
    for i in range(0,len(df)):   # Iterating over the entire dataset 
        
        # RE to remove all other punctuations,numbers etc except the letters  on the combined column
        
        text_processing = re.sub('[^a-zA-Z]',' ',df['movie_details'][i])
        
        # Convert to lowercase for simplicity
        
        text_processing = text_processing.lower()
        
        # Split into words
        
        text_processing = text_processing.split()
        
        # Stem those words in the content which are not stopwords
        
        text_processing = [stemmer.stem(text_vals) for text_vals in text_processing if not text_vals in stopwords.words('english')]
        
        # Assemble the combined words processed
        
        text_processing = ' '.join(text_processing)
        
        # Put into the list
        
        text_data.append(text_processing)
        
    return text_data


text_data_train_file = nlp_processing(df)

## 6. View the processed text file

text_data_train_file
    
# Vectorizing the processed text data to numerical values using TD-IDF technique
def vectorizer(text_data):
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    vectorizer = TfidfVectorizer(max_features=9000)  # declaring the vectorizer object and max features selected 9000
    
    X_col = vectorizer.fit_transform(text_data).toarray()  # convert the processed text to vectors
    
    return X_col

X_training_set = vectorizer(text_data_train_file)
X_training_set

# EDA on Genre data

df['GENRE'].value_counts()

plt.figure(figsize=(70,30))

sns.countplot(y=df['GENRE'],data=df)

# Encoding Genre to numerical with Label Encoder

y = df['GENRE']

y

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

y = encoder.fit_transform(y)

# Model building

## 3. Shape of X and y

print(f"Shape of X data : {X_training_set.shape}")
print(f"Shape of y data : {y.shape}")

## 4. Train Test Split - Training Data csv set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_training_set, y, test_size=0.25, random_state = 1)

## 5. Naive Bayes

from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

y_pred

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test,y_pred)

print(accuracy)

## 6. Logistic Regression

from sklearn.linear_model import LogisticRegression

model_log = LogisticRegression()

model_log.fit(X_train, y_train)

y_pred_log = model_log.predict(X_test)

y_pred_log


accuracy_log = accuracy_score(y_test,y_pred_log)

print(accuracy_log)

# Performing the same model testing on the test csv provided in dataset

df_test_csv = pd.read_csv('D:/CodSoft Internship/1. TASK_1 Genre Classification Dataset/test_data.txt',sep=':::',names=['ID','TITLE','DESCRIPTION'])

df_test_csv.head()

df_test_csv.columns


## 1. Merge Title and Description columns

df_test_csv['movie_details'] = df_test_csv['TITLE'] + '' + df_test_csv['DESCRIPTION']
df_test_csv['movie_details']

## 2. Text processing and cleaning

text_data_test_file = nlp_processing(df_test_csv)

text_data_test_file

## 3. Vectorize Test csv

X_testing_set = vectorizer(text_data_test_file)

X_testing_set

## 4. Load test data results

df_solution_test = pd.read_csv('D:/CodSoft Internship/1. TASK_1 Genre Classification Dataset/test_data_solution.txt',sep=':::',names=['ID','TITLE','GENRE','DESCRIPTION'])

df_solution_test.columns

y_sol = df_solution_test['GENRE']

y_sol = encoder.fit_transform(y_sol)

y_sol

y_sol.shape


## Model testing - Naive Bayes

y_sol_pred = model.predict(X_testing_set) 

accuracy_sol_NB = accuracy_score(y_sol,y_sol_pred)

accuracy_sol_NB 

## Model testing - Logistic Regression

y_sol_log = model_log.predict(X_testing_set)

accuracy_sol_log = accuracy_score(y_sol,y_sol_log)

accuracy_sol_log
