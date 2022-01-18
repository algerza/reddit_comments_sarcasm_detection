
print("Starting app...")

###################################################################################################
###
### Install all the necessary packages 
###
###################################################################################################

print("Installing requirements...")

# get_ipython().system('pip install -r requirements_logit.txt')


###################################################################################################
###
### Import all the necessary packages and custom functions (from the functions.py file)
###
###################################################################################################

print("Importing packages...")

import time
import os
import numpy as np
import pandas as pd
import itertools
import pickle

import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold




###################################################################################################
###
### Load custom functions
###
###################################################################################################

print("Importing custom functions...")

def get_sarcasm_test_df_no_label():
    '''Get the data from the public Google Drive folder and load the test dataset'''

    url='https://drive.google.com/file/d/1XVOpt7i8729wIG477eTJeIQ_ehfa20p0/view?usp=sharing'
    url='https://drive.google.com/uc?id=' + url.split('/')[-2]
    test_df = pd.read_csv(url)
    
    return test_df


def get_sarcasm_test_df():
    '''Get the data from the public Google Drive folder and load the test dataset'''

    url='https://drive.google.com/file/d/1faIoQrb2ZQrlNaPP2keRUV8MZQQ599Sx/view?usp=sharing'
    url='https://drive.google.com/uc?id=' + url.split('/')[-2]
    test_df = pd.read_csv(url)
    
    return test_df
    

def get_sarcasm_train_df():
    '''Get the data from the public Google Drive folder and load the train dataset'''
    
    url='https://drive.google.com/file/d/1YnLnmoUDnj5ADr--KFLdJr7qsjkQN-GO/view?usp=sharing'
    path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
    df_1 = pd.read_csv(path, index_col=None)

    url='https://drive.google.com/file/d/1JDvEfYnebI-Cn71kRLF4BP3yM18wNnzV/view?usp=sharing'
    path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
    df_2 = pd.read_csv(path, index_col=None)

    url='https://drive.google.com/file/d/1SMxrY4VbNSI1t5R9qUZwgcPhWujJ6WyS/view?usp=sharing'
    path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
    df_3 = pd.read_csv(path, index_col=None)

    url='https://drive.google.com/file/d/1bN4CcWmpikc6mUukU1FoK2HS8yvnlYB3/view?usp=sharing'
    path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
    df_4 = pd.read_csv(path, index_col=None)

    train_df = df_1.append(df_2).append(df_3).append(df_4).reset_index()
    train_df.drop('index', axis=1, inplace=True)
    
    return train_df



def basic_cleaning(text):
    '''Make text lowercase, remove punctuation and quotations'''
    import re
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[^\w\s]', '', text)
    text = re.sub('\"', '', text)
    text = re.sub('\?', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('\n', '', text)
    return text


def remove_stopwords(text):
    '''Remove stopwords'''
    from nltk.corpus import stopwords
    text = text.split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    text = " ".join(text)
    return text


def replace_contractions(text):
    '''Remove stopwords'''
    # A list of contractions from http://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
    contractions = { 
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "must've": "must have",
    "mustn't": "must not",
    "needn't": "need not",
    "oughtn't": "ought not",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that'd": "that would",
    "that's": "that is",
    "there'd": "there had",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where'd": "where did",
    "where's": "where is",
    "who'll": "who will",
    "who's": "who is",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are"
    }

    text = text.split()
    new_text = []
    for word in text:
        if word in contractions:
            new_text.append(contractions[word])
        else:
            new_text.append(word)
    text = " ".join(new_text)
    return text





def clean_text(text, remove_stopwords = True):

    '''Remove unwanted characters, stopwords, and format the text to create fewer nulls word embeddings'''
    
    import re
    from nltk.corpus import stopwords

    # Convert words to lower case
    text = text.lower()
    
    # Replace contractions with their longer forms 
    if True:
        text = text.split()
        new_text = []
        for word in text:
            if word in contractions:
                new_text.append(contractions[word])
            else:
                new_text.append(word)
        text = " ".join(new_text)
    
    # Format words and remove unwanted characters
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\<a href', ' ', text)
    text = re.sub(r'&amp;', '', text) 
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'\'', ' ', text)
    
    # remove stop words
    if remove_stopwords:
        text = text.split()
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
        text = " ".join(text)
        
    return text





###################################################################################################
###
### Get the data from the Google Drive public folders
###
###################################################################################################

print("Loading data...")

# Load train and test dataframes
test_df = get_sarcasm_test_df()
train_df = get_sarcasm_train_df()



# 1. Modelling: Logistic Regression on 'comment' field without data cleaning


print("Preparing data...")

###################################################################################################
###
### Prepare the datasets for the model
###
###################################################################################################

# Initiate
start_time = time.time()

# Keep only the necessary columns
test_df_model = test_df[['id', 'label', 'comment']].dropna()
train_df_model = train_df[['id', 'label', 'comment']].dropna()

# Set training dataset
X_train = train_df_model['comment']
Y_train = train_df_model['label']

# Set test dataset
X_test = test_df_model['comment']
Y_test = test_df_model['label']

# Show how does the dataframe look like
train_df_model.head()




###################################################################################################
###
### Set up our text vectorisation and our model
###
###################################################################################################

print("Vectorising our data...")

# Transforming our text fields by quantifying the relevance of string representations. We will
#  set up the vectorizer considering unigrams (1,1) and bigrams (2,2) and ignoring terms that 
#  have a document frequency strictly lower than 2
tf_idf = TfidfVectorizer(ngram_range=(1, 2), min_df=2)


# Logistic Regression


# Initiate logistic regression for modelling. For the lbfgs solvers set verbose to any positive 
#   number for verbosity.
logit = LogisticRegression(n_jobs=4, solver='lbfgs', verbose=1, max_iter=1000000)


# Apply TF-IDF to our train and test text fields
X_train_transformed = tf_idf.fit_transform(X_train)
X_test_transformed = tf_idf.transform(X_test)



###################################################################################################
###
### Train our model and predict over the test dataset
###
###################################################################################################

print("Training our model and predicting labels...")


# Train the train dataset with Logistic Regression
logit.fit(X_train_transformed, Y_train)

# Predict labels (is the comment sarcastic or not) on the test dataset
predictions_values = logit.predict(X_test_transformed)




###################################################################################################
###
### Evaluate the results
###
################################################################################################### 
        

# Inidicate when it finishes
end_time = time.time()

# Check the accuracy of the model
accuracy_score(Y_test, predictions_values)
print("The logistic regression model predicts correctly %.2f percent of the Reddit comments"%(accuracy_score(Y_test, predictions_values)*100))



###################################################################################################
###
### Store the results
###
###################################################################################################

# Store the final results from the logistic regression model without cleaning (since it was our best performer)
logit_results = pd.DataFrame({'id': test_df_model.id, 'predicted': predictions_values})
print(logit_results.head())
# logit_results.to_csv("log_reg_results.csv")

# Store the results for comparison
score = round(accuracy_score(Y_test, predictions_values),2)
model_name = 'log_reg_comment_only_without_cleaning'

# Create a comparison table to append the results
comparison_table = pd.DataFrame([[model_name, score, round(end_time - start_time,0)]], columns = ['Model', 'Accuracy', 'Execution_Time_Seconds'])

print(comparison_table)

print("Task completed!")

