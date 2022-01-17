import pandas as pd


################################################################################################
###
### Get the data from Google Drive public folder
###
################################################################################################

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





# Function to plot the confusion matrix
def plot_confusion_matrix(actual, predicted, classes,
                          normalize=False,
                          title='Confusion matrix', figsize=(7,7),
                          path_to_save_fig=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    from sklearn.metrics import confusion_matrix
    from matplotlib import pyplot as plt
    import numpy as np

    cm = confusion_matrix(actual, predicted).T
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=figsize)
    cmap=plt.cm.Blues
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Predicted label')
    plt.xlabel('True label')
    
    if path_to_save_fig:
        plt.savefig(path_to_save_fig, dpi=300, bbox_inches='tight')







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




def score_classification(df):
    '''Classifies the comment score into groups'''
    
    if (df['score'] < -20):
        return '1. -20'
    elif (df['score'] >= -20) and (df['score'] < -10):
        return '2. -20 - -10'
    elif (df['score'] >= -10) and (df['score'] < 0):
        return '3. -10-0'
    elif (df['score'] >= 0) and (df['score'] < 5):
        return '4. 0-5'
    elif (df['score'] >= 5) and (df['score'] < 10):
        return '5. 5-10'
    elif (df['score'] >= 10) and (df['score'] < 20):
        return '6. 10-20'
    elif (df['score'] >= 20):
        return '7. 20+'




def number_of_words_groups(df):
    '''Classifies the number of words per comment into groups'''
    
    if (df['number_of_words'] >= 1) and (df['number_of_words'] < 4):
        return '1-4'
    elif (df['number_of_words'] >= 4) and (df['number_of_words'] < 8):
        return '4-8'
    elif (df['number_of_words'] >= 8) and (df['number_of_words'] < 16):
        return '8-16'
    elif (df['number_of_words'] >= 16):
        return '16+'








