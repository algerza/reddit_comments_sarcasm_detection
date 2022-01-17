# Reddit comments sarcasm detection 

Sarcasm has been part of our language for many years. It is the caustic use of irony, in which words are used to communicate the opposite of their surface meaning, in a humorous way or to mock someone or something. Understanding sarcasm is not always obvious, because it depends on your language skills and your knowledge of other people’s minds. For example, how would you classify the sentence “What a fantastic musician!”? Detecting sarcasm is much harder in text, as there are no additional cues. But what about a computer? Is it possible to train a machine learning model that can detect whether a sentence is sarcastic or not?

## Problem statement
The goal of this project is to predict whether a comment is sarcastic or not based on 1 million comments scrapped from Reddit - also called sub-reddits. This means, we are facing a binary classification problem that involves incorporating NLP techniques to feed to our ML/DL models.

## Content of this repo
1. Custom functions to load and clean the data located in the functions.py file
2. Exploratory data analysis' Jupyter notebook
3. Logistc Regression model's Jupyter notebook
4. RoBERTa model's Jupyter notebook
5. Final results for each model (csv's with their predicted labels) and summary table with a model comparison

## Next steps
- Compare more models' performance (i.e. Random Forest)
- Create more datasets for RoBERTa in order to improve accuracy: comment + parent comment, clean data, etc
- Re-organize and polish the code for a smooth deployment with Docker
- Refactor the code, loading an already trained model, and predict only the latest inputs to simulate a production environment where fast and accurate output is required
