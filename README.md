# Sentiment-Recognition-on-IMDB
Recognizing the sentiment and classifying each review to be either positive or negative given the 50k labeled IMDB movie reviews in text format. 

## Project Requirements:
Jupyter notebook with python 3

Scikit-learn machine learning library

NLTK - Natural Language tool kit library on python3

BS4 - Beautiful Soup library on python3

tqdm- tqdm library for python 3

Matplotlib


a cell is included in the jupyter notebook to ensure all prerequisite libraries are available (just uncomment it , it is under "imports and requirements")

all needed prior is installing Conda:

https://conda.io/docs/installation.html

## Project Overview
Sentiment Classification is a common task in the field of Natural Language Processing, it is
about extracting the sentiment of a person given a text they wrote as input.

Extracting sentiment from text can lead to build more intelligent systems acting according to
those sentiments like chatbots or smart assistants.

Although this task is considered a complex task even for humans due to absence of facial
expressions and voice tones in text but it is fairly much easier when given a paragraph of more
of text written for the purpose of delivering sentiment such as the case with movie reviews.

The dataset was constructed back in 2011 by Andrew L. Mass et al. in the paper “Learning Word
Vectors for Sentiment Analysis”, was a Kaggle challenge in 2016 and considered a Sentiment
Analysis benchmark. Another Dataset used for this task is "Rotten Tomatoes" Dataset.

## Problem Statement
Given the 50k labeled IMDB movie reviews in text format it is required to extract the sentiment
and classify each review (from the 25k test set) to be either positive or negative, then compare
the predicted results to the real labels and output the classification accuracy.

The problem is to be solved by extracting features out of blocks of text (reviews) and mapping
feature values and combinations to either positive or negative sentiment classes through the
usage of a couple of Supervised Machine Learning Algorithms (Logistic regression and Naive
Bayes), Also some text preprocessing will be done to facilitate and improve quality of extracted
features.

## Results
#### I- Logistic Regression

Before tuning

The obtained classification accuracy with the default hyperparameter values (C=1) was
86.868%.


After tuning

GridsearchCV was used to tune the C hyperparameter of the Logistic regression Classifier on
10% of training set, where C parameter is the inverse of the regularization strength.
Obtained accuracy: 87.928 % at C=50


#### II-Naive Bayes

Before tuning

The obtained classification accuracy with the default hyperparameter values (alpha=1) was
84.676 %.


After tuning

GridsearchCV was used again to tune the alpha hyperparameter of the Multinomial Naive Bayes
Classifier on 10% of training set, where alpha parameter is the smoothing factor.
Obtained accuracy: 84.6319% at alpha=0.5 .


K-Folds Cross Validation:

Besides hyperparameter tuning using Grid Search, K-Folds Cross Validation is now used to
validate the robustness of the model by making several train-validation splits within data
and train the model on them and test on different validation sets, to detect overfitting and
underfitting signs.


## Analysis,Methodology,Conclusion and Visuals are all included in the project report in pdf format and the code in IPython notebook and html formats.

## Credits
InProceedings{maas-EtAl:2011:ACL-HLT2011, author = {Maas, Andrew L. and Daly, Raymond E.
and Pham, Peter T. and Huang, Dan and Ng, Andrew Y. and Potts, Christopher},
title = {Learning Word Vectors for Sentiment Analysis},
booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational
Linguistics: Human Language Technologies},
month = {June},
year = {2011},
address = {Portland, Oregon, USA},
publisher = {Association for Computational Linguistics},
pages = {142--150},
url = {​http://www.aclweb.org/anthology/P11-1015}​ }
References
Potts, Christopher. 2011. On the negativity of negation. In Nan Li and David Lutz, eds.,
Proceedings of Semantics and Linguistic Theory 20, 636-659.
