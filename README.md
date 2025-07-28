# codetech-task-2

**COMPANY**: CODTECH IT SOLUTIONS

**NAME**: MEGHANA ANIPIREDDY

**INTERN ID**: CT06DG2670

**DOMAIN**: MACHINE LEARNING

**DURATION**: 6 WEEEKS

**MENTOR**: NEELA SANTOSH

**📝 Task Overview**
This repository contains my submission for Task 2 of the internship, where I performed Sentiment Analysis on a dataset of customer reviews using Natural Language Processing (NLP) techniques. The primary goal was to classify the sentiments (positive or negative) from textual data using machine learning, particularly with TF-IDF vectorization and a Logistic Regression classifier.

Sentiment analysis is a powerful application of NLP that determines the emotional tone behind textual content. It is widely used in customer feedback systems, product review analysis, social media monitoring, and more.

**📊 Dataset**
The dataset used for this task consists of customer reviews labeled with their corresponding sentiment as either Positive or Negative.
Each entry contains a review in textual form and a target label indicating the sentiment. This binary classification task aims to teach a model how to distinguish between the two classes based on linguistic patterns.

**⚙ Task Instructions**
According to the given internship task:

Objective: Perform sentiment analysis using TF-IDF for text representation and Logistic Regression for classification.

Deliverable: A Jupyter Notebook demonstrating preprocessing, vectorization, model training, and sentiment evaluation.

Completion Certificate: Will be issued on the internship's end date.

**🔬 Model Pipeline**
The following steps were implemented in the Jupyter Notebook:

Text Preprocessing

Lowercased text

Removed punctuation and special characters

Removed stopwords

Applied tokenization

Feature Extraction using TF-IDF

Used TfidfVectorizer from sklearn to convert text data into numerical form.

Explored unigram and bigram combinations to enhance feature coverage.

_Model Building_: Logistic Regression

Trained a LogisticRegression model on the transformed text data.

Split the dataset into training and testing sets (typically 80/20 split).

_Model Evaluation_

Calculated accuracy, precision, recall, and F1-score.

Printed classification reports for performance comparison.

**🖼 Output Snapshots**
_🔍 Evaluation Results_
As shown in the image sentiment_analysis_output.jpg (included in the repository):

Accuracy: 0.0

Precision / Recall / F1-Score: All metrics are zero for both classes (Positive, Negative)

This result indicates that the model did not generalize correctly and failed to classify the test data. Possible reasons include:

Very small test size (e.g., only 3 samples)

Class imbalance in the dataset

Insufficient training data

Underfitting due to model simplicity or lack of sufficient features

**📈 Learnings & Insights**
Despite the poor evaluation metrics in this run, the task was highly educational. It provided key insights into:

How raw text is transformed into numerical vectors

The importance of preprocessing in NLP

The challenges of model performance in imbalanced or small datasets

The utility of logistic regression for binary text classification

In future iterations, I plan to:

Expand the dataset size

Apply class balancing techniques (like SMOTE or oversampling)

Try advanced models such as SVM or fine-tuned BERT for better accuracy

**🔧 Tech Stack**
Python

Jupyter Notebook

Scikit-learn

NLTK

Pandas & NumPy

Matplotlib (optional for visualization)

**📁 Files Included**
cd-2.ipynb: The main Jupyter notebook containing preprocessing, model implementation, and evaluation.

cd-2pic1.jpg,cd-2pic2.jpg: A screenshot showing the evaluation metrics output.

**OUTPUT:**
