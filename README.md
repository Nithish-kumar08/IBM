# Spam classification
This code performs a comprehensive text classification project, specifically for spam detection. Let's break down each section:

Importing Libraries:
The code begins by importing essential Python libraries that will be used for data analysis, data visualization, natural language processing (NLP), and machine learning.
Loading and Initial Data Analysis:

It reads a CSV file named 'spam.csv' using pandas and specifies the encoding as 'ISO-8859-1'.
The df DataFrame is created to hold the data.
df.head() displays the first few rows of the DataFrame.
df.info() provides information about the DataFrame's structure and data types.

Data Cleaning:
The code renames columns in the DataFrame to 'target' and 'sms' for clarity.
It drops unnecessary columns ('Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4').
Duplicates are checked using df.duplicated().sum() and removed, keeping the first occurrence.
Missing values are checked using df.isna().sum().

Label Encoding:
The 'target' column is transformed into binary values: 'ham' is encoded as 0, and 'spam' is encoded as 1.

Exploratory Data Analysis (EDA):
The code prepares a 1x2 subplot for EDA visualizations.
The first plot, a countplot, shows the distribution of 'target' values ('ham' and 'spam') using sns.countplot.
The second plot is a pie chart displaying the distribution of 'target' values as percentages.

Natural Language Processing (NLP):
The code downloads necessary resources for NLP using nltk.download('punkt').

Feature Engineering:

New features are created:
sentences_count: The number of sentences in each SMS.
words_count: The number of words in each SMS.
characters_count: The total number of characters in each SMS.
Visualization of Features:

Three histograms are plotted to visualize the distribution of these features, both for 'ham' and 'spam' messages.

Text Preprocessing:
Text preprocessing is performed, including:
Converting text to lowercase.
Tokenization of text using NLTK's nltk.word_tokenize.
Removing non-alphanumeric characters and stopwords.
Stemming using the Porter Stemmer.

Word Clouds:
Word clouds are generated to visualize the most common words in 'spam' and 'ham' messages using the WordCloud library.

Feature Extraction - Count Vectorization:
The text data is transformed into numerical vectors using the CountVectorizer.

Data Splitting:
The dataset is split into training and testing sets using train_test_split.

Testing Different Models:
Three different Naive Bayes models (GaussianNB, BernoulliNB, MultinomialNB) are tested using the test_models function, and their accuracy and precision scores are computed. Confusion matrices are also plotted.
Feature Extraction - TF-IDF Vectorization:

The text data is transformed into numerical vectors using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization.
Testing Different Models with TF-IDF Features:

The same three Naive Bayes models are tested again using TF-IDF features, and their scores are recorded.
Testing a Variety of Models:

A variety of classification models, including Logistic Regression, Support Vector Classifier (SVC), K-Nearest Neighbors (KNN), Decision Trees, Random Forest, Extra Trees, and others, are tested. Their accuracy and precision scores are reported.

Hyperparameter Tuning:
Hyperparameters for the best-performing models (Random Forest, MultinomialNB, KNN) are tuned using GridSearchCV to find optimal settings.
Retraining with Optimized Models:

The models with optimized hyperparameters are retrained, and their performance is evaluated.

Ensemble Model:
A VotingClassifier is created with the best models, and its accuracy and precision scores are reported.
In summary, this code demonstrates a complete pipeline for text classification, including data cleaning, EDA, feature engineering, text preprocessing, feature extraction, model testing, hyperparameter tuning, and the creation of an ensemble model for spam detection.
