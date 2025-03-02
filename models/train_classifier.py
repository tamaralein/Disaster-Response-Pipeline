import sys
import re
import pickle
import numpy as np
import pandas as pd
import nltk
nltk.download("punkt_tab")
nltk.download("wordnet")
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import multioutput
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def load_data(database_filepath):
    """
    Load the dataset from a CSV file.

    Returns:
        tuple: A tuple containing:
            - X (pd.DataFrame): The messages.
            - Y (pd.DataFrame): The category columns.
    """
    engine = create_engine("sqlite:///"+database_filepath)
    df = pd.read_sql_table("df",con=engine)
    X = df["message"]
    Y = df.iloc[:,2:]
    return X, Y

def tokenize(text):
    """
    Tokenize and lemmatize the input text.

    Args:
        text (str): The text to tokenize.

    Returns:
        list: A list of cleaned tokens.
    """

    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Builds a machine learning pipeline with feature extraction and a classifier. Uses Grid Search to further improve the model.

    Returns:
        fitted model
    """
    #create pipeline
    pipeline = Pipeline([
    ('features', FeatureUnion([

        ('text_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize, token_pattern=None)),
            ('tfidf', TfidfTransformer())
        ]))
    ])),

    ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    #include grid search
    parameters = { 
    'clf__estimator__n_estimators': [50, 100],
    'clf__estimator__learning_rate': [1, 2] 
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, cv=2, verbose=2)

    return cv


def evaluate_model(model, X_test, Y_test):
    """
    Evaluate the performance of a machine learning model on a test dataset.

    This function makes predictions using the provided model and generates 
    a classification report for each output category in the test dataset.

    Args:
        model: The trained machine learning model to evaluate.
        X_test (pd.DataFrame): The messages of the test dataset
        Y_test (pd.DataFrame): The category columns of the test dataset.

    Returns:
        None: This function prints the classification report for each output category but does not return any values.
    """
    #prediction
    Y_pred_test = model.predict(X_test)

    for i in range(Y_test.shape[1]):
        print("---------------------- " + Y_test.columns[i] + " ----------------------\n")
        print(classification_report(Y_test.iloc[:,i].values, Y_pred_test[:,i]))


def save_model(model, model_filepath):
    """
    Save a trained machine learning model as a pickle file to a specified file path.

    Args:
        model (object): The trained machine learning model to be saved. 
        model_filepath (string): The path the model should be saved to.

    Returns:
        None: This function does not return any value. It saves the model to the specified file path.
    """
    with open(model_filepath,'wb') as f:
        pickle.dump(model,f)
    
    return None


def main():
    """
    Main function to execute the disaster response model training pipeline.

    This function performs the following steps:
    1. Checks if the correct number of command-line arguments is provided.
    2. Loads the data from the specified SQLite database file.
    3. Splits the data into training and testing sets.
    4. Builds a machine learning model using the training data.
    5. Trains the model on the training data.
    6. Evaluates the model's performance on the test data.
    7. Saves the trained model to a specified file path.

    Command-Line Arguments:
        - database_filepath (str): The file path to the SQLite database containing the disaster messages.
        - model_filepath (str): The file path where the trained model should be saved (as a pickle file).

    Returns:
        None: This function does not return any value. It prints messages to indicate the progress of the pipeline.
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')


    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/Disaster_Response.db model.pkl')


if __name__ == '__main__':
    main()


