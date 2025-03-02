# Disaster Response Pipeline Project

### Table of Contents

1. [Installation](#installation)
2. [Project Overview](#overview)
3. [File Structure](#files)
4. [Instructions](#files)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

I worked with Visual Studio Code and Python Version 3.11.2. 

Following libraries should be installed: sys, re, pickle, numpy, pandas, nltk, sqlalchemy, sklearn, plotly, flask.


## Project Overview<a name="overview"></a>

This project analyzes disaster data from Appen (formerly Figure 8) to build a model that classifies disaster messages that were sent during disaster events. Furthermore, the project includes a web app where new messages can be entered and classified in several categories: Furthermore, the app includes visualizations of the training data.

## File structure<a name="files"></a>
### Folder "data":
- **disaster_categories.csv**: A CSV file that includes the categories associated with each disaster message. Each row represents a message and its corresponding categories, which are used for classification.
  
- **disaster_messages.csv**: A CSV file containing the actual disaster messages sent during various emergencies. Each row represents a unique message that needs to be classified into categories.

- **Disaster_Response.db**: A SQLite database file that stores the cleaned and processed data from the disaster messages and categories. This database is used for efficient querying and data retrieval during model training and evaluation.

- **process_data.py**: A Python script that implements the ETL process. It reads the disaster messages and categories datasets, cleans the data, and saves the cleaned data into the SQLite database.

### Folder "models":

- **model.pkl**: A pickle file that contains the trained machine learning model. This file can be loaded later to make predictions without needing to retrain the model.

- **train_classifier.py**: A Python script that trains the machine learning classifier on the cleaned data stored in the SQLite database. It includes code for loading the data, building the model pipeline, and saving the trained model to a pickle file.

  
### Folder "app":
- **run.py**: Flask file that runs app

#### Folder "templates":
    - master.html: main page of web app
    - go.html: classification result page of web app

## Instructions<a name="instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database:
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/Disaster_Response.db`
    - To run ML pipeline that trains classifier and saves:
        `python models/train_classifier.py data/Disaster_Response.db models/model.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://127.0.0.1:3000


## Licensing, Authors, Acknowledgements<a name="licensing"></a>

The datasets were privided by Appen. Otherwise, feel free to use the code here as you would like! 

