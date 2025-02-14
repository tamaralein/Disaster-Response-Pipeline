# Disaster Response Pipeline Project

### Table of Contents

1. [Installation](#installation)
2. [Project Overview](#overview)
3. [Instructions](#files)
4. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

I worked with Visual Studio Code and Python Version 3.11.2. 

Following libraries should be installed: sys, re, pickle, numpy, pandas, nltk, sqlalchemy, sklearn, plotly, flask


## Project Overview<a name="overview"></a>

This project analyzes disaster data from Appen (formerly Figure 8) to build a model that classifies disaster messages that were sent during disaster events. Furthermore, the project includes a web app where new messages can be entered and classification results in several categories are visualized.

## Instructions<a name="instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/Disaster_Response.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/Disaster_Response.db models/classifier1.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


## Licensing, Authors, Acknowledgements<a name="licensing"></a>

The datasets were privided by Appen. Otherwise, feel free to use the code here as you would like! 
