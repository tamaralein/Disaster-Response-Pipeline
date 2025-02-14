# Disaster Response Pipeline Project

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [Instructions](#files)
4. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

I worked with Visual Studio Code and Python Version 3.11.2. There should be no necessary libraries to run the code. 

## Project Motivation<a name="motivation"></a>

For this project, I analyzed a dataset from Kaggle with Seattle Airbnb data. I looked at 3 main questions:

1. Which accomodation features influence the price the most?
2. Which neighbourhoods have the highest prices?
3. Which time of the year is the most expensive one?

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/Disaster_Response.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/Disaster_Response.db models/classifier1.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


## Licensing, Authors, Acknowledgements<a name="licensing"></a>

You can find the Licensing for the data and other descriptive information at the Kaggle link here https://creativecommons.org/publicdomain/zero/1.0/.  Otherwise, feel free to use the code here as you would like! 
