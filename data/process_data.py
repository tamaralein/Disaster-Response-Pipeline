import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load messages and categories datasets from specified file paths.

    Args:
        messages_filepath (str): The file path to the messages CSV file.
        categories_filepath (str): The file path to the categories CSV file.

    Returns:
        tuple: A tuple containing:
            - messages (pd.DataFrame): The DataFrame containing the messages data.
            - categories (pd.DataFrame): The DataFrame containing the categories data.
    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return messages, categories


def clean_data(messages, categories):
    """
    Clean and merge messages and categories datasets.

    This function performs the following operations:
    - Drops the "original" column from the messages DataFrame.
    - Merges the messages and categories DataFrames into a single DataFrame.
    - Splits the categories column into separate columns for each category.
    - Converts category values to binary (0 or 1).
    - Drops duplicate rows.

    Args:
        messages (pd.DataFrame): The DataFrame containing the messages data.
        categories (pd.DataFrame): The DataFrame containing the categories data.

    Returns:
        df (pd.DataFrame): A cleaned and merged DataFrame containing messages and binary category columns.
    """

    #drop "original" column from messages where more than 50% of the rows have a missing values
    messages = messages.drop(["original"],axis=1)

    #merge messages and categories into new dataset df
    df = pd.concat([categories, messages], axis = 1).drop(columns=["id"])
    
    #split categories in several columns and use the category name as column headers
    categories_splitted = categories['categories'].str.split(';', expand = True)
    row = categories_splitted.iloc[0]
    category_colnames = row.str[:-2]
    categories_splitted.columns = category_colnames

    #Convert category values to just numbers 0 or 1.
    for column in categories_splitted:
        categories_splitted[column] = categories_splitted[column].str[-1]
    categories_splitted[column] = pd.to_numeric(categories_splitted[column])

    #drop the old categories column and add categories_splitted as new columns
    df = df.drop(columns=["categories"])
    df = pd.concat([df,categories_splitted], axis = 1)

    #remove duplicates
    df.drop_duplicates(inplace=True)
    return df



def save_data(df, database_filename):
    """
    Save the cleaned dataset into a SQLite database.

    Args:
        df (pd.DataFrame): The DataFrame containing the cleaned data to be saved.
        database_filename (str): The file path where the SQLite database will be saved.

    Returns:
        None: This function does not return any value. It saves the DataFrame to the specified database file.
    """
    engine = create_engine("sqlite:///"+database_filename)
    df.to_sql('df', engine, index=False)
    pass  


def main():
    """
    Main function to load data, clean it, and save it to a database.

    This function checks for the correct number of command-line arguments,
    loads the datasets, cleans the data, and saves the cleaned data to a SQLite database.

    Returns:
        None: This function does not return any value.
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        messages, categories = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(messages, categories)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'Disaster_Response.db')


if __name__ == '__main__':
    main()

