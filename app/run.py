import json
import plotly
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine

app = Flask(__name__)

def tokenize(text):
    """
    Tokenize and lemmatize the input text.

    Args:
        text (str): The text to tokenize.

    Returns:
        list: A list of cleaned tokens that have been lemmatized, lowercased, and stripped of whitespace.
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine("sqlite:///"+database_filepath)
df = pd.read_sql_table('df', engine)

# load model
model = joblib.load(model_filepath)

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """
    Render the index webpage that displays visualizations of the training data.

    Returns:
        str: Rendered HTML template for the index page with visualizations.
    """
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # Show distribution of different categories
    category = list(df.columns[2:])
    category_counts = (df.iloc[:,2:] != 0).sum().values

    # create visuals
    graphs = [
        #Distribution of message genres
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        
        #Distribution of message categories
        {
            'data': [
                Bar(
                    x=category,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    """
    Handle user input and display model results.

    This function retrieves the user input from the query, uses the trained 
    model to predict classification for the input text, and renders the 
    results on a new webpage.

    Returns:
        str: Rendered HTML template for the results page with classification results.
    """
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[2:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    """
    Run the Flask web application.

    This function starts the Flask web server and listens for incoming requests 
    on the specified host and port.

    Returns:
        None: This function does not return any value.
    """
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()