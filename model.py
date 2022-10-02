"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import re
import unidecode
import pickle
import json
import datetime
import warnings
import time
import numpy      as np
import pandas     as pd
import wordninja

from   nltk.tokenize                   import word_tokenize
from   sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from   nltk.corpus                     import stopwords
from   nltk.stem                       import WordNetLemmatizer

def predict_lemma(words, lemmatizer):
    return [lemmatizer.lemmatize(word) for word in words]  

def remove_stopwords(tokens):    
    return [t for t in tokens if t not in stopwords.words('english')]

def preprocess_wordninja(sentence):      
        def split_words(x):
            x=wordninja.split(x)
            x= [word for word in x if len(word)>1]
            return x
        new_sentence=[ ' '.join(split_words(word)) for word in sentence.split() ]
        return ' '.join(new_sentence)


def cleanData(dataIn=None):
    pattern     = r'(https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}[-a-zA-Z0-9()@:%_+.~#?&/=]*)' 
    resultsData = pd.DataFrame()

    for index, row in dataIn.iterrows():
        row['links']        = re.findall(pattern, row['message'])
        row['moreWords']    = ''
        row['hashTags']     = ''
        row['cleanMessage'] = row['message']
        row['cleanMessage'] = re.sub(pattern , ' ', row['cleanMessage']) # remove URLs
        row['cleanMessage'] = re.sub('https?', ' ', row['cleanMessage']) # remove for half links


        #remove hashtags and handles
        if ('@' in row['cleanMessage']) or ('#' in row['cleanMessage']):

            for hashTag in re.findall('[#][\w]*' , row['cleanMessage']):
                if len(hashTag)>3:
                    row['hashTags'] =  row['hashTags'] + ' ' + hashTag

            row['cleanMessage'] = re.sub('[@#][\w]*', '', row['cleanMessage']) 

        row['cleanMessage'] = re.sub('[^a-zA-Z#]', ' ', row['cleanMessage']) # remove numetric

        #left out since most urls had most of the title in the actual tweet and it takes reeeeally long to scrape
        """
        #try scrape from the links we found earlier, just the titles
        for link in row['links']:
            try:
                page  = requests.get(link)
            except:
                continue
            else:
                if page.status_code != 404:                
                    rPage = BeautifulSoup(page.content, 'html.parser')
                    try:
                        if 'youtube' in rPage.title.text.lower():
                            rPage.title.text
                        row['moreWords'] = row['moreWords'] + '  ' + rPage.title.text
                        print(row['message'],'-->',row['moreWords'])
                    except:
                        continue
        """

        row['moreWords']    = unidecode.unidecode(row['moreWords'])
        row['cleanMessage'] = unidecode.unidecode(row['cleanMessage'])

        for hashTag2 in re.findall('[#][\w]*' , row['moreWords']): 
            if len(hashTag2)> 3:
                row['hashTags'] = row['hashTags'] + ',' + hashTag2    

        row['moreWords']    = re.sub('[^a-zA-Z#]', ' ', row['moreWords']) # remove numetric
        row['hashTags']     = row['hashTags'].lower()

        row['cleanMessage'] = row['cleanMessage'] + ' ' + row['moreWords']
        row['cleanMessage'] = row['cleanMessage'].lower()
        row['cleanMessage'] = preprocess_wordninja(row['cleanMessage'])
        row['cleanMessage'] = re.sub("\s[\s]+", " ",row['cleanMessage']).strip()  
        row                 = row.to_frame().transpose()    

        #print(index)

        #just append the temporary df to the output one
        if resultsData.empty:
            resultsData = row
        else:
             resultsData = pd.concat([resultsData,row])

    #just looking to get words with length > 1:
    resultsData['cleanMessage']= resultsData['cleanMessage'].apply(lambda x: ' '.join ([w for w in x.split() if len (w)>1 and w != 'rt']))

    return resultsData


def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.
    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
    #label
    testData         = feature_vector_df.copy()
    testDataCleaned  = cleanData(testData)
    testDataCleaned['tokens'] = testDataCleaned['cleanMessage'].apply(word_tokenize)

    lemmatizer               = WordNetLemmatizer()
    testDataCleaned['lemma'] = testDataCleaned['tokens'].apply(predict_lemma, args=(lemmatizer, ))
    testDataCleaned['lemmaNoStopWords']   = testDataCleaned['tokens'].apply(remove_stopwords)
    testDataCleaned['lemmaNoStopWords2']  = testDataCleaned['lemmaNoStopWords'] 


    resultsData = pd.DataFrame()
    for index,row in testDataCleaned.iterrows(): 
        row['lemmaNoStopWords2'] = ''
        for word in row['lemmaNoStopWords']:
            row['lemmaNoStopWords2'] = row['lemmaNoStopWords2'] + ' ' + word    
    
        row = row.to_frame().transpose()
        #just append the temporary df to the output one
        if resultsData.empty:
            resultsData = row
        else:
            resultsData = pd.concat([resultsData,row])
            
    testDataCleaned =  resultsData

    testX      = testDataCleaned['lemmaNoStopWords2'].values
    print(testDataCleaned['lemmaNoStopWords2'])
    vectorizer = TfidfVectorizer()
    testXfid   = vectorizer.fit_transform(testX)



    
    print('made it here')
    
    
    return testXfid

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    with open(path_to_model,'rb') as file:
        unpickled_model = pickle.load(file)
    return unpickled_model


""" You may use this section (above the make_prediction function) of the python script to implement 
    any auxiliary functions required to process your model's artifacts.
"""

def make_prediction(data, model):
    """Prepare request data for model prediction.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    print(prep_data)
    model = load_model('assets/trained-models/mlr_model.pkl')
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standardisation.
    return prediction.tolist()
