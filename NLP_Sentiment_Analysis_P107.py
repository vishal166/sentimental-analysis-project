import streamlit as st 
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import pickle
from tensorflow import keras
import sys
import matplotlib.pyplot as plt
import altair as alt
st.set_page_config(layout="wide")
import time
import random
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer,WordNetLemmatizer
from wordcloud import WordCloud
from textblob import Word
from datetime import datetime
from textblob import TextBlob
global col1, col2,col3
col1, col2,col3 = st.columns([3,1,1]) 
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()


def remove_tags(string):
    removelist = ""
    result = re.sub(r'^RT[\s]+', '', string)
    # remove hyperlinks
    result = re.sub(r'https?:\/\/.*[\r\n]*', '', result)
    result = re.sub(r'#', '', result)
    # removing hyphens
    result = re.sub('-', ' ', result)
    # remove linebreaks
    result = re.sub('<br\s?\/>|<br>', "", result)
    # remving numbers
    result = re.sub(r"(\b|\s+\-?|^\-?)(\d+|\d*\.\d+)\b",'',result)

    result = re.sub('','',result)          #remove HTML tags
    result = re.sub('https://.*','',result)   #remove URLs
    result = result.lower()
    return result
    
    
def lemmatize_text(text):
    st = ""
    for w in w_tokenizer.tokenize(text):
        st = st + lemmatizer.lemmatize(w) + " "
    return st




def sentiment_analysis(content):
    result = pd.DataFrame([[1,2,3]],columns=["TextBlob_Polarity","TextBlob_Subjectivity","TextBlob_Analysis"])
    def getSubjectivity(text):
        return TextBlob(text).sentiment.subjectivity

    # #Create a function to get the polarity
    def getPolarity(text):
        return TextBlob(text).sentiment.polarity

    # #Create two new columns ‘Subjectivity’ & ‘Polarity’
    result['TextBlob_Subjectivity'] =    getSubjectivity(content[0])
    result['TextBlob_Polarity'] = getPolarity(content[0])
    def getAnalysis(score):
        if score[0] < 0:
            return 'Negative'
        else:
            return 'Positive'
    result['TextBlob_Analysis'] = getAnalysis(result['TextBlob_Polarity'])

    all_text = ' '.join(word for word in content)
    wordcloud = WordCloud(width=200, height=200, colormap='Wistia',background_color='white', mode='RGBA').generate(all_text)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    fig, ax = plt.subplots(figsize=(4,3))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

    return result


def predict_sentiment(content):
    
   
    
    t = [content]
    
    result = sentiment_analysis(t)
    # convert to a number
    model = load_model()
    vec = load_vector()
    sentence = vec.transform(t)
    prediction = model.predict(sentence)
    if prediction[0] == 1 and result["TextBlob_Analysis"][0] == "Positive":
        s = '<h1 class="postive-msg">Positive Review \N{smiling face with sunglasses}</h1>'
    else:
        s = '<h1 class="negative-msg">Negative Review \N{unamused face}</h1> '
    st.write("<h4>Predicted Sentiment : ",s,"</h4>\n",unsafe_allow_html=True)
    

def load_model():
    model = open("finalized_model.pkl", 'rb') 
    return pickle.load(model)
def load_vector():
    vector = open("vector.pkl", 'rb') 
    return pickle.load(vector)

def run_sentiment_analysis(content):
    sentence = pd.DataFrame([[content]], columns=['short_review'])
    sentence["short_review"] = sentence["short_review"].apply(lambda cw : remove_tags(cw))

    # Remove Stop Words
    stop_words = set(stopwords.words('english'))
    sentence["short_review"] = sentence["short_review"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

    sentence['short_review'] = sentence["short_review"].apply(lemmatize_text)
    
    final_content = predict_sentiment(sentence["short_review"][0])    
    #st.write(final_content)


    


def main():
    with col1:
        st.title("Amazon Review Prediction")
        txt = st.text_area(label="Add Your Review",height=150)
        if st.button('Predict'):
            global dfFinal
            
            with st.spinner('Wait for it...loading'):
                time.sleep(1)        
            run_sentiment_analysis(txt)
                 
    
if __name__=='__main__':
    main()



