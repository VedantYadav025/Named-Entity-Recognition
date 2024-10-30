import streamlit as st
import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import pandas as pd

model_path = "svm_classifier.joblib"

# vectorizer = DictVectorizer(sparse=True)
svm_clf = joblib.load(model_path)

# Feature extractor function: using the same one as for word2vec POS tagging
def word2features(sent, i):
    word_list = sent.split()
    word = word_list[i]

    features = {
        'word.lower()': word.lower(),
        'is_upper': word.isupper(),
        'is_title': word.istitle(),
        'is_digit': word.isdigit(),
        'prefix-1': word[:1],
        'suffix-1': word[-1:],
        'prefix-2': word[:2],
        'suffix-2': word[-2:]
    }

    return features

def detect_named_entities(sentence):
    word_list = sentence.split()
    token_label = []
    for i, word in enumerate(word_list):
        token = word
        label = svm_clf.predict(word2features(sentence, i))  
        token_label.append((token, label))
    return token_label


st.title("Named Entity Detection Demo")
st.write("Enter a sentence to see which words are recognized as named entities.")

# Text input field for sentence
sentence = st.text_input("Enter a sentence:")

if sentence:
    # Perform named entity detection
    entities = detect_named_entities(sentence)
    
    # Prepare data for the DataFrame
    data = []
    for token, label in entities:
        if label == "Entity":
            data.append((f"<span style='color:green'>{token}</span>", label))
        else:
            data.append((token, label))

    # Create a DataFrame
    df = pd.DataFrame(data, columns=["Token", "Label"])

    # Display results in a table
    st.write("**Named Entity Detection Results:**")
    st.markdown(df.to_html(escape=False), unsafe_allow_html=True)  # Use escape=False to render HTML
