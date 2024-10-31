import streamlit as st
import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import string

# POS tag dictionary in colln2003 tagset
all_pos_tags = {'"': 0, "''": 1, '#': 2, '$': 3, '(': 4, ')': 5, ',': 6, '.': 7, ':': 8, '``': 9, 'CC': 10, 'CD': 11, 'DT': 12,
 'EX': 13, 'FW': 14, 'IN': 15, 'JJ': 16, 'JJR': 17, 'JJS': 18, 'LS': 19, 'MD': 20, 'NN': 21, 'NNP': 22, 'NNPS': 23,
 'NNS': 24, 'NN|SYM': 25, 'PDT': 26, 'POS': 27, 'PRP': 28, 'PRP$': 29, 'RB': 30, 'RBR': 31, 'RBS': 32, 'RP': 33,
 'SYM': 34, 'TO': 35, 'UH': 36, 'VB': 37, 'VBD': 38, 'VBG': 39, 'VBN': 40, 'VBP': 41, 'VBZ': 42, 'WDT': 43,
 'WP': 44, 'WP$': 45, 'WRB': 46}

reverse_pos_tags = {value: key for key, value in all_pos_tags.items()}
# print(reverse_pos_tags)

# for key, value in reverse_pos_tags.items():
#   print(f"{type(key)}")
#   print(f"{type(value)}")
#   break

def get_root_word(word):
  if word.endswith("'s"):
    return word[:-2]
  if word.endswith("'"):
    return word[:-1]
  return word

model_path = "svm_classifier_with_pos.joblib"

# vectorizer = DictVectorizer(sparse=True)
svm_clf = joblib.load(model_path)

# Loading the pos tagger
pos_tagger = joblib.load("crf_pos_tagger.joblib")

# Fucntion to predict POS tags


# Convert tagged sentences into a list of (word, tag) pairs
def get_sentences_with_features(tagged_sentences):
    def word2features(sent, i):
        word = sent[i][0]
        features = {
            'bias': 1.0,
            'word.lower()': word.lower(),
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
        }
        if i > 0:
            word1 = sent[i - 1][0]
            features.update({
                '-1:word.lower()': word1.lower(),
                '-1:word.istitle()': word1.istitle(),
                '-1:word.isupper()': word1.isupper(),
            })
        else:
            features['BOS'] = True  # Beginning of Sentence

        if i < len(sent) - 1:
            word1 = sent[i + 1][0]
            features.update({
                '+1:word.lower()': word1.lower(),
                '+1:word.istitle()': word1.istitle(),
                '+1:word.isupper()': word1.isupper(),
            })
        else:
            features['EOS'] = True  # End of Sentence

        return features

    return [
        ([word2features(sent, i) for i in range(len(sent))], [tag for word, tag in sent])
        for sent in tagged_sentences
    ]

def dummy_tag_tokenized_sentence(untagged_tokenized_sentence):
    result = []
    for word in untagged_tokenized_sentence:
        result.append((word, 'DUMMY'))
    return result

def predict_POS_Tags(sent):
    tokens = word_tokenize(sent)
    dummy_tag_sent = dummy_tag_tokenized_sentence(tokens)
    dummy_tag_sent_list = []
    dummy_tag_sent_list.append(dummy_tag_sent)
    feature_tensor = get_sentences_with_features(dummy_tag_sent_list)
    x_train = [sent[0] for sent in feature_tensor]
    predicted_tags = pos_tagger.predict(x_train)
    # return predicted_tags[0]
    output = []
    for i in range(len(predicted_tags[0])):
        output.append((tokens[i], predicted_tags[0][i]))
    return output
 

# Feature extractor function: using the same one as for word2vec POS tagging

# Feature extractor function: using the same one as for word2vec POS tagging

# Feature extractor function: using the same one as for word2vec POS tagging
def word2features(sent, i):
    word_list = sent.split()
    word = word_list[i]
    pos_tag = predict_POS_Tags(sent)[i][1]
    features = {
        'word.lower()': word.lower(),
        'is_upper': word.isupper(),
        'is_digit': word.isdigit(),
        'is_title': word.istitle(),
        'is_digit': word.isdigit(),
        'prefix-1': word[:1],
        'suffix-1': word[-1:],
        'prefix-2': word[:2],
        'suffix-2': word[-2:],
        'first_word_capital': word == word.lower(),
        'is_preposition': 1 if int(pos_tag) == 15 else 0,
        'is_determiner': 1 if int(pos_tag) == 12 else 0
    }
    for tag_num, tag_name in reverse_pos_tags.items():
        features[f'pos_tag_{tag_name}'] = 1 if int(pos_tag) == tag_num else 0

    if i > 0:
        prev_word = word_list[i - 1]
        prev_pos_tag = predict_POS_Tags(sent)[i - 1][1]
        
        features.update({
            'root_word': get_root_word(word),
            'prev_word.lower()': prev_word.lower(),
            'prev_is_upper': prev_word.isupper(),
            'prev_is_title': prev_word.istitle(),
            'prev_is_digit': prev_word.isdigit(),
            'prev_prefix-1': prev_word[:1],
            'prev_suffix-1': prev_word[-1:],
            'prev_prefix-2': prev_word[:2],
            'prev_suffix-2': prev_word[-2:],
            'starts_with_capital': 1 if word[0].upper() == word[0] else 0
        })

        features['is_preceded_by_the'] = 1 if prev_word.lower() == 'the' else 0

        for tag_num, tag_name in reverse_pos_tags.items():
            features[f'prev_pos_tag_{tag_name}'] = 1 if int(prev_pos_tag) == tag_num else 0
        
    
    else:
        # Special tokens for start of sentence
        features['BOS'] = True
   
    if i < len(word_list) - 1:
        next_word = word_list[i + 1]
        next_pos_tag = predict_POS_Tags(sent)[i + 1][1]
        features.update({
            'next_word.lower()': next_word.lower(),
            'next_is_upper': next_word.isupper(),
            'next_is_title': next_word.istitle(),
            'next_is_digit': next_word.isdigit(),
            'next_prefix-1': next_word[:1],
            'next_suffix-1': next_word[-1:],
            'next_prefix-2': next_word[:2],
            'next_suffix-2': next_word[-2:]
        })   
        for tag_num, tag_name in reverse_pos_tags.items():
            features[f'next_pos_tag_{tag_name}'] = 1 if int(next_pos_tag) == tag_num else 0 
    
    else:
        # Special token for the end of sentence
        features['EOS'] = True # End of Sentence
 
        
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