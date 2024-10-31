#!/usr/bin/env python

# Load the CoNLL-2003 dataset
from datasets import load_dataset

dataset = load_dataset("conll2003")


# Seperate the training testing and validation datasets

train_data = dataset['train']
test_data = dataset['test']
val_data = dataset['validation']
dataset['train']


# Print an example
train_data[0]


# In our problem statement we only need whether the entity is named or not, i.e. we'll be using binary labels.

# Replace all NER labels by 1 and let the rest remain 0
def modify_ner_tags(example):
    example['ner_tags'] = [1 if tag > 0 else 0 for tag in example['ner_tags']]
    return example

train_data = train_data.map(modify_ner_tags) # Modify data labels as per above function
val_data = val_data.map(modify_ner_tags)
test_data = test_data.map(modify_ner_tags)


# print example
train_data[0]


# Seeing how many words are Named entities which are not capitalized
ans = 0
list_of_such_words = []
for data in train_data:
  for idx, word in enumerate(data['tokens']):
    if ((word == word.lower()) and (data['ner_tags'][idx] == 1)):
      ans += 1 
      list_of_such_words.append(word)
    print(data)
print(ans)


print(list_of_such_words)


all_pos_tags = {'"': 0, "''": 1, '#': 2, '$': 3, '(': 4, ')': 5, ',': 6, '.': 7, ':': 8, '``': 9, 'CC': 10, 'CD': 11, 'DT': 12,
 'EX': 13, 'FW': 14, 'IN': 15, 'JJ': 16, 'JJR': 17, 'JJS': 18, 'LS': 19, 'MD': 20, 'NN': 21, 'NNP': 22, 'NNPS': 23,
 'NNS': 24, 'NN|SYM': 25, 'PDT': 26, 'POS': 27, 'PRP': 28, 'PRP$': 29, 'RB': 30, 'RBR': 31, 'RBS': 32, 'RP': 33,
 'SYM': 34, 'TO': 35, 'UH': 36, 'VB': 37, 'VBD': 38, 'VBG': 39, 'VBN': 40, 'VBP': 41, 'VBZ': 42, 'WDT': 43,
 'WP': 44, 'WP$': 45, 'WRB': 46}


from datasets import load_dataset

# Load the CoNLL-2003 dataset
dataset = load_dataset("conll2003")

# Define the target phrase
target_phrase = "Bank of America"

# Function to check if a phrase exists in a sentence
def contains_phrase(sentence, phrase):
    return " ".join(sentence) == phrase

# Search for the phrase in each partition (train, test, validation)
found = False
for split in dataset.keys():
    for example in dataset[split]:
        sentence_tokens = example['tokens']
        sentence = " ".join(sentence_tokens)
        if target_phrase in sentence:
            print(f'Found "{target_phrase}" in {split} set:', sentence)
            found = True
            break  # Remove this break if you want to search for multiple instances
    if found:
        break

if not found:
    print(f'"{target_phrase}" was not found in the CoNLL-2003 dataset.')

reverse_pos_tags = {value: key for key, value in all_pos_tags.items()}
print(reverse_pos_tags)

for key, value in reverse_pos_tags.items():
  print(f"{type(key)}")
  print(f"{type(value)}")
  break


import string 

def get_root_word(word):
  if word.endswith("'s"):
    return word[:-2]
  if word.endswith("'"):
    return word[:-1]
  return word

str1 = "Facebook's"
str2 = "Thomas'"
get_root_word(str1)
get_root_word(str2)


def word2features(sent, i):
    word = sent['tokens'][i]
    pos_tag = sent['pos_tags'][i]

    features = {
        'word.lower()': word.lower(),
        'is_upper': word.isupper(),
        'is_title': word.istitle(),
        'is_digit': word.isdigit(),
        'prefix-1': word[:1],
        'suffix-1': word[-1:],
        'prefix-2': word[:2],
        'suffix-2': word[-2:],
        'first_word_capital': word == word.lower(),
        'is_preposition': 1 if pos_tag == 15 else 0,
        'is_determiner': 1 if pos_tag == 12 else 0
    }

    # Add current word's POS tag features
    for tag, num in all_pos_tags.items():
        features[f'pos_tag_{tag}'] = 1 if pos_tag == num else 0

    # Add previous word's features if not at the start of the sentence
    if i > 0:
        prev_word = sent['tokens'][i - 1]
        prev_pos_tag = sent['pos_tags'][i - 1]
        
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
        
        for tag, num in all_pos_tags.items():
            features[f'prev_pos_tag_{tag}'] = 1 if prev_pos_tag == num else 0
    else:
        # Special tokens for start of sentence
        features['BOS'] = True  # Beginning of Sentence

    # Add next word's features if not at the end of the sentence
    if i < len(sent['tokens']) - 1:
        next_word = sent['tokens'][i + 1]
        next_pos_tag = sent['pos_tags'][i + 1]
        
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
        
        for tag, num in all_pos_tags.items():
            features[f'next_pos_tag_{tag}'] = 1 if next_pos_tag == num else 0
    else:
        # Special tokens for end of sentence
        features['EOS'] = True  # End of Sentence

    return features



# Create training, test and valdation dataset
def create_data(dataset):
    X = []
    y = []

    for data in dataset:
        X.extend([word2features(data, i) for i in range(len(data['tokens']))])
        y.extend(data['ner_tags'])

    return X, y 

for key, value in reverse_pos_tags.items():
  print(f"{type(key)}")
  print(f"{type(value)}")
  break

X_train, y_train = create_data(train_data)
X_val, y_val = create_data(val_data)
X_test, y_test = create_data(test_data)


X_train[0], y_train[0], type(X_train)


from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

vectorizer = DictVectorizer(sparse=True) # generate sparse vector from feature vector
svm_clf = make_pipeline(vectorizer, SVC(kernel='linear', random_state=42))


vectorized_data = vectorizer.fit_transform(X_train[0])
# Show the feature names and the resulting vector
print("Feature names:", vectorizer.get_feature_names_out())
print("Vectorized data:\n", vectorized_data)


svm_clf.fit(X_train, y_train)


# Saving the obtained weights of the svm classifier

import joblib


joblib.dump(svm_clf, 'svm_classifier_with_pos.joblib')


# Print training test and validation accuracy
from sklearn.metrics import accuracy_score

# Predictions on each dataset
y_train_pred = svm_clf.predict(X_train)
y_val_pred = svm_clf.predict(X_val)

# Calculate accuracy for each dataset
train_accuracy = accuracy_score(y_train, y_train_pred)
val_accuracy = accuracy_score(y_val, y_val_pred)

print(f"Training Accuracy: {train_accuracy}")
print(f"Validation Accuracy: {val_accuracy}")


# Load trained model for inference

# Load joblib model
svm_clf_loaded = joblib.load('svm_classifier_with_pos.joblib')


from sklearn.metrics import accuracy_score

y_test_pred = svm_clf_loaded.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Test Accuracy: {test_accuracy}")


# Plot confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_test_pred, labels = [0,1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Named', 'Named'])
disp.plot()
plt.title('Confusion matrix for NER')
plt.show()


import nltk
from nltk.tokenize import word_tokenize 

# Load the model
pos_tagger = joblib.load('crf_pos_tagger.joblib')

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
    


sample_sent = "Washington DC is the capital of United States of America"
print(predict_POS_Tags(sample_sent))
print(predict_POS_Tags(sample_sent)[1][1])
type(predict_POS_Tags(sample_sent)[1][1])


# Feature extractor function: using the same one as for word2vec POS tagging
def word2features2(sent, i):
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



sent = "Washington DC is the capital of United States of America"
word2features2(sent, 1)




sent = "Dr. Martin Luther King Jr. delivered his famous 'I Have a Dream' speech in Washington, D.C. during the 1963 March on Washington." 
for i, word in enumerate(sent.split()):
  print(f"Word: {word}, Entity: {svm_clf_loaded.predict(word2features2(sent, i))}")

