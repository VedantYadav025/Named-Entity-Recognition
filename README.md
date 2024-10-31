# Named Entity Recognition

We use Support Vector Machines with linear kernels to predicted whether a token (word) in a sentence is a named entity or not.

We are doing binary classification here.

The features used for SVM are:
* Basic semantic features (whether the word is capital or not, suffix, prefix. etc.)

* We use POS (part of speech) tag of the current word, previous word and the next word.

* We take special cases, such as if the word is a preposition or a determiner, we add a feature   which helps us show whether the word is surrounder by captial letter words or not.


# Steps to Use

* Clone the repository

* Install the following libraries:
  numpy
  matplotlib
  nltk
  datasets (huggingface)
  sklearn
  string
 
* Run the following command: `streamlit run ner_demo.py`
