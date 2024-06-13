# importing streamlit
import streamlit as st
import pickle

#importing nltk (word_tokenization)
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords') #optional

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

# stopwords.words("english")

# importing string
import string
# string.punctuation


# pickle
import pickle

# last one -> ps package installed
# 1 lower case
# Tokenization
# Removing special characters
# Removing stop words and punctuation
# stemming.

# we're taking 1 functin to convert it into lower case

# 2 Tokenization

# we're going 1 function to convert it into words_tokenization

# 1 -> for the lowe case
def Tran_Lw(text):
    text = text.lower()

    # 2 -> for the word tokenization
    text = nltk.word_tokenize(text)  # break_down the words from the sentences in list.

    #    take an empty list.

    # 3 -> for the special characters and punctuation stuffs.
    y = []

    for i in text:
        if i.isalnum():  # isalnum -> including lower words and numerical except special characters.
            y.append(i)

    #   just copy once remove it.
    text = y[:]

    #   y is empty now.
    y.clear()

    # 4 -> for the stop words
    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)

    # again as same for stemming
    text = y[:]
    y.clear()

    # 5 -> for the stemming
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# sample_exe:
# Tran_Lw("Okay name ur price as long as its legal! Wen can I pick them up? Y u ave x ams xx")


# loaded tidf

tidf= pickle.load(open('vectorizer2.pkl', 'rb')) #read binay mode

# loaded mode
model = pickle.load(open('modelimprovement2.pkl', 'rb'))

# title
st.title("Email Spam Detection")

# input
input_sms= st.text_area("Enter the message")

# then click to butthon ro predict the output
if st.button('Predict'):
    # Now, there are 4 steps processes
    # 1. preprocessing
    transformed_sms = Tran_Lw(input_sms)

    # 2. vectorizer
    vector_input = tidf.transform([transformed_sms]) #passing in a list.

    # 3. predict
    result = model.predict(vector_input)[0]

    # 4. display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
