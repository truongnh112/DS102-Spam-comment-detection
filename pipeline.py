import re
import pandas as pd
import preprocessing
from preprocessing import *
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import CountVectorizer

c_vectorizer = CountVectorizer(ngram_range=(1, 1))
dingoc_vectorizer = CountVectorizer(ngram_range=(1,2))
X_train = pd.read_excel('data/X_train.xlsx')
c_vectorizer.fit_transform(X_train["comment"])
dingoc_vectorizer.fit_transform(X_train["comment"])

def encoder_count_dingoc(raw):
    pre_cmt = dingoc_vectorizer.transform([raw])
    return pre_cmt

def encoder_count(raw):
    pre_cmt = c_vectorizer.transform([raw])
    return pre_cmt
def encoder_list(raw_list):
    pre_list = c_vectorizer.transform(raw_list)
    return pre_list



def predict_raw(model, raw_cmt):
    # tiền xử lý dữ liệu sử dụng module model_rf_preprocess. 
    pre_cmt = encoder_count(raw=raw_cmt)
    # phán đoán nhãn
    pred = model.predict(pre_cmt)
    if pred[0] == 0:
        return "0. Không spam"
    elif pred[0] == 1:
        return "1. Spam"

def predict_raw_LR(model, raw_cmt):
    pre_cmt = encoder_count_dingoc(raw=raw_cmt)
    # phán đoán nhãn
    pred = model.predict(pre_cmt)
    if pred[0] == 0:
        return "0. No spam"
    elif pred[0] == 1:
        return "1. Spam"
