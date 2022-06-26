import streamlit as st
import pandas as pd
import numpy as np
import pickle
import preprocessing
from preprocessing import *
from pipeline import *
from build_data import *
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt


#load model da train

#gridSVM = pickle.load(open('Model/gridsvm.pkl', 'rb'))

SVMclf = pickle.load(open('Model/Best_model_SVM_(1,1)_Count_grid(cv=10).pkl', 'rb'))
KNNclf = pickle.load(open('Model/knn_model.pkl', 'rb'))
LRclf  = pickle.load(open('Model/model_logistic.pkl', 'rb'))
NBclf = pickle.load(open('Model/model_NB_11_Count.pkl', 'rb'))




def analyze(result):
 
    num_of_spam=0
    num_of_no_spam=0
 
    for pred in result:
      
        if pred == 0:
            num_of_no_spam = num_of_no_spam +1
        elif pred == 1:
            num_of_spam = num_of_spam +1
    
    st.write('Spam', num_of_spam)
    st.write('No Spam', num_of_no_spam)
    
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = 'Spam', 'No spam'
    sizes = [num_of_spam, num_of_no_spam ]
    explode = (0, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    st.pyplot(fig1)
    # objects = ("rất tiêu cực", "hơi tiêu cực", "trung tính", "hơi tích cực", "rất tích cực")
    # y_pos = np.arange(len(objects))
    # performance = [num_of_vneg, num_of_sneg, num_of_neu, num_of_spos, num_of_vpos]
    # df = pd.DataFrame(performance,objects )
    # st.bar_chart(df)




def precdict_by_link(model,list_cmt):
    X_test = encoder_list(list_cmt)
    y_pred = model.predict(X_test)
    return y_pred




def main():
    
    
    st.set_page_config(
    page_title="SPAM DETECTION",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
    )
    
    col1, col2 = st.columns([6, 4])

    with col1:
        st.header("NHẬN DIỆN SPAM TRONG BÌNH LUẬN SẢN PHẨM")
        url = st.text_input("Link sản phẩm: ")

        if url:
            r = re.search(r"i\.(\d+)\.(\d+)", url)
            shop_id, item_id = r[1], r[2]
            st.write("Shop ID: ", shop_id)
            st.write("Product ID: ", item_id)
            try:
                crawl_data(url)
                build_dataset()
            except:
                st.write("")
            data = pd.read_csv('data/dataset.csv')
            with open('data/dataset.csv', 'rb') as csv:
                file_container = st.expander("Check your crawl .csv")
                shows = pd.read_csv('data/dataset.csv')
                file_container.write(shows)
                st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='comment.csv',
                mime='text/csv',
                )

            if st.button('Analyze'):
                    with st.spinner("Analyzing..."):

                        result = precdict_by_link(SVMclf,data['comment'])
                        analyze(result=result)

                        st.success(f'Analysis finished')               
           
       

                
    
       
        

    with col2:
        st.markdown("**DỰ ĐOÁN MỘT BÌNH LUẬN**")
        option = st.selectbox('Select a review form:',
        ['None', 'SVM Kernel RBF', 'K - Nearest Neighbor', 'Naive Bayes', 'Logistic Regression'])

        st.write('Options: ',  option)

        with st.form(key="text"):
            raw_review = st.text_area("Review")
            submit = st.form_submit_button(label="Submit")

        st.write("Submit: ", submit)
        if submit:
            with st.spinner("Predicting..."):

                if option == 'SVM Kernel RBF':
                    model = SVMclf
                    st.write(predict_raw(model, raw_review))
                if option == 'K - Nearest Neighbor':
                    model = KNNclf
                    st.write(predict_raw(model, raw_review))                   
                if option == 'Naive Bayes':
                    model = NBclf
                    st.write(predict_raw(model, raw_review)) 
                if option == 'Logistic Regression':
                    model = LRclf
                    st.write(predict_raw_LR(model, raw_review))

           
    
    

if __name__ == "__main__":
    main()