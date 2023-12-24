import nltk
from nltk.tokenize import word_tokenize
import streamlit as st
import joblib
import pickle
import streamlit.components.v1 as components
import requests
import numpy as np
import re
from PIL import Image
import plotly.graph_objects as go
from transformers import pipeline
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer

import pandas as pd

import pickle
# Load the fitted CountVectorizer and SVM model
with open('C:/Users/user/Documents/PSM/Save model/SVMNew.pkl', 'rb') as f:
    bow_vectorizer, SVM = pickle.load(f)

flat = ['ada', 'adakah', 'adakan', 'adalah', 'adanya', 'adapun', 'agak', 'agar', 'akan', 'aku', 'akulah', 'akupun', 'al', 'alangkah', 'amat', 'antara', 'antaramu', 'antaranya', 'apa', 'apa-apa', 'apabila', 'apakah', 'apapun', 'atas', 'atasmu', 'atasnya', 'atau', 'ataukah', 'ataupun', 'bagaimana', 'bagaimanakah', 'bagi', 'bagimu', 'baginya', 'bahawa', 'bahawasanya', 'bahkan', 'bahwa', 'banyak', 'banyaknya', 'barangsiapa', 'bawah', 'beberapa', 'begitu', 'begitupun', 'belaka', 'belum', 'belumkah', 'berada', 'berapa', 'berikan', 'beriman', 'berkenaan', 'berupa', 'beserta', 'biarpun', 'bila', 'bilakah', 'bilamana', 'bisa', 'boleh', 'bukan', 'bukankah', 'bukanlah', 'dahulu', 'dalam', 'dalamnya', 'dan', 'dapat', 'dapati', 'dapatkah', 'dapatlah', 'dari', 'daripada', 'daripadaku', 'daripadamu', 'daripadanya', 'demi', 'demikian', 'demikianlah', 'dengan', 'dengannya', 'di', 'dia', 'dialah', 'didapat', 'didapati', 'dimanakah', 'engkau', 'engkaukah', 'engkaulah', 'engkaupun', 'hai', 'hampir', 'hampir-hampir', 'hanya', 'hanyalah', 'hendak', 'hendaklah', 'hingga', 'ia', 'iaitu', 'ialah', 'ianya', 'inginkah', 'ini', 'inikah', 'inilah', 'itu', 'itukah', 'itulah', 'jadi', 'jangan', 'janganlah', 'jika', 'jikalau', 'jua', 'juapun', 'juga', 'kalau', 'kami', 'kamikah', 'kamipun', 'kamu', 'kamukah', 'kamupun', 'katakan', 'ke', 'kecuali', 'kelak', 'kembali', 'kemudian', 'kepada', 'kepadaku', 'kepadakulah', 'kepadamu', 'kepadanya', 'kepadanyalah', 'kerana', 'kerananya', 'kesan', 'ketika', 'kini', 'kita', 'ku', 'kurang', 'lagi', 'lain', 'lalu', 'lamanya', 'langsung', 'lebih', 'maha', 'mahu', 'mahukah', 'mahupun', 'maka', 'malah', 'mana', 'manakah', 'manapun', 'masih', 'masing', 'masing-masing', 'melainkan', 'memang', 'mempunyai', 'mendapat', 'mendapati', 'mendapatkan', 'mengadakan', 'mengapa', 'mengapakah', 'mengenai', 'menjadi', 'menyebabkan', 'menyebabkannya', 'mereka', 'merekalah', 'merekapun', 'meskipun', 'mu', 'nescaya', 'niscaya', 'nya', 'olah', 'oleh', 'orang', 'pada', 'padahal', 'padamu', 'padanya', 'paling', 'para', 'pasti', 'patut', 'patutkah', 'per', 'pergilah', 'perkara', 'perkaranya', 'perlu', 'pernah', 'pertama', 'pula', 'pun', 'sahaja', 'saja', 'saling', 'sama', 'sama-sama', 'samakah', 'sambil', 'sampai', 'sana', 'sangat', 'sangatlah', 'saya', 'se', 'seandainya', 'sebab', 'sebagai', 'sebagaimana', 'sebanyak', 'sebelum', 'sebelummu', 'sebelumnya', 'sebenarnya', 'secara', 'sedang', 'sedangkan', 'sedikit', 'sedikitpun', 'segala', 'sehingga', 'sejak', 'sekalian', 'sekalipun', 'sekarang', 'sekitar', 'selain', 'selalu', 'selama', 'selama-lamanya', 'seluruh', 'seluruhnya', 'sementara', 'semua', 'semuanya', 'semula', 'senantiasa', 'sendiri', 'sentiasa', 'seolah', 'seolah-olah', 'seorangpun', 'separuh', 'sepatutnya', 'seperti', 'seraya', 'sering', 'serta', 'seseorang', 'sesiapa', 'sesuatu', 'sesudah', 'sesudahnya', 'sesungguhnya', 'sesungguhnyakah', 'setelah', 'setiap', 'siapa', 'siapakah', 'sini', 'situ', 'situlah', 'suatu', 'sudah', 'sudahkah', 'sungguh', 'sungguhpun', 'supaya', 'tadinya', 'tahukah', 'tak', 'tanpa', 'tanya', 'tanyakanlah', 'tapi', 'telah', 'tentang', 'tentu', 'terdapat', 'terhadap', 'terhadapmu', 'termasuk', 'terpaksa', 'tertentu', 'tetapi', 'tiada', 'tiadakah', 'tiadalah', 'tiap', 'tiap-tiap', 'tidak', 'tidakkah', 'tidaklah', 'turut', 'untuk', 'untukmu', 'wahai', 'walau', 'walaupun', 'ya', 'yaini', 'yaitu', 'yakni', 'yang', 'la']




# Define the text classification function
def clean_text(tweet):
    # (Your preprocessing code remains unchanged)
       # Preprocess the text if needed
    if type(tweet) == float:
        return ""
    temp = tweet.lower()
    temp = re.sub("'", "", temp) # to avoid removing contractions in english
    temp = re.sub("@[A-Za-z0-9_]+","", temp)
    temp = re.sub("#[A-Za-z0-9_]+","", temp)
    temp = re.sub(r'http\S+', '', temp)
    temp = re.sub('[()!?]', ' ', temp)
    temp = re.sub('negative', '', temp)
    temp = re.sub('positive', '', temp)
    temp = re.sub('neutral', '', temp)
    temp = re.sub('\[.*?\]',' ', temp)
    temp = re.sub("[^a-z0-9]"," ", temp)
    temp = re.sub(r'[0-9]', '', temp)
    temp = temp.split()
    temp = [w for w in temp if not w in flat]
    temp = " ".join(word for word in temp)
    return temp


# Define the text classification function
def classify_text(text):
    # Preprocess the text if needed
    a = nltk.word_tokenize(text) 

    # Transform the input text using the loaded CountVectorizer
    X_vector = bow_vectorizer.transform(a)

    # Make predictions using the SVM model
    y_predict = SVM.predict(X_vector)

    if y_predict.any() == 0:
        p = "Tweet free from cyberbullying"
    else:
        p = "Tweet contains cyberbullying"

    return p  # Return the predicted class or label




#3D Plot for explain SVM
from sklearn.datasets import make_gaussian_quantiles# Construct dataset
X1, y1 = make_gaussian_quantiles(cov=1.,
                                 n_samples=1000, n_features=2,
                                 n_classes=2, random_state=1)
x1 = pd.DataFrame(X1,columns=['x','y'])
y1 = pd.Series(y1)

x1=x1.values

trace = go.Scatter(x=x1[:,0],y=x1[:,1],mode='markers',marker = dict(size = 12,color = y1,colorscale = 'Viridis'))
data=[trace]

layout = go.Layout()
print("2D PLot")
fig1 = go.Figure(data=data,layout=layout)



r = np.exp(-(x1 ** 2).sum(1)* 0.3)    ## exp(-gamma|x1-x2|**2) here gamma 0.3
x1 = np.insert(x1,2,r,axis=1)


model = LinearSVC(C=1.0, loss='hinge')
clf = model.fit(x1, y1)

Z = lambda X,Y: (-clf.intercept_[0]-clf.coef_[0][0]*X-clf.coef_[0][1]*Y) / clf.coef_[0][2]
# The equation of the separating plane is given by all x so that np.dot(svc.coef_[0], x) + b = 0.
# Solve for w3 (z)

trace1 = go.Mesh3d(x = x1[:,0], y = x1[:,1], z = Z(x1[:,0],x1[:,1])) ## for separating plane
trace2 = go.Scatter3d(x=x1[:,0], y=x1[:,1],z=x1[:,2],mode='markers',marker = dict(size = 3,color = y1,colorscale = 'Viridis')) ## for vector plots
data=[trace1,trace2]
fig2 = go.Figure(data=data,layout={})

#Bow section
d1 = "Saya seorang yang ceria"
d2 =  "Buku itu baru dijual"
d3 =  "Mari makan takoyaki"
d4 = "Buku itu tidak dijual disini"
set = [d1,d2,d3,d4]
bow = CountVectorizer()
features = bow.fit_transform(set)

p=features.todense()

df = pd.DataFrame(p, columns=bow.get_feature_names_out())
df.index = ['Text 1','Text 2','Text 3','Text 4']
# Streamlit app UI
tab1, tab2,tab3 = st.tabs(["SVM Info","BOW Info" ,"Sentiment Analysis using SVM"])
with tab1:
  st.title('Support Vector Machine(SVM)')
  st.header('What is SVM?')
  st.subheader('Support Vector Machine is a classification algorithm. SVM is a discriminative classifier that usually define by separating hyperplane')
  st.header('How SVM works?')
  i = Image.open('SVM.png')
  st.image(i,caption="SVM optimal hyperplane")
  st.subheader('SVM algorithm works by seperate two or more class using hyperplane. The algorithm will find the best hyperplane to isolate the data between the class. The goal was to produce the hyperplane that have large margin to get the correct prediction.')
  st.caption('Find more here: https://towardsdatascience.com/https-medium-com-pupalerushikesh-svm-f4b42800e989')
  st.header('SVM 3D Visualization')
  st.plotly_chart(fig1, use_container_width=True)
  st.plotly_chart(fig2, use_container_width=True)
  st.caption('Check out here: https://www.kaggle.com/code/sankha1998/3d-plot-on-svm-non-linear')
with tab2:
  st.title('Bag of Word(BOW)')
  st.subheader('Bag of Word is a traditional feature extraction, where it was used to convert the raw data into numerical representation by counting all the word in the corpus.')
  st.subheader('Example:')
  st.subheader('Text 1: Saya seorang yang ceria')
  st.subheader('Text 2: Buku itu baru dijual')
  st.subheader('Text 3: Mari makan takoyaki')
  st.subheader('Text 4: Buku itu tidak dijual disini')
  st.dataframe(df.style)
with tab3:

  st.title('Embedded Tweet')
  def theTweet(tweet):
    api = "https://publish.twitter.com/oembed?url={}".format(tweet)
    response = requests.get(api)
    res = response.json()["html"]
    return res


  res = theTweet('https://twitter.com/khairulaming/status/1714203047096963188')
    #st.write(res)
  components.html(res,height=1000)
  st.title('Sentiment Analysis using SVM')
  user_input = st.text_input("Enter your text here:")
  b = st.checkbox('Include Bag Of Word result')
  if st.button("Classify"):

      preprocessed_tweet = clean_text(user_input)
      prediction = classify_text(preprocessed_tweet)
      specific_model = pipeline(model="patrickxchong/bert-tiny-bahasa-cased-sentiment")
      y=specific_model(user_input)
      col1 , col2 = st.columns(2)
      col1.write(f"**Prediction**: {prediction}")
      col2.write(f"**Sentiment**: {y[0]['label']}")
      


      if y[0]['label'] == 'positive':
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = y[0]['score'],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Sentiment score"},
            gauge = {'axis': {'range': [0, 1]}, 'bar': {'color': "yellow"}}))

        st.plotly_chart(fig, use_container_width=True)

      else:
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = y[0]['score'],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Sentiment score"},
            gauge = {'axis': {'range': [0, 1]}, 'bar': {'color': "red"}}))
        st.plotly_chart(fig, use_container_width=True)


      if b:
        features1 = bow.fit_transform([user_input])
        p=features1.todense()
        df1 = pd.DataFrame(p, columns=bow.get_feature_names_out())
        df1.index = ['User Input']
        st.subheader("Bag of Word result")
        st.dataframe(df1.style)      

    