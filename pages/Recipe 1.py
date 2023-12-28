import nltk
from nltk.tokenize import word_tokenize
import streamlit as st
import joblib
import torch
import streamlit.components.v1 as components
import requests
from nltk.tree import Tree
from graphviz import Source
import graphviz
import pickle
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import re
from PIL import Image
import plotly.graph_objects as go
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoTokenizer,AutoModel,pipeline

import malaya.graph
import malaya
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import logging

logging.basicConfig(level=logging.INFO)





# Path to your pre-trained model's configuration file
model_path = "C:/Users/user/Documents/PSM/BERT_Ver2/Transformers-Text-Classification-BERT-Blog-main/model/"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('mesolitica/bert-base-standard-bahasa-cased')
#model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
model = AutoModel.from_pretrained('DeFamy/BERT-FineTune-CyberbullyingMalay').to(device)

flat = ['ada', 'adakah', 'adakan', 'adalah', 'adanya', 'adapun', 'agak', 'agar', 'akan', 'aku', 'akulah', 'akupun', 'al', 'alangkah',  'amat', 'antara', 'antaramu', 'antaranya', 'apa', 'apa-apa', 'apabila', 'apakah', 'apapun', 'atas', 'atasmu', 'atasnya', 'atau', 'ataukah', 'ataupun', 'bagaimana', 'bagaimanakah', 'bagi', 'bagimu', 'baginya', 'bahawa', 'bahawasanya', 'bahkan', 'bahwa', 'banyak', 'banyaknya', 'barangsiapa', 'bawah', 'beberapa', 'begitu', 'begitupun', 'belaka', 'belum', 'belumkah', 'berada', 'berapa', 'berikan', 'beriman', 'berkenaan', 'berupa', 'beserta', 'biarpun', 'bila', 'bilakah', 'bilamana', 'bisa', 'boleh', 'bukan', 'bukankah', 'bukanlah', 'dahulu', 'dalam', 'dalamnya', 'dan', 'dapat', 'dapati', 'dapatkah', 'dapatlah', 'dari', 'daripada', 'daripadaku', 'daripadamu', 'daripadanya', 'demi', 'demikian', 'demikianlah', 'dengan', 'dengannya', 'di', 'dia', 'dialah', 'didapat', 'didapati', 'dimanakah', 'engkau', 'engkaukah', 'engkaulah', 'engkaupun', 'hai', 'hampir', 'hampir-hampir', 'hanya', 'hanyalah', 'hendak', 'hendaklah', 'hingga', 'ia', 'iaitu', 'ialah', 'ianya', 'inginkah', 'ini', 'inikah', 'inilah', 'itu', 'itukah', 'itulah', 'jadi', 'jangan', 'janganlah', 'jika', 'jikalau', 'jua', 'juapun', 'juga', 'kalau', 'kami', 'kamikah', 'kamipun', 'kamu', 'kamukah', 'kamupun', 'katakan', 'ke', 'kecuali', 'kelak', 'kembali', 'kemudian', 'kepada', 'kepadaku', 'kepadakulah', 'kepadamu', 'kepadanya', 'kepadanyalah', 'kerana', 'kerananya', 'kesan', 'ketika', 'kini', 'kita', 'ku', 'kurang', 'lagi', 'lain', 'lalu', 'lamanya', 'langsung', 'lebih', 'maha', 'mahu', 'mahukah', 'mahupun', 'maka', 'malah', 'mana', 'manakah', 'manapun', 'masih', 'masing', 'masing-masing', 'melainkan', 'memang', 'mempunyai', 'mendapat', 'mendapati', 'mendapatkan', 'mengadakan', 'mengapa', 'mengapakah', 'mengenai', 'menjadi', 'menyebabkan', 'menyebabkannya', 'mereka', 'merekalah', 'merekapun', 'meskipun', 'mu', 'nescaya', 'niscaya', 'nya', 'olah', 'oleh', 'orang', 'pada', 'padahal', 'padamu', 'padanya', 'paling', 'para', 'pasti', 'patut', 'patutkah', 'per', 'pergilah', 'perkara', 'perkaranya', 'perlu', 'pernah', 'pertama', 'pula', 'pun', 'sahaja', 'saja', 'saling', 'sama', 'sama-sama', 'samakah', 'sambil', 'sampai', 'sana', 'sangat', 'sangatlah', 'saya', 'se', 'seandainya', 'sebab', 'sebagai', 'sebagaimana', 'sebanyak', 'sebelum', 'sebelummu', 'sebelumnya', 'sebenarnya', 'secara', 'sedang', 'sedangkan', 'sedikit', 'sedikitpun', 'segala', 'sehingga', 'sejak', 'sekalian', 'sekalipun', 'sekarang', 'sekitar', 'selain', 'selalu', 'selama', 'selama-lamanya', 'seluruh', 'seluruhnya', 'sementara', 'semua', 'semuanya', 'semula', 'senantiasa', 'sendiri', 'sentiasa', 'seolah', 'seolah-olah', 'seorangpun', 'separuh', 'sepatutnya', 'seperti', 'seraya', 'sering', 'serta', 'seseorang', 'sesiapa', 'sesuatu', 'sesudah', 'sesudahnya', 'sesungguhnya', 'sesungguhnyakah', 'setelah', 'setiap', 'siapa', 'siapakah', 'sini', 'situ', 'situlah', 'suatu', 'sudah', 'sudahkah', 'sungguh', 'sungguhpun', 'supaya', 'tadinya', 'tahukah', 'tak', 'tanpa', 'tanya', 'tanyakanlah', 'tapi', 'telah', 'tentang', 'tentu', 'terdapat', 'terhadap', 'terhadapmu', 'termasuk', 'terpaksa', 'tertentu', 'tetapi', 'tiada', 'tiadakah', 'tiadalah', 'tiap', 'tiap-tiap', 'tidak', 'tidakkah', 'tidaklah', 'turut', 'untuk', 'untukmu', 'wahai', 'walau', 'walaupun', 'ya', 'yaini', 'yaitu', 'yakni', 'yang', 'la']


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
    #temp = [w for w in temp if not w in flat]
    temp = " ".join(word for word in temp)
    return temp

# Define the text classification function
def BERT(input_text):
    
    # Example input text
    #input_text = "Saya sayang negara Malaysia"

    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt")

    inputs = {key: tensor.to(device) for key, tensor in inputs.items()}
    # Make predictions
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the predicted class
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    
    if predicted_class == 0:
        p = "Tweet free from cyberbullying"

    else:
        p = "Tweet contains cyberbullying"

    return p

#Function Constituency
def transformer(model: str = 'xlnet', quantized: bool = False, **kwargs):
    """
    Load Transformer Constituency Parsing model, transfer learning Transformer + self attentive parsing.

    Parameters
    ----------
    model : str, optional (default='bert')
        Model architecture supported. Allowed values:

        * ``'bert'`` - Google BERT BASE parameters.
        * ``'tiny-bert'`` - Google BERT TINY parameters.
        * ``'albert'`` - Google ALBERT BASE parameters.
        * ``'tiny-albert'`` - Google ALBERT TINY parameters.
        * ``'xlnet'`` - Google XLNET BASE parameters.

    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result : malaya.model.tf.Constituency class
    """

qmodel = malaya.constituency.transformer(model = 'xlnet',quantized = True)

#Function 2 Constituency
def parse_nltk_tree(self, string: str):

    """
    Parse a string into NLTK Tree, to make it useful, make sure you already installed tktinker.

    Parameters
    ----------
    string : str

    Returns
    -------
    result: nltk.Tree object
    """

#Vectorize
def vectorize(self, string: str):
    """
    vectorize a string.

    Parameters
    ----------
    string: List[str]

    Returns
    -------
    result: np.array
    """


# Streamlit app UI
tab1, tab2 = st.tabs(["BERT Info", "Sentiment Analysis using BERT"])

with tab1:
  st.title('Bidirectional Encoder Representation from Transformer(BERT)')
  st.header("What is BERT?")
  st.subheader("BERT is one of the part from Transformer which is build from stack of encoder. BERT is a language model which learn and understand the language itself.")
  i = Image.open('Transformers.png')
  st.image(i,caption="BERT Architecture")

  st.header("How BERT works?")
  st.subheader("BERT understand the language using two different unsupervised task simultaneously which is Masked Language Model(MLM) and Next Sentence Prediction(NSP). This phase is called as Pre-Trained phase, then it can be fine tuning to be applied in many task.")
  st.caption("Find more about BERT here:https://heidloff.net/article/foundation-models-transformers-bert-and-gpt/")
  st.header("BERT Application")
  c1 , c2,c3 = st.columns(3)
  c1.subheader("1. Text Summarization")
  c2.subheader("2. Question Answering")
  c3.subheader("3. Sentiment Analysis")



with tab2:
  st.title('This section is under maintenance🛠️')


    
          
          
