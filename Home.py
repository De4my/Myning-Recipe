from PIL import Image
import streamlit as st
import pandas as pd
flat = ['ada', 'adakah', 'adakan', 'adalah', 'adanya', 'adapun', 'agak', 'agar', 'akan', 'aku', 'akulah', 'akupun', 'al', 'alangkah',  'amat', 'antara', 'antaramu', 'antaranya', 'apa', 'apa-apa', 'apabila', 'apakah', 'apapun', 'atas', 'atasmu', 'atasnya', 'atau', 'ataukah', 'ataupun', 'bagaimana', 'bagaimanakah', 'bagi', 'bagimu', 'baginya', 'bahawa', 'bahawasanya', 'bahkan', 'bahwa', 'banyak', 'banyaknya', 'barangsiapa', 'bawah', 'beberapa', 'begitu', 'begitupun', 'belaka', 'belum', 'belumkah', 'berada', 'berapa', 'berikan', 'beriman', 'berkenaan', 'berupa', 'beserta', 'biarpun', 'bila', 'bilakah', 'bilamana', 'bisa', 'boleh', 'bukan', 'bukankah', 'bukanlah', 'dahulu', 'dalam', 'dalamnya', 'dan', 'dapat', 'dapati', 'dapatkah', 'dapatlah', 'dari', 'daripada', 'daripadaku', 'daripadamu', 'daripadanya', 'demi', 'demikian', 'demikianlah', 'dengan', 'dengannya', 'di', 'dia', 'dialah', 'didapat', 'didapati', 'dimanakah', 'engkau', 'engkaukah', 'engkaulah', 'engkaupun', 'hai', 'hampir', 'hampir-hampir', 'hanya', 'hanyalah', 'hendak', 'hendaklah', 'hingga', 'ia', 'iaitu', 'ialah', 'ianya', 'inginkah', 'ini', 'inikah', 'inilah', 'itu', 'itukah', 'itulah', 'jadi', 'jangan', 'janganlah', 'jika', 'jikalau', 'jua', 'juapun', 'juga', 'kalau', 'kami', 'kamikah', 'kamipun', 'kamu', 'kamukah', 'kamupun', 'katakan', 'ke', 'kecuali', 'kelak', 'kembali', 'kemudian', 'kepada', 'kepadaku', 'kepadakulah', 'kepadamu', 'kepadanya', 'kepadanyalah', 'kerana', 'kerananya', 'kesan', 'ketika', 'kini', 'kita', 'ku', 'kurang', 'lagi', 'lain', 'lalu', 'lamanya', 'langsung', 'lebih', 'maha', 'mahu', 'mahukah', 'mahupun', 'maka', 'malah', 'mana', 'manakah', 'manapun', 'masih', 'masing', 'masing-masing', 'melainkan', 'memang', 'mempunyai', 'mendapat', 'mendapati', 'mendapatkan', 'mengadakan', 'mengapa', 'mengapakah', 'mengenai', 'menjadi', 'menyebabkan', 'menyebabkannya', 'mereka', 'merekalah', 'merekapun', 'meskipun', 'mu', 'nescaya', 'niscaya', 'nya', 'olah', 'oleh', 'orang', 'pada', 'padahal', 'padamu', 'padanya', 'paling', 'para', 'pasti', 'patut', 'patutkah', 'per', 'pergilah', 'perkara', 'perkaranya', 'perlu', 'pernah', 'pertama', 'pula', 'pun', 'sahaja', 'saja', 'saling', 'sama', 'sama-sama', 'samakah', 'sambil', 'sampai', 'sana', 'sangat', 'sangatlah', 'saya', 'se', 'seandainya', 'sebab', 'sebagai', 'sebagaimana', 'sebanyak', 'sebelum', 'sebelummu', 'sebelumnya', 'sebenarnya', 'secara', 'sedang', 'sedangkan', 'sedikit', 'sedikitpun', 'segala', 'sehingga', 'sejak', 'sekalian', 'sekalipun', 'sekarang', 'sekitar', 'selain', 'selalu', 'selama', 'selama-lamanya', 'seluruh', 'seluruhnya', 'sementara', 'semua', 'semuanya', 'semula', 'senantiasa', 'sendiri', 'sentiasa', 'seolah', 'seolah-olah', 'seorangpun', 'separuh', 'sepatutnya', 'seperti', 'seraya', 'sering', 'serta', 'seseorang', 'sesiapa', 'sesuatu', 'sesudah', 'sesudahnya', 'sesungguhnya', 'sesungguhnyakah', 'setelah', 'setiap', 'siapa', 'siapakah', 'sini', 'situ', 'situlah', 'suatu', 'sudah', 'sudahkah', 'sungguh', 'sungguhpun', 'supaya', 'tadinya', 'tahukah', 'tak', 'tanpa', 'tanya', 'tanyakanlah', 'tapi', 'telah', 'tentang', 'tentu', 'terdapat', 'terhadap', 'terhadapmu', 'termasuk', 'terpaksa', 'tertentu', 'tetapi', 'tiada', 'tiadakah', 'tiadalah', 'tiap', 'tiap-tiap', 'tidak', 'tidakkah', 'tidaklah', 'turut', 'untuk', 'untukmu', 'wahai', 'walau', 'walaupun', 'ya', 'yaini', 'yaitu', 'yakni', 'yang', 'la']
t = Image.open('New.png')
st.image(t, width=None)




st.title("Home Page")
st.subheader("Myning Recipe website are dashboard developed for social media analysis which focus on the cyberbullying. This website is purpose for showing the result of the experiment for Final Year Project. ")
st.caption("*Take note that almost every feature available in the project are produce by other people and this website are only for experiment with lots of theory only.")



st.header("**Word Cloud**")
col1 , col2 = st.columns(2)
col1.subheader("Cyberbullying")
image = Image.open('Word Cloud Cyberbullying.png')
col1.image(image)


col2.subheader("Non Cyberbullying")
image1 = Image.open('Word Cloud NonCyberbullying.png')
col2.image(image1)

st.header("**Word Frequency**")
#Analysis Graph
df = pd.read_csv('Clean_Data3.csv')
#Sebelum ni pakai Tagged_MixedNew.csv

def cyberbullying_type_data(cb_type, column_name='target'):
    subset = df[df[column_name] == cb_type]
    text_data = subset.Text.values
    return text_data

def top_frequency_words(text, ng_range=(1,1), n=None):
    vector = CountVectorizer(ngram_range = ng_range).fit(text)
    bag_of_words = vector.transform(text)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vector.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

def most_used_words_phrases(cb_data, n=10):
    unigrams = top_frequency_words(cb_data,(1,1),n)
    unigram_data = pd.DataFrame(unigrams, columns = ['Text' , 'count'])
    return unigram_data


cyberbully_data = most_used_words_phrases(cyberbullying_type_data('Cyberbully'), 20)
noncyberbully_data = most_used_words_phrases(cyberbullying_type_data('NonCyberbully'), 20)


def create_word_bar(data, title):
    fig = px.bar(data, x = 'Text', y = 'count', color = 'Text',
                     labels={
                         'count': "Word Frequency"
                     },
                     title=title)
    
    st.plotly_chart(fig, use_container_width=True)


create_word_bar(cyberbully_data, 'Cyberbully')
create_word_bar(noncyberbully_data, 'NonCyberbully')


st.header("**Network Visualization**")
def make_networks(cyberbullying_type):
    subset = df[df['target'] == cyberbullying_type].Text.values
    bigrams = top_frequency_words(subset,(2,2), 100)
    
    bigrams_list = []
    for bigram in bigrams:
        bigrams_list.append(bigram[0].split())
        
    # initializing the graph
    graph = nx.DiGraph()
    #Adding the Nodes
    for node in bigrams_list:
        graph.add_nodes_from([node[0]])
    # Connecting the nodes with Edges
    for edge in bigrams_list:
        graph.add_edges_from([(edge[0], edge[1])])
    
    # Determing the degree of each node, that is the number of connected edges adjacent to a node.
    degrees = dict(nx.degree(graph))
    
    nx.set_node_attributes(graph, name = 'degree', values = degrees)
    # Adding an offset number of 5 so that nodes with small degree are also visible in the graph visualization
    number_to_adjust_by = 5
    adjusted_node_size = dict([(node, degree + number_to_adjust_by) for node, degree in nx.degree(graph)])
    nx.set_node_attributes(graph, name = 'adjusted_node_size', values = adjusted_node_size)
    
    size_by_this_attribute = 'adjusted_node_size'
    color_by_this_attribute = 'adjusted_node_size'
    #     Blues8, Reds8, Purples8, Oranges8, Viridis8, Spectral8
    color_palette = Blues8

    if cyberbullying_type == "Cyberbully":
        color_palette = Oranges8
    elif cyberbullying_type == "NonCyberbully":
        color_palette = Spectral8
        
    title = cyberbullying_type
    HOVER_TOOLTIPS = [
        ("Character", "@index"),
        ("Degree", "@degree")
    ]
    plot = figure(tooltips = HOVER_TOOLTIPS,
                  tools = "pan, wheel_zoom, save, reset", active_scroll = 'wheel_zoom',
                x_range = Range1d(-10.1, 10.1), y_range = Range1d(-10.1, 10.1), title = title)
    pos = nx.spring_layout(graph, scale=10, center=(0, 0))
    network_graph = from_networkx(graph, pos, scale=10, center = (0, 0))

    minimum_value_color = min(network_graph.node_renderer.data_source.data[color_by_this_attribute])
    maximum_value_color = max(network_graph.node_renderer.data_source.data[color_by_this_attribute])
    network_graph.node_renderer.glyph = Circle(size = size_by_this_attribute, 
                                               fill_color = linear_cmap(color_by_this_attribute, 
                                                                        color_palette, 
                                                                        minimum_value_color,
                                                                        maximum_value_color)
                                              )

    network_graph.edge_renderer.glyph = MultiLine(line_alpha = 0.5, line_width = 1)
    plot.renderers.append(network_graph)
    st.bokeh_chart(plot, use_container_width=True)
    return (plot, degrees, graph)


cyberbully_network_plot, cyberbully_degree, cyberbully_graph = make_networks('Cyberbully')
noncyberbully_network_plot,noncyberbully_degree, noncyberbully_graph = make_networks('NonCyberbully')

st.header("**Special Thanks to 🙌:**")
st.subheader("1.Huseinzol.Most of the dataset are collected from Huseinzol contribution through GitHub and including dependency , consituency parsing and Word Vector Visualization.")
st.caption("Link Dataset: https://github.com/huseinzol05/malay-dataset/blob/master/sentiment/supervised-twitter/data.csv")
st.caption("Link Consituency Parsing & Word Vector Visualization: https://malaya.readthedocs.io/en/stable/load-constituency.html#Parse-into-NLTK-Tree")
st.caption("Link Dependency Parsing : https://malaya-graph.readthedocs.io/en/latest/dependency-parser-text-to-kg.html")
st.divider()
st.subheader("2.Patrickxchong.Sentiment Analysis for bahasa.")
st.caption("Link HuggingFace: https://huggingface.co/patrickxchong/bert-tiny-bahasa-cased-sentiment")
st.divider()
st.subheader("3.Mesolitica.BERT base standard for bahasa.")
st.caption("Link HuggingFace: https://huggingface.co/mesolitica/bert-base-standard-bahasa-cased")
st.divider()

st.subheader("4.Aditya Singh.Network and Analysis Plots.")
st.caption("Link Kaggle: https://www.kaggle.com/code/aditya26sg/cyberbullying-tweet-networks-and-analysis-plots/notebook")

st.divider()
st.subheader("5.Sankha Subhra Mondal.SVM 3D Visualization.")
st.caption("Link Kaggle: https://www.kaggle.com/code/sankha1998/3d-plot-on-svm-non-linear")







