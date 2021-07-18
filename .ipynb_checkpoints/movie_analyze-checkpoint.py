import numpy as np
import pandas as pd
import re
import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage,dendrogram

movies_df = pd.read_csv("movies.csv")
movies_df.head()

nltk.download('punkt')

stemmer = SnowballStemmer("english")
def tokenize_and_stem(text):
    
    # Tokenize by sentence, then by word
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    
    # Filter out raw tokens to remove noise
    filtered_tokens = [token for token in tokens if re.search('[a-zA-Z]', token)]
    
    # Stem the filtered_tokens
    stems = [stemmer.stem(t) for t in filtered_tokens]
    
    return stems

words_stemmed = tokenize_and_stem("Today (May 19, 2016) is his only daughter's wedding.")
print(words_stemmed)

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem,
                                 ngram_range=(1,3))

tfidf_matrix = tfidf_vectorizer.fit_transform([x for x in movies_df["Plot"]])

print(tfidf_matrix.shape)

km = KMeans(n_clusters=5)

km.fit(tfidf_matrix)

clusters = km.labels_.tolist()

movies_df["Cluster"] = clusters

movies_df['Cluster'].value_counts() 

similarity_distance = 1 - cosine_similarity(tfidf_matrix)

mergings = linkage(similarity_distance, method='complete')

dendrogram_ = dendrogram(mergings,
               labels=[x for x in movies_df["Title"]],
               leaf_rotation=90,
               leaf_font_size=16,
)

fig = plt.gcf()
_ = [lbl.set_color('r') for lbl in plt.gca().get_xmajorticklabels()]
fig.set_size_inches(108, 21)

plt.show()

fig.savefig('movies_demo.png')

def find_similar(title):
  index = movies_df[movies_df['Title'] == title].index[0]
  vector = similarity_distance[index, :]
  most_similar = movies_df.iloc[np.argsort(vector)[1], 1]
  return most_similar
print(find_similar("Coco")) # prints "Stuart Little"