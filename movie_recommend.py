#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
df1=pd.read_csv('tmdb_5000_credits.csv')
df2=pd.read_csv('tmdb_5000_movies.csv')


# In[2]:


df1.info()


# In[3]:


df2.info()


# In[4]:


df1.describe().T


# In[5]:


df1.describe(include=object).T


# In[6]:


df2.describe().T


# In[7]:


df2.describe(include=object).T


# In[8]:


df1.columns


# In[9]:


df2.columns


# In[10]:


df1.columns = ['id', 'title', 'cast', 'crew']


# In[11]:


# Merging df1 and df2 on id

df = df2.merge(df1, on=['id','title'])


# In[12]:


df.head()


# In[13]:


df.info()


# In[14]:


df['release_date'] = pd.to_datetime(df['release_date'])


# # Demographic Filtering
# 
# - choose a metric to rate a movie
# - recommend based on the rated movie to the users
# 
# Simple average rating doesn't make sense, since a movie with high rating with few votes cannot be considered better than a movie with sligtly lower rating with many votes. Thus we will use IMDB's weighted rating(WR), which is following

# $$
# Weighted Rating(WR)=(\frac{v}{v+m}\cdot R)+(\frac{m}{v+m}\cdot C),
# $$
# where
# - v is the number of votes the movie has received
# - m is the minimum number of votes required to be listed
# - R is the average rating of the movie
# - C is the mean vote across the whole report

# In[15]:


C = df['vote_average'].mean(); C


# In[16]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8,3))
# Create a distribution plot
sns.histplot(df['vote_count'], kde=True, color='skyblue')

# Add labels and title
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Distribution Plot')

# Show plot
plt.show()


# In[17]:


m = df['vote_count'].quantile(.9)


# In[18]:


movies = df.loc[df['vote_count']>=m]
movies.shape


# In[19]:


def wr(x, m = m, C = C):
    v = x['vote_count']
    R = x['vote_average']
    
    return(v/(v+m)*R+m/(v+m)*C)


# In[20]:


movies['score'] = movies.apply(wr, axis = 1)


# In[21]:


movies = movies.sort_values('score',ascending=False)


# In[22]:


movies[['title','score','release_date']].head(10)


# Okay those movies are too old. Let's explore the movies after 2001-01-01.

# In[23]:


movies.sort_values('release_date')['release_date']


# In[24]:


movies1 = df.loc[df['release_date']>='2001-01-01']
movies1.shape


# In[25]:


movies1['score'] = movies1.apply(wr, axis = 1)


# In[26]:


movies1 = movies1.sort_values('score',ascending=False)
movies1[['title','score','release_date']].head(10)


# In[27]:


import matplotlib.pyplot as plt

pop = df.sort_values('popularity',ascending=False).head(5)

plt.figure(figsize=(8,3))
plt.barh(pop['title'],pop['popularity'])
plt.gca().invert_yaxis()
plt.xlabel('Popularity')
plt.title('Top 5 Popular Movies')
plt.show()


# # Content Based Filtering

# For anyone who has dabbled in text processing, you're likely familiar with the necessity of transforming the textual content into a numerical format that a computer can understand. A common step in this process involves converting the "word vector" of each document or text overview. In this discussion, we'll explore how to compute the Term Frequency-Inverse Document Frequency (TF-IDF) vectors for each document overview, enhancing the way we analyze and interpret textual data.
# 
# ### Understanding TF-IDF
# 
# Before we dive into the computation, let's break down what TF-IDF stands for and why it's important:
# 
# 1. **Term Frequency (TF):** This measures the relative frequency of a word in a document compared to the total number of words it contains. The formula is straightforward: $(TF = \frac{\text{Number of times term appears in a document}}{\text{Total number of terms in the document}}$). It helps in understanding how significant a word is in a specific document.
# 
# 2. **Inverse Document Frequency (IDF):** This assesses the general importance of a term across all documents. The logic here is that words appearing in many documents may not be useful in distinguishing one document from another. The formula used is: $(IDF = \log\left(\frac{\text{Total number of documents}}{\text{Number of documents containing the term}}\right)$). IDF diminishes the weight of terms that occur very frequently across documents.
# 
# By multiplying TF by IDF, we can determine the overall importance of each word to the documents in which they appear, balancing the frequency of terms within individual documents and across the entire document set.
# 
# ### The Power of TF-IDF in Action
# 
# Utilizing TF-IDF transforms a text dataset into a structured format, producing a matrix where:
# - Each **column** represents a unique word within the overall document vocabulary (comprising all words that appear in at least one document).
# - Each **row** represents an individual document (e.g., a movie overview).
# 
# This transformation is particularly effective in diminishing the significance of frequently occurring words in plot overviews that might otherwise dominate the thematic analysis of texts, thereby refining the final similarity scores between documents.
# 
# ### Leveraging Scikit-learn for Efficiency
# 
# Thankfully, implementing TF-IDF doesn't require building the algorithm from scratch. The `scikit-learn` library in Python provides a convenient `TfidfVectorizer` class, allowing you to generate a TF-IDF matrix efficiently with just a few lines of code. This is a boon for data scientists and developers, streamlining the preprocessing of textual data for machine learning and analytics projects.
# 
# In summary, the computation of TF-IDF vectors is a pivotal step in text processing, enabling the extraction of meaningful numerical features from raw text. This, in turn, facilitates a wide array of applications, from content recommendation systems to thematic analysis of documents. Thanks to tools like `scikit-learn`, harnessing the power of TF-IDF is more accessible than ever, making it an essential technique in the toolkit of modern data professionals.

# In[39]:


#Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer

#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
df['overview'] = df['overview'].fillna('')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(df['overview'])

#Output the shape of tfidf_matrix
tfidf_matrix.shape


# In our dataset comprising 4,800 movies, we observe a diverse vocabulary of over 20,000 unique words used across their descriptions.
# 
# Armed with this vocabulary matrix, we embark on the task of computing similarity scores between movies. Among the plethora of available metrics—such as Euclidean, Pearson, and cosine similarity—there's no one-size-fits-all solution. Each metric excels in distinct scenarios, making it imperative to experiment with various options.
# 
# For our analysis, we opt for cosine similarity due to its independence from magnitude and its efficiency in computation. Cosine similarity offers a straightforward and rapid approach to quantifying the resemblance between two movies. Mathematically, it is articulated as follows:
# 
# Certainly! The cosine similarity between two vectors $( \mathbf{a} $) and $( \mathbf{b} $) is defined as the cosine of the angle between them in a multidimensional space. Mathematically, it's represented as:
# 
# $$ \text{cosine_similarity}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|} $$
# 
# Where:
# - $( \mathbf{a} \cdot \mathbf{b} $) denotes the dot product of vectors $( \mathbf{a} $) and $( \mathbf{b} $).
# - $( \|\mathbf{a}\| $) and $( \|\mathbf{b}\| $) represent the Euclidean norms (lengths) of vectors $( \mathbf{a} $) and $( \mathbf{b} $), respectively.
# 
# This formula calculates the cosine of the angle between the vectors, which ranges from -1 (perfectly opposite) to 1 (perfectly aligned), with 0 indicating orthogonality (perpendicularity).
# 
# In the context of movie similarity, these vectors typically represent feature vectors derived from movie descriptions or other relevant attributes, and the cosine similarity measures how similar these movies are based on their feature representations.

# In[40]:


# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


# We will develop a function that, upon receiving a movie title, returns a selection of the top 10 movies most akin to it. To accomplish this, our first step involves creating a reverse map that links movie titles back to their respective indices within our metadata DataFrame. This setup is essential for pinpointing the exact location of a movie in our dataset based on its title.

# In[41]:


#Construct a reverse map of indices and movie titles
indices = pd.Series(df.index, index=df['title']).drop_duplicates()


# In[43]:


# Function that takes in movie title as input and outputs most similar movies
def get_recommend(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return df['title'].iloc[movie_indices]


# In[48]:


get_recommend('Inception')


# In[ ]:




