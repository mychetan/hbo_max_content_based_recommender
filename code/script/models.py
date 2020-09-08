import pandas as pd 
import numpy as np 
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

df = pd.read_csv('../data/final_hbo_data_2_1.csv', index_col=0)

#Simple Recommender System: Filters shows based on genre and type and return top content
def top_content(genre=None, rank=10, content_type = None):
    '''Returns the most popular show based on genre or type (show or movie)'''
    
    data = df.copy()
    data['popularity_imdb'] = data['popularity_score'] * data['imdb_rating']
    
    #Sort by genre if genre is provided 
    if genre == None:
        data = data 
    else:
        data = data.loc[df['genre'].str.lower().str.contains(genre.lower())]
    
    #Sort by content type (movie or show ) if provided 
    if content_type == None:
        data = data 
    elif content_type.lower() == 'show':
        data = data.loc[data['type'] == 'show']
    else:
        data = data.loc[data['type'] == 'movie']
    
    #Sort data by the popularity_imdb scores and return only the 95th percentile 
    data.sort_values('popularity_imdb', ascending=False, inplace=True)
    recom = data.loc[data['popularity_imdb'] > data['popularity_imdb'].quantile(0.95)]
    return recom[['title', 'year', 'plot', 'genre', 'rating', 'imdb_rating', 'type']].head(rank)


#Recommender 1: Content-Based recommender that uses the plot and genre to generate recommendations
def recommender_1(title, num=5):
    '''Recommender that plot summary and genre. Takes in an arguement of a title and number of contents to display'''
    data = df.copy()
    
    #Vectorizered the genre and plot and concatenate the matrix together
    tf_genre = TfidfVectorizer()
    genre = tf_genre.fit_transform(data['genre'])
    tf = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
    tf_matrix = tf.fit_transform(data['plot'])
    tf_matrix = np.append(tf_matrix.toarray(), genre.toarray(), axis=1)
    
    #Get the cosine similarity of matrix 
    cosine_sim = cosine_similarity(tf_matrix, tf_matrix)
    
    #Create a Series where data index are the values and the index are the titles of movies/shows 
    indices = pd.Series(data.index, index=data.title.str.lower())
    
    #Sorts the cosine similarity for the specified title and pull the top n similar shows. Return a data frame
    sim_score = sorted(list(enumerate(cosine_sim[indices[title.lower()]])), key=lambda x: x[1], reverse=True)
    recomm_contents = data.loc[[i[0] for i in sim_score[1:num+1]]]
    return recomm_contents[['title', 'year', 'plot', 'genre', 'rating', 'imdb_rating', 'type']]


#Recommender 2: Content-Based recommender that uses plot, genre, and predict probabilities of genre.
#Uses Logistic Regression to get MPAA/TV ratings prediction probabilities.
def log_reg(df):
    '''Logistic regression model that predicts the MPAA/TV ratings'''
    
    df2 = df.copy()
    ratings = ratings = {'Not Rated': 0,'Approved': 0, 'Passed': 0,'TV-Y': 1, 'TV-Y7': 2,'TV-G': 3, 'G': 3, 'TV-PG': 4,
               'PG': 4,'PG-13': 8,'TV-14': 10,'R': 12,'TV-MA': 13,'NC-17': 15}

    #Binarize type(show:1, and movie:0) and convert MPAA/TV ratings to numerical equivalent
    df2['type'] = df2['type'].map({'show': 1, 'movie': 0})
    df2['rating'] = df2['rating'].map(ratings)

    #Set X and y variables. Drop the id, rating, and title from the X variable
    X = df2.drop(columns= ['id', 'rating', 'title'])
    y = df2['rating'] 

    #Vectorize movies/shows' plot and convert document-matrix into a data frame
    tf = TfidfVectorizer(stop_words='english', ngram_range=(1,2), min_df=15)
    tokens = tf.fit_transform(X['plot'])
    df_tokens = pd.DataFrame(tokens.toarray(), columns= tf.get_feature_names())

    #Vectorize movies/shows' plot and convert the document-matrix into a data frame
    tf_genre = TfidfVectorizer()
    genre = tf_genre.fit_transform(X['genre'])
    df_genre = pd.DataFrame(genre.toarray(), columns= tf_genre.get_feature_names())

    #Concatenate the plot and genre data frame and combine it with the original X variable.
    #Drop the plot and genre columns 
    token_genre = pd.concat([df_tokens, df_genre], axis=1)
    X = pd.concat([X, token_genre], axis=1).drop(columns=['plot', 'genre'])

    #Split data to training and testing sets 
    X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42, stratify=y)

    #Reduce Model Dimensionality by running a PCA 
    pca = PCA(n_components=50, random_state=42)
    Z_train = pca.fit_transform(X_train)
    Z_test = pca.transform(X_test)
    
    #Since y-variable is multi-class a multinomial will be set as an agruement, also solver was change to newton-cg
    #so that the model can converge
    lr = LogisticRegression(solver='newton-cg', multi_class='multinomial')
    lr.fit(Z_train, y_train)
    
    return lr, pca, Z_train, Z_test, y_train, y_test, X 

def recommender_2(title, num=5):
    '''Recommender based on genre, plot, and predict probabilities of MPAA/TV rating. Takes an arguement of title and number of content to display''' 
    
    #Load needed info
    lr, pca, Z_train, Z_test, y_train, y_test, X = log_reg(df)
    data = df.copy()
    
    #Vectorized genre
    tf_genre = TfidfVectorizer()
    genre = tf_genre.fit_transform(data['genre'])
    
    #Vectirized plot
    tf = TfidfVectorizer(stop_words='english', ngram_range=(1,2), min_df=15)
    tf_matrix = tf.fit_transform(data['plot'])
    
    #Concatenate vectorized plot and genre matrix. Then, append predict probabilities matrix
    tf_matrix = np.append(tf_matrix.toarray(), genre.toarray(), axis=1)
    tf_matrix = np.append(tf_matrix, lr.predict_proba(pca.transform(X)), axis=1)
    
    #Get the cosine similarity 
    cosine_sim = cosine_similarity(tf_matrix, tf_matrix)
    
    #Generate a series when content index as values and titles as the index
    indices = pd.Series(data.index, index=data.title.str.lower())
    
    #Sort by similarity score and filter for the top n most similar shows
    #Filter data frame with the indices isolated
    sim_score = sorted(list(enumerate(cosine_sim[indices[title.lower()]])), key=lambda x: x[1], reverse=True)
    recomm_contents = data.loc[[i[0] for i in sim_score[1:num+1]]]
    
    #Returned data frame
    return recomm_contents[['title', 'year', 'plot', 'genre', 'rating', 'imdb_rating', 'type']]


#Recommender 3: Content-Based recommender that is based on plot, genre, MPAA/TV rating, and average rating (IMDB & TMDB)
def recommender_3(title, num=5):
    '''Recommender that is based on plot, genere, and MPAA/TV rating, and average score (IMDB and TMDB)'''

    #Set dataframe and ratings dictionary
    data = df.copy()
    ratings = {'Not Rated': 0,'Approved': 0, 'Passed': 0,'TV-Y': 1, 'TV-Y7': 2,'TV-G': 3, 'G': 3, 'TV-PG': 4,
               'PG': 4,'PG-13': 8,'TV-14': 10,'R': 12,'TV-MA': 13,'NC-17': 15}
    
    #Get average rating score and convert MPAA/TV ratings to numeric equivalent 
    data['rating_score'] = data['rating'].map(ratings)
    num_info = data[['imdb_rating', 'rating_score']]
    
    #Vectorized genre
    tf_genre = TfidfVectorizer()
    genre = tf_genre.fit_transform(data['genre'])
    
    #Vectirized plot
    tf = TfidfVectorizer(stop_words='english', ngram_range=(1,2), min_df=15)
    tf_matrix = tf.fit_transform(data['plot'])
    
    #Concatenate vectorized plot and genre matrix. Then, append predict probabilities matrix
    tf_matrix = np.append(tf_matrix.toarray(), genre.toarray(), axis=1)
    tf_matrix = np.append(tf_matrix, num_info, axis=1)
    
    #Get the cosine similarity 
    cosine_sim = cosine_similarity(tf_matrix, tf_matrix)
    
    #Generate a series when content index as values and titles as the index
    indices = pd.Series(data.index, index=data.title.str.lower())
    
    #Sort by similarity score and filter for the top n most similar shows
    #Filter data frame using the isolated indices of similar shows
    sim_score = sorted(list(enumerate(cosine_sim[indices[title.lower()]])), key=lambda x: x[1], reverse=True)
    recomm_content = data.loc[[i[0] for i in sim_score[1:num+1]]] 
    return recomm_content[['title', 'year', 'plot', 'genre', 'rating', 'imdb_rating', 'type']]