import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from justwatch import JustWatch
from sklearn.feature_extraction.text import CountVectorizer

#Converts the genre id to genre names 
def convert_genre(df):
    '''Pulls the list of genre from JustWatch API and converts the genre id to assigned genre names (genre is stored as a list of string ids)'''
    just_watch = JustWatch(country='US')
    genre = just_watch.get_genres()
    genre_dict = {}
    for i in genre:
        genre_dict[i['id']] = i['translation']
    
    #Convert string genre into a list then to an integer before getting values from the genre_dict. 
    df['genre'] = df['genre'].map(lambda i: i.strip('[]').split(','))
    df['genre'] = df['genre'].map(lambda x: [genre_dict.get(int(i)) for i in x])
    return df 


#Creates barplot that let's you specify if it is a horizontal or vertical plot
def count_plot(df, title, x=None, y=None):
    '''Creates a barplot given an arguement for either an x (vertical bar plot) or y (horizontal bar plot)'''
    plt.figure(figsize=(15,5))
    sns.countplot(x=x, y=y, data=df, palette=['slategrey'])
    plt.title(title)
    
#Creates a line plot
def line_plot(x, y, y_label, title):
    '''Creates a line plot with pre-set figure size (15,5)'''
    plt.figure(figsize=(15,5))
    plt.plot(x, y)
    plt.ylabel(y_label)
    plt.title(title);
    
#Arrange genre column to a list of strings 
def preprocessing_genre(df):
    '''Converts genre column from a string to a list of string. Additionally, removes extra characters such as space, &, and '' from the string'''
    df['genre'] = df['genre'].map(lambda i: " ".join([x.lstrip(" '").rstrip("''")\
                                      .replace(" & ", "and").replace(" ", "").replace("-", "") 
                                      for x in i.strip("[]").split(",")]))
    return df 

#Arrange column into a document-matrix
def tokenizer(df):
    '''tokenize column of interest and returns it into a pandas data frame.'''
    cvec = CountVectorizer()
    tokens = cvec.fit_transform(df)
    genre = pd.DataFrame(tokens.toarray(), columns=cvec.get_feature_names())
    return genre 

#Get Directors for shows/movies
def df_director(file):
    director_list = []
    for data in file:
        director = {}
        if data.get('credits') == None:
                director['director'] = None 
        else:
            for items in data.get('credits'):
                if items['role'] == 'DIRECTOR':
                    director['director'] = items['name']
        director_list.append(director)
    return pd.DataFrame(director_list)

#Fix the Rating system aggregating ratings that are similar to each other
def fix_rating(df):
    '''Aggregate similar ratings together (Ex: PG-13(there a space after) to PG-13'''
    df['rating'].replace('M/PG', 'PG', inplace=True)
    df['rating'].replace('PG-13 ', 'PG-13', inplace=True)
    df['rating'].replace('GP', 'PG', inplace=True)
    df['rating'].replace('Passed ', 'Passed', inplace=True)
    df['rating'].replace('TV-14 ', 'TV-14', inplace=True)
    df['rating'].replace('TV-G ', 'TV-G', inplace=True)
    df['rating'].replace('R ', 'R', inplace=True)
    df['rating'].replace('TV-MA ', 'TV-MA', inplace=True)
    df['rating'].replace('TV-PG ', 'TV-PG', inplace=True)
    return df
    