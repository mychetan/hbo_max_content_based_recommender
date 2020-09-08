#Load needed imports
import time 
import pickle
import pandas as pd
from justwatch import JustWatch

#Instantiate JustWatch and set country to US. This isolates the collected data to only come from US content. 
just_watch = JustWatch(country='US')

#Pulls the first page of HBO Max content from JustWatch API to use as parameters to pull other pages. 
def initial_data(): 
    data = just_watch.search_for_item(providers=['hbm'], page=1)
    return [data], data['total_results']

#Pulls the entire HBO Max library from the JustWatch API and returns an output of a list of dictionaries (JSON files). Using the data collected from the initial pull total_size defines the total count of the entire HBO Max library. This was set as a guide for the while loop to continue pulling information until all the data is collected. 
def hbo_content_list():
    data_list, total_size = intial_data() 
    size = len(data_list[0]['items']) 
    page_num = 2 
    while size < total_size:
        try: 
            data = just_watch.search_for_item(providers=['hbm'], page=page_num)
            data_list.append(data)
            size += len(data['items'])
            page_num += 1 
        
            #Print out status of pull 
            if size % 30 == 0: 
                print("Number of data pulled:")
                print(size)
                print("Page Number:")
                print(page_num - 1)

            #Save each updated list into a textfile called intial_data
            with open('initial_data.txt', 'wb') as output:
                pickle.dump(data_list, output)
            
            time.sleep(5)
      
        except:
            pass
    return data_list

#Arrange pulled data into a pandas data frame, extracting information for content id, title, and type (show or movie) to be used as columns. 
def json_df_content():
    content = []
    data_list = hbo_content_list()
    for items in data_list:
        for item in items['items']:
            show = {}
            show['id'] = item['id']
            show['title'] = item['title']
            show['type'] = item['object_type']
            content.append(show)
    return pd.DataFrame(content)

df_1 = json_df_content()
df_1.to_csv('df_1.csv')

#Pull the granular details from each HBO Max content (i.e., plot, ratings, genre, cast, ...). The final output will be a list of dictionaries (JSON files). 
def add_info(df):
    content = []
    for i in df.index:
        show = just_watch.get_title(title_id = df.loc[i, 'id'], content_type= df.loc[i, 'type'])
        content.append(show)

        #Saves each pull into raw_data.txt 
        with open('raw_data.txt', 'wb') as output:
            pickle.dump(content, output)

        #Print number of data pulled at each loop, make it easier to resume if a 429 error occur
        print("Number of data pulled:")
        print(len(content))

        time.sleep(5)
    return content

#Arrange additional information data into a pandas data frame. Extracted information will be used a columns. 
def json_df_add():
    content = []
    raw_data = add_info(df_1)
    for data in raw_data:
        show_info = {}
        show_info['year'] = data.get('original_release_year')
        show_info['plot'] = data.get('short_description')
        show_info['genre'] = data.get('genre_ids')
        show_info['rating'] = data.get('age_certification')
        if data.get('scoring') == None:
                show_info['avg_rating'] = None 
        else:
            for score in data.get('scoring'):
                if score['provider_type'] == 'imdb:score':
                    show_info['imdb_rating'] = score['value']
                elif score['provider_type'] == 'tmdb:score':
                    show_info['tmdb_rating'] = score['value']
                if score['provider_type'] == 'tmdb:popularity':
                    show_info['popularity_score'] = score['value']
        
        content.append(show_info)
    return pd.DataFrame(content)

df_2 = json_df_add()
df_2.to_csv('df_2.csv')

#Combine two data frames into final CSV file. 
hbo_data = pd.concat([df_1, add_info_2], axis=1)
hbo_data.to_csv('hbo_data.csv')




