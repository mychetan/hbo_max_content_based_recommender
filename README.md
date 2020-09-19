# HBO Max Recommender System

---
### Content: 
- [Problem Statement](#Problem-Statement)
- [Executive Summary](#Executive-Summary)
- [Project Files](#Project-Files)
- [Data Directory](#Data-Directory)
- [Data Collection and Cleaning](#Data-Collection-and-Cleaning)
- [Modeling](#Modeling)
- [Conclusion, Recommendations, and Future Improvements](#Conclusion,-Recommendations,-and-Future-Improvement)
- [References](#Reference)

---

### Problem Statement

On May 27, 2020, AT&T and Warner Media launched their newly re-invented streaming service called HBO Max. It boasts over 10,000 hours of content with 1980 titles (1610 movies and 370 shows). Additionally, it's partnered with 35 content providers ranging from DC Comics, Sesame Workshop, Studio Ghibli, Cartoon Network, Adult Swim, Bad Robot Production, etc. However, with all of its excellent content, it is falling behind against other competitors. Upon inspection of its interface, a noticeable missing component is the "Recommended for you" section. Recommender systems have become an industry standard. Its usage is seen across various streaming services, such as Netflix and Hulu. HBO did try to address this issue with [Recommended by Human](https://www.humanreco.hbo.com/), where users can watch testimonies or read tweets about HBO content. As nicely designed, the webpage is, however, it lacks efficiency in user interface. For one, users will have to go to an external website to get recommendations. Second, it is a not well-known website. As a fellow HBO Max user, I did not know this existed until I did a deep dive into the HBO Max platform. Therefore, I created a Content-Based Recommender System build on HBO Max current library to address the issues I have mentioned above. This system uses subscriber's watch history to return similar titles based on genre, MPAA/TV rating, plot summary, and IMDB ratings. This project aims to introduce a system to incorporate into the existing HBO Max app so that it will be more accessible for HBO Max subscribers to discover new content.  

---

### Executive Summary 

I establish a Content-Based Recommender system built on HBO Max library using Natural Language Processing (NLP). It first creates a vectorized matrix using the title's genre and plot. This matrix is then aggregated to the title's numerical features, such as IMDB and encoded MPAA ratings. A cosine similarity is calculated from the generated matrix to return similar contents to the user based on inputted title (previously watch history). The final recommender system is arranged into a flask app located in the App directory called app.py. 

---

### Project Files (Repository Organization)

#### Main Directory
- capstone_presentation.pdf
- README.md
- images: Stores the images used in the README
- App: Contains the flask app and its associated html, css, and javascript files
- **Code Directory:**
    - 01_data_collection_cleaning.ipynb: Notebook code for collecting and cleaning HBO Max data. 
    - 02_EDA.ipynb: Notebook code for HBO Max Data EDA
    - 03_recommender.ipynb: Notebook that contains results and analysis for Simple Recommender and Recommender Systems 1-3 
    - 04_recommender_BERT.ipynb: Notebook that contains code, results, and analysis from Recommenders 4 and 5
    - **Script Directory:**
        - data_collection.py: Python script used to scrap and organize data to a CSV file
        - function.py: Python script that contains functions used throughout the project 
        - models.py: Python script that contains the code for Simple Recommender and Recommenders 1-3 
    - **Pickle Directory:** Contains pickled items
        - log_reg.pkl: Pickled Logistic Regression model 
        - pca.pkl: Pickled PCA model 
        - X.pkl: Pickled X variable used to get predict probabilites 

---

### Data Directory

|Feature|Type|Description|
|---|---|---|
|ID|integer|Unique JustWatch.com id identifier|
|Title|string|Name of show or movie|
|Type|string| Identify content type as either a show or movie|
|Year|integer|Year released|
|Plot|string|Plot summary of show or movie|
|Genre|list of strings|Genre of shows or movie|
|Rating|string|Motion Picture Association film rating system (MPAA)/TV ratings|
|Popular Score|float|TMDB popularity rating|
|TMDB Rating|float|TMDB rating score (1-10)|
|IMDB Rating|float|IMDB rating score (1-10)|

---

### Data Collection and Cleaning 

**Data Collection:** 

Due to the JustWatch API's limitations, I utilized a third-party scrapper (JustWatch.py) to gather the entire HBO Max's library. Credit for the scrapper belongs to the original creator [dawoudt](https://github.com/dawoudt/JustWatchAPI). 

For this project, the data collection step is broken down to two-part: First and Second Requests. The first part's goal is to collect all the titles that are being distributed by HBO Max. For project simplicity, compiled titles are only from the U.S. region. I begun by conducting an initial pull to gather data to be used as guidelines. The initial output is a dictionary that contains a key called total_count, which was used as a guiding parameter to dictate when the while loop will stop. This information is then passed through a second function to collect the title, unique id, and type (show/movie) of HBO Max content. The collected data is then arranged into a dataframe called df_1. A problem observed from the scraper is that it raises an error when the status code is not equal to 200. To mitigate this scenario, I enclosed the while loop in a try and except statement. Additionally, I added print statements to track the page number and total count of data collected and pickled each pull request into a .txt file. This setup will make it easier to resume data collection if the cycle breaks. 

The Second Request aims to get the granular details needed for the EDA and modeling/recommender process. It uses the title's unique id and content type as identifiers to collect data from the API. Like the method above, output products are arranged into a dataframe with the following columns (id, plot, MPAA/TV rating, genre, popularity score, IMDB rating, and TMDB rating). This dataframe is concatenated to df_1 to create the final dataframe. The second part of the data collection step follows the same precautions as the first part. 

Each loops are throttled for 5 seconds to comply with rate-limit policies and politeness; the selected time is from a trial and error process.

The final combined dataframe contains 10 columns and 1980 rows. Around 20% of the data has missing information. These values are manually imputed using data from the IMDB website. For future iterations, a process similar to the first and second request would be ideal. However, IMDB requires an API key to access their database. Due to this project's timeline, it would not be feasible to wait for a response on their end. Additionally, their scrapper would require re-doing the entire process again since each show and movie will have a different id.  

The data collection step is done through Amazon's AWS server to maximize time and efficiency. The data collection script is found in the script direction under the name data_collection.py. 

**Data Cleaning:** 

I first converted the genre ids to their assigned names using the `convert_genre` function.  Then, I removed extra spaced from the MPAA/TV ratings using the `fix_rating` function. 

Both mentioned functions are found in the script directory under the function.py. 

---

### Exploratory Data Analysis

The HBO Max library is consists of 370 TV shows and 1610 movies. The majority of their content has an MPAA/TV rating of R, TV-MA, and PG-13. Additionally, we see that the top genres are Action-Adventure, Comedy, and Drama. Interestingly, these observations follow the history and identity that HBO had established. The company started as a tv channel, mainly showing movies targeted for adult audiences.

Additionally, it reflects the platform's partnered content providers. The majority of partnered film studios and production companies are focus on creating films or content targeted for adult audiences. Some examples are Adult Swim, Bad Robot Productions (known for titles such as Alias, Fringe, and Westworld), TBS (known for Conan, Seinfeld, and Friends), and TNT (known for broadcasting classic films). However, HBO Max also caters to young audiences, as seen by their partnership with Sesame Workshop and Cartoon Network. 

**Figure 1: Breakdown of HBO Max content type (Movie or Show)**
![](image/TV_vs_Movie.png) 

**Figure 2: HBO Max content by MPAA Rating***
![](image/MPAA_ratings.png)

**Figure 3: HBO Max content by genre**
![](image/Genre.png)

I also explore the relationship between year released, popularity score, and IMDB and TMDB ratings. Since the IMDB and TMDB follow similar behavior, for simplicity, the explanation will be clumped into one. There is a negative trend observed for year released and IMDB/TMDB ratings. Classic shows appear to have a higher average rating than newer shows. In contract, popularity and year release has a positive relationship, where more recent contents are more popular compared to classics.

Furthermore, the majority of HBO Max content has an average rating of between 5-8. After examining top and bottom titles, high popularity titles are well-established shows/movies with long-run history with strong fanbases such as Friends, South Park, and Doctor Who. In comparison, the least popular shows are mostly documentaries.

**Figure 4: Year Release vs Populary Score**
![](image/year_popularity.png)

**Figure 5: Year Release vs Average Rating**
![](image/year_vs_avgrating.png)

**Figure 6: Popularity Score vs Average Rating**
![](image/popularity_vs_ratings.png)

---
### Modeling
The modeling process can be broken down into three categories: Simple Recommender, Content-Based Recommender using TfidVectorizer, and Content-Based Recommender using BERT. Results are evaluated based on personal judgement and through google recommemdations. 

**Simple Recommender:** This recommender follows a very straightforward approach. It uses various filtering techniques to get the recommended titles. The recommender's goal is to build a generalized system that showcases movies/shows that fall within the same genre while returning popular highly-rated content. The reasoning behind this idea is that audiences tend to prefer popular shows/movies with high ratings. The setup includes generating an engineered feature that multiples popularity score with IMDB ratings; this will help magnify the desired content. Although successful, the model suffers from extreme limitations. First, it lacks user personalization, where it will continue to provide the same recommendations to anyone as long as they put the same genre. Second, the established link between contents is relatively shallow; it only considers similarities in genre.  

**Content-Based Recommender using TfidVectorizer:** This recommender utilizes Natural Language Processing (NLP) to provide recommendations to users. Unlike the previous recommender, this will provide a more personalized result. For this category, three models were developed to test various feature combinations: 
1. **Plot & Genre (Recommender 1):** Unlike the Simple Recommender, this system uses a vectorized matrix of plots and genres to establish content relationships, leading to a more personalized result. Although the top four recommended shows are similar to South Park, however, passed that point, the suggested titles shifted to children shows. They do significantly differ from South Park, which contains more explicit and mature content. A limitation of this recommender is that it fails to consider MPAA/TV ratings. As evident from the discrepancies that arose from the difference in intended audiences.
2. **Plot, Genre, & Predicted Probabilities of MPAA/TV Ratings (Recommender 2):** For this recommender, I decided to add more complexity to the system by utilizing a Logistic Regression model to classify MPAA/TV ratings. The reasoning behind this is that the effects of the features used to train the model will add more weight to the MPAA/TV ratings, emphasizing it when calculating cosine similarity. To train the model, I used the following features: vectorized plot, vectorized genre, IMDB ratings, TMDB ratings, year released, popularity score, and type (show/movie). Due to the large shape of the training matrix (1980 X 60,000+), I used a PCA, with only 50 components, to reduce dimensionality. The extracted predicted probabilities are then appended to the document matrix, similar to Recommender 1, and used to calculate the cosine similarity. This recommender did better than Recommender 1. Using the same testing title (South Park), only two recommended shows did not fit with the rest. An explanation for the error is that the model is introducing noises creating the discrepancies in the recommendations.  
3. **Plot, Genre, MPAA/TV Ratings, & Average Ratings (Recommender 3):** For this recommender, I  decided to incorporate MPAA/TV ratings in a more straightforward method. I first encoded the MPAA/TV ratings into numerical values (selected values are based on own interpretations of those ratings). I then added them together with the IMDB ratings to generate a matrix of numerical features. They are then appended to a vectorized matrix, similar to Recommender 1, to calculate cosine similarity.
Like Recommender 2, the encoded MPAA/TV ratings will magnify its effect when calculating similarities, resulting in better fine-tune recommendations. Out of the three recommenders previously mentioned, this one performed the best. It is not showing any questionable or error in its recommendations. However, I was curious to see if switching to a more sophisticated vectorizer will affect the results.

**Content-Based Recommender using BERT:** 
1. **Plot, Genre, Average Rating (IMDB & TMDB score), and MPAA/TV Rating (Recommender 4):** For this recommender, I used the same techniques from Recommender 1. I vectorized each selected feature and aggregated them together into a document matrix to calculate cosine similarity. The results from this recommender are worst compared to the other ones mentioned above. It is giving recommendations that does have any relevance to the testing title (South Park). An explanation for this observation is that similarities are based on the count of identical ids. Therefore, there is a greater emphasis on the plot summary, while minimizing the effect of genre when calculating results.

2. **Plot, Genre, and encoded MPAA/TV Ratings:** To address the problems from Recommender 4, I decided to combine the procedure from Recommender 3 and 4. However, instead of using IMDB ratings, I used the average between the IMDB and TMDB ratings. The resulting recommendations did better than Recommender 4. I began to see more suggested titles similar to South Park. However, it still exhibits results that are different from the testing title. 

**Recommender 3 was used for demo in a flask app. Please look as App directory for details**

---

### Conclusion, Recommendations, and Future Improvements

My recommendation to AT&T and Warner Media is to incorporate Recommender 3 into the existing HBO Max application. It is a Content-Based recommender system that is built on the current HBO Max library. This proposal will be an excellent move for the company due to the benefits of Recommender Systems. First, it can improve user's retention rate; catering to user's preferences can lead to long term loyal subscribers. Second, serving similar content leads to habit development that influences users usage pattern. Third, there is a strong correlation between view counts of a video and its top referer video. Lastly, incorporating this recommender will make it easier and more efficient for subscribers to discover new content without the hassle of connecting to another external website. 

To differentiate HBO Max from other streaming services, I did not add any bias towards HBO original titles. The reason behind this is that a primary concern of users from other platforms is that recommender systems keep returning original or in-house contents, removing opportunities to find hidden gems. Similarly, series creators follow the same sentiment, where their creations get canceled due to the higher priority for original titles. Through this method, it gives every title an even playing field for getting recommended.

As mentioned earlier future aspiration is integrating it into the HBO Max app to offer personalized recommendations to users. Additionally, I would like to conduct various A/B testing with the current HBO Max interface to learn more about its user demographics. Moreover, this recommender is just a starting point. There are plenty of methods that can be employed to add complexity to this recommender. For instance, adding Collaborative Filtering will result in better-personalized results. Enabling users to rate their watch history will give better insight into their preferences. Another example is applying a deep learning model, such as reinforcement learning. In this way, the system can continue to grow and conform to the user's preferences, the more they utilize the HBO Max app. Lastly, adding additional features, such as directors, keywords, and actors, to the proposed recommender. These other features will add more complexity to the recommender, resulting in better clustering of similar titles.  

---

### References
- HTML template taken from https://html5up.net/
- https://sigmoidal.io/recommender-systems-recommendation-engine/
- https://www.statista.com/statistics/1136139/hbo-now-to-hbo-max-subscribers-post-launch-us/
- https://www.statista.com/statistics/1116918/possible-hbo-max-subscribers-by-generation/
- https://www.statista.com/statistics/778912/video-streaming-service-multiple-subscriptions/
- https://www.researchgate.net/publication/220269659_The_impact_of_YouTube_recommendation_system_on_video_views
- https://fortune.com/2019/07/25/netflix-cancels-tuca-and-bertie-algorithm/
- https://www.kaggle.com/rounakbanik/movie-recommender-systems
- https://www.kdnuggets.com/2019/11/content-based-recommender-using-natural-language-processing-nlp.html
- https://medium.com/@armandj.olivares/building-nlp-content-based-recommender-systems-b104a709c042
