import numpy as np 
from flask import Flask, Response, request, render_template
import static.models as model

app = Flask('recommender_app', static_folder="/Users/mychetan/Projects/content_based_recommender/App/static")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/submit')
def submit():
    data = request.args
    title = str(data['Title'])
    top = int(data['Top'])
    recommender = model.recommender_3(title, top)
    df = recommender[['title', 'year', 'plot', 'type']]
    return render_template('elements.html', title=title, column_names=df.columns.values, 
    row_data=list(df.values.tolist()), link_column= "title", zip=zip)


if __name__ == "__main__":
    app.run(debug=True)

