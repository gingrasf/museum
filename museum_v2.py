import pandas as pd
import requests
import json
import numpy as np
import re
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from pymongo import MongoClient


def filter_data(df, visitor_treshold, year):
    df = df[df.nb_visitor_per_year > visitor_treshold]
    return df[df.year == str(year)]


def clean_data(data_table_html):
    """
    It will:
        * Remove title and header rows
        * Rename the columns
        * Convert nb of visitor to numpy int64
    """
    df = pd.read_html(data_table_html)[0]
    df = df.drop([0, 1])  # dropping 1st title row and 2nd header row
    df.rename(
        columns={
            0: "museum",
            1: "city",
            2: "nb_visitor_per_year",
            3: "year"
        },
        inplace=True)
    df.year = df.year.map(lambda x: re.compile(r'(\d+)').match(x).group())
    df.museum = df.museum.map(lambda x: re.compile(
        '^([^[]+)([[]*)').match(x).group(1))
    df.nb_visitor_per_year = pd.to_numeric(
        df.nb_visitor_per_year, errors='coerce').fillna(0).astype(np.int64)
    return df


def fetch_data_from_remote_source_convert_to_data_frame():
    museumsUrl = "http://en.wikipedia.org/w/api.php?action=parse&format=json&page=List_of_most_visited_museums&utf8=1&section=1"

    response = requests.get(museumsUrl)

    data_table_html = response.json()["parse"]["text"]["*"]
    return data_table_html


def get_wikibase_id(page):
    try:
        url = "https://en.wikipedia.org/w/api.php?action=query&prop=pageprops&format=json&redirects&titles=" + page
        response = requests.get(url)
        pageIdKey = next(iter(response.json()["query"]["pages"].keys()))
        return response.json()["query"]["pages"][pageIdKey]["pageprops"]["wikibase_item"]
    except:
        return "NotFound"

def get_wikibase_property_int64(wikibase_id, prop_id, fallback_function=None, *args):
    try:
        url = "https://www.wikidata.org/w/api.php?action=wbgetentities&languages=en&format=json&ids=" + wikibase_id
        response = requests.get(url)
        return np.int64(response.json()["entities"][wikibase_id]["claims"][prop_id][0]["mainsnak"]["datavalue"]["value"]["amount"])
    except:
        if fallback_function is not None:
            return fallback_function(*args)
        else:
            return np.int64(-9999) # Not Found


def get_wikibase_property_list(wikibase_id, prop_id):
    try:
        url = "https://www.wikidata.org/w/api.php?action=wbgetentities&languages=en&format=json&ids=" + wikibase_id
        response = requests.get(url)

        type_ids = []
        for prop in response.json()["entities"][wikibase_id]["claims"][prop_id]:
            type_ids.append(prop["mainsnak"]["datavalue"]["value"]["id"])
        return type_ids
    except:
        return []


def get_wikibase_labels_for_ids(allType):
    try:
        url = "https://www.wikidata.org/w/api.php?action=wbgetentities&languages=en&format=json&props=labels&ids=" + \
            "|".join(allType)
        response = requests.get(url)

        id_label_map = {}
        for key in response.json()["entities"].keys():
            id_label_map[key] = response.json(
            )["entities"][key]["labels"]["en"]["value"]
        return id_label_map
    except:
        return {}


def scrape_population_from_wikipedia(city):
    url = " http://en.wikipedia.org/w/api.php?action=query&prop=revisions&rvprop=content&format=json&redirects&rvsection=0&titles=" + city
    response = requests.get(url)
    pageIdKey = next(iter(response.json()["query"]["pages"].keys()))
    content_text = response.json(
    )["query"]["pages"][pageIdKey]["revisions"][0]["*"]
    pop = re.compile(
        r"population_total = ([\d|,]+)").search(content_text).group(1)
    return np.int64(pop.replace(',', ''))

def scrape_public_transit_from_wikipedia(museum):
    url = " http://en.wikipedia.org/w/api.php?action=query&prop=revisions&rvprop=content&format=json&redirects&rvsection=0&titles=" + museum
    response = requests.get(url)
    pageIdKey = next(iter(response.json()["query"]["pages"].keys()))
    content_text = response.json(
    )["query"]["pages"][pageIdKey]["revisions"][0]["*"]
    publictransit = re.compile(
        r"publictransit\s*=([^\\\n]+)").search(content_text)
    if publictransit is None: return 0
    else: return 1 

def get_most_visited_museums(visitor_treshold = 2000000, year = 2017, force_reload = False, no_mongo = False):
    """
        Retrieve the most visited museums list from wikipedia, clean and filter the data.
    """
    file_name = 'most_visited_museums.csv'

    if not no_mongo:
        mongo_client = MongoClient(host="mongodb://mongo:27017", serverSelectionTimeoutMS=3000)
    else:
        mongo_client = None
    # Use this instead for Local mongo with docker-machine ip
    # mongo_client = MongoClient(host="mongodb://192.168.99.100:27017", serverSelectionTimeoutMS=5000)

    if not force_reload and checkIfAlreadyPersisted(mongo_client, file_name):
        df = readDfFromPersistence(mongo_client, file_name)
        print("Data loaded form persistence store")
        return df
    else:
        print("Fetching data from remote sources")
        data_table_html = fetch_data_from_remote_source_convert_to_data_frame()
        df = clean_data(data_table_html)
        df = filter_data(df, visitor_treshold, year)
        df['city_wikibase_id'] = df.apply(
            lambda row: get_wikibase_id(row.city), axis=1)
        df['museum_wikibase_id'] = df.apply(
            lambda row: get_wikibase_id(row.museum), axis=1)
        # P1082 is wikibase id for population
        df['population'] = df.apply(lambda row: get_wikibase_property_int64(
            row.city_wikibase_id, "P1082", scrape_population_from_wikipedia, row.city ), axis=1)
        # P1436 is the collection size of property
        df['collection_size'] = df.apply(
            lambda row: get_wikibase_property_int64(row.museum_wikibase_id, "P1436"), axis=1)
        # P31 is the instance of property
        df['museums_type_ids'] = df.apply(
            lambda row: get_wikibase_property_list(row.museum_wikibase_id, "P31"), axis=1)            
        allType = set(
            [item for sublist in df['museums_type_ids'].values for item in sublist])
        id_label_map = get_wikibase_labels_for_ids(allType)
        df['museums_type'] = df.apply(lambda row: build_list_of_museums_labels_from_ids(
            row.museums_type_ids, id_label_map), axis=1)            
        df['is_art_museum'] = df.apply(lambda row: does_contains(
            "art museum", row.museums_type), axis=1)
        df['is_science_museum'] = df.apply(lambda row: does_contains(
            "science museum", row.museums_type), axis=1)
        df['is_national_museum'] = df.apply(lambda row: does_contains(
            "national museum", row.museums_type), axis=1)
        df['is_nat_history_museum'] = df.apply(lambda row: does_contains(
            "natural history museum", row.museums_type), axis=1)
        df['nb_of_interesting_type'] = df.apply(lambda row: (row.is_art_museum + row.is_science_museum + row.is_national_museum + row.is_nat_history_museum), axis=1)                        
        df['has_public_transit'] = df.apply(
            lambda row: scrape_public_transit_from_wikipedia(row.museum), axis=1)            
        persist_df(df, mongo_client, file_name)
    return df


def checkIfAlreadyPersisted(mongo_client, file_name):
    csv_file = Path('./' + file_name)
    return checkIfPesistedInMongo(mongo_client) or csv_file.is_file()


def checkIfPesistedInMongo(mongo_client):
    if mongo_client is None:
        return False
    try:
        db = mongo_client['museums']
        # Making sure connection works
        mongo_client.server_info()
        cursor = db['museums'].find()
        return cursor.collection.count_documents({}) > 0
    except:
        return False


def readDfFromPersistence(mongo_client, file_name):
    if checkIfPesistedInMongo(mongo_client):
        print("Reading data fromd mongo")
        db = mongo_client['museums']
        mongo_cursor = db['museums'].find()
        return pd.DataFrame(list(mongo_cursor))
    else:
        print("Reading data fromd csv")
        return pd.read_csv(file_name)


def persist_df(df, mongo_client, file_name=None):
    if mongo_client is not None:
        try:
            print("Saving to mongo")
            db = mongo_client['museums']
            records = json.loads(df.to_json(orient='records'))
            db['museums'].insert(records)
        except:
            print("No mongo instance found")

    if (file_name is not None):
        print("Saving to csv")
        df.to_csv(file_name, index=False)


def does_contains(value, list):
    if value in list:
        return 1
    else:
        return 0


def plot_base_chart(x, y):
    plt.scatter(x, y, c='red', label='Scatter Plot')
    plt.xlabel('City Population')
    plt.ylabel('Museum Visit')
    plt.legend()
    plt.show()


def do_simple_linear_regression(x, y):
    x = x.reshape(len(x), 1)

    # Creating regression model using sklearn
    reg = LinearRegression()

    # Fitting it to X and Y that we defined prevoiusly
    reg = reg.fit(x, y)

    print(f"coefficient (b1) is: {reg.coef_}")
    print(f"constant is (b0) is: {reg.intercept_}")
    print(f"R squared is {reg.score(x, y)}")
    return reg

def create_and_plot_regression_model(x, y):
    reg = do_simple_linear_regression(x, y)
    check_and_compute_p_value(x, y)
    compute_and_print_root_mean_squares_error(x, y, reg.intercept_, reg.coef_)

    y_pred = plot_simple_regression(x, y, reg, 'Simple Regression Line (Complete Data)')

    return y_pred

def plot_simple_regression(x, y, reg, title):
    x = x.reshape(len(x), 1)
    y_pred = reg.predict(x)

    plt.plot(x, y_pred, color='green', label='Regression Line')
    plt.scatter(x, y, c='red', label='Actual Data')
    plt.title(title)
    plt.xlabel('City Population')
    plt.ylabel('Museum Visit per Year')
    plt.legend()
    plt.show()
    return y_pred


def plot_residuals(y, y_pred):
    plt.scatter(y_pred, y_pred - y, color='green', label='Residuals')
    plt.hlines(y=0, xmin=0, xmax=50, color='black')
    plt.title('Residuals (Complete Data)')
    plt.xlabel('Actual Museum Visit per Year')
    plt.ylabel('Residuals')    
    plt.legend()
    plt.show()


def compute_and_print_root_mean_squares_error(x, y, b0, b1):
    rmse = 0
    m = len(x)
    for i in range(m):
        y_pred = b0 + b1 * x[i]
        rmse += (y[i] - y_pred) ** 2
    rmse = np.sqrt(rmse/m)
    print(f"Root Mean Squares Error is {rmse[0]}")


def check_and_compute_p_value(x, y):
    x = sm.add_constant(x)
    model = sm.OLS(y, x)
    reg = model.fit()
    x_pvalue = reg.pvalues[1]
    if (x_pvalue < 0.05):
        print(f"p_value {x_pvalue} is satistically significant")
    else:
        print(f"p_value {x_pvalue} is not satistically significant")


def linear_regression(museums_df, variable):
    city_pop = museums_df[variable].values
    museum_visit = museums_df['nb_visitor_per_year'].values
    plot_base_chart(city_pop, museum_visit)
    museum_visit_prediction = create_and_plot_regression_model(
        city_pop, museum_visit)
    plot_residuals(museum_visit, museum_visit_prediction)

def linear_regression_prediction(museums_df, variable):
    sample = museums_df.sample(frac=0.8, random_state=1)
    test_sample = museums_df.drop(sample.index)

    city_pop = sample[variable].values
    museum_visit = sample['nb_visitor_per_year'].values
    reg = do_simple_linear_regression(city_pop, museum_visit)
    check_and_compute_p_value(city_pop, museum_visit)

    prediction = pd.DataFrame(test_sample[['museum','city','nb_visitor_per_year'] + [variable]])
    to_predict = prediction[variable].values
    to_predict = to_predict.reshape(len(to_predict), 1)
    prediction['predicted_visits'] = reg.predict(to_predict)
    prediction['residuals'] = prediction.apply(lambda row: row.nb_visitor_per_year - row.predicted_visits, axis=1)
    prediction['residuals %'] = prediction.apply(lambda row: row.residuals/row.nb_visitor_per_year*100, axis=1)
    print("Predictions on test sample")
    print(prediction)

def multiple_linear_regression_prediction(museums_df, variable_list):

    sample = museums_df.sample(frac=0.8, random_state=1)
    test_sample = museums_df.drop(sample.index)

    x = sample[variable_list]
    y = sample['nb_visitor_per_year'].values

    reg = LinearRegression()

    # Fitting it to X and Y that we defined prevoiusly
    reg = reg.fit(x, y)

    # with statsmodels
    x = sm.add_constant(x) # adding a constant
    
    model = sm.OLS(y, x).fit()
    
    print_model = model.summary()
    print(print_model)

    prediction = pd.DataFrame(test_sample[['museum','city','nb_visitor_per_year'] + variable_list])
    to_predict = prediction[variable_list].values
    prediction['predicted_visits'] = reg.predict(to_predict)
    prediction['residuals'] = prediction.apply(lambda row: row.nb_visitor_per_year - row.predicted_visits, axis=1)
    prediction['residuals %'] = prediction.apply(lambda row: row.residuals/row.nb_visitor_per_year*100, axis=1)
    print("Predictions on test sample")
    print(prediction)

def complete_multiple_linear_regression(museums_df, variable_list):
    x = museums_df[variable_list]
    y = museums_df['nb_visitor_per_year'].values

    reg = LinearRegression()

    # Fitting it to X and Y that we defined prevoiusly
    reg = reg.fit(x, y)

    print(f"coefficient (b1) is: {reg.coef_}")
    # find the value of b (the constant/intercept) rounded up to 2 dec places
    print(f"constant is (b0) is: {reg.intercept_}")

    # with statsmodels
    x = sm.add_constant(x) # adding a constant
    
    model = sm.OLS(y, x).fit()
    
    print_model = model.summary()
    print(print_model)

def build_list_of_museums_labels_from_ids(ids, id_label_map):
    labels = []
    for id in ids:
        if re.search('museum', id_label_map[id], re.IGNORECASE):
            labels.append(id_label_map[id])
    return labels


def fetch_data(no_mongo=False):
    return get_most_visited_museums(force_reload=False, no_mongo=no_mongo)

def main():
    museums_df = get_most_visited_museums(force_reload=False, no_mongo=True)
    # linear_regression_prediction(museums_df, 'population')
    # multiple_linear_regression_prediction(museums_df, ['population', 'is_art_museum', 'is_national_museum', 'is_science_museum', 'is_nat_history_museum', 'nb_of_interesting_type','has_public_transit'])
    # linear_regression(museums_df, 'population')
    # variable = 'population'
    # city_pop = museums_df[variable].values
    # museum_visit = museums_df['nb_visitor_per_year'].values
    # plot_base_chart(city_pop, museum_visit)
    # museum_visit_prediction = create_and_plot_regression_model(city_pop, museum_visit)

# main()