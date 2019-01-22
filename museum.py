import pandas as pd
import requests
import json
import numpy as np
import re
import matplotlib.pyplot as plt
from pathlib import Path
from SPARQLWrapper import SPARQLWrapper, JSON



def get_sparql_results(endpoint_url, query):
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()



def get_city_data(cities, force_reload=False):
    file_name = "cities.json"

    json_file = Path('./' + file_name)
    if not force_reload and json_file.is_file():
        # with open(file_name) as f:
            # json_data = json.load(file_name)
            with open(file_name, 'r') as data_file:
             json_str= data_file.read()
             json_data = json.loads(json.loads(json_str))
    else:
        json_data = get_cities_from_remote_source(file_name)
    df = convert_cities_to_dataframe(json_data, cities)
    # csv_file_name = "cities.csv"
    # df.to_csv(csv_file_name, index=False)   
    return df

def convert_cities_to_dataframe(json_data, cities):
    data = []
    for json_item in json_data:
        city = to_city(json_item)
        if (city["city"] in cities):
            data.append(city)

    df = pd.DataFrame(data)
    df.population = pd.to_numeric(df.population, errors='coerce').fillna(0).astype(np.int64)
    return df


def to_city(city):
    json_city = {}
    json_city["city"] = city["cityLabel"]["value"]
    json_city["population"] = city["population"]["value"]
    return json_city
    # return json.dumps(json_city)

def get_cities_from_remote_source(file_name):
    endpoint_url = "https://query.wikidata.org/sparql"
    query = """
    SELECT DISTINCT ?cityLabel ?population ?gps
    WHERE
    {
    ?city wdt:P31/wdt:P279* wd:Q515 .
    ?city wdt:P1082 ?population .
    ?city wdt:P625 ?gps .
    SERVICE wikibase:label {
        bd:serviceParam wikibase:language "en" .
    }
    }
    ORDER BY DESC(?population)"""
    results = get_sparql_results(endpoint_url, query)

    json_data = json.dumps(results["results"]["bindings"])
    with open(file_name, 'w') as outfile:
        json.dump(json_data, outfile)
    return json_data

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
    df = df.drop([0, 1]) # dropping 1st title row and 2nd header row
    df.rename(
    columns={
        0 : "museum",
        1 : "city",
        2 : "nb_visitor_per_year",
        3 : "year"    
    },
    inplace=True)
    df.year= df.year.map(lambda x: re.compile('(\d+)').match(x).group())
    df.nb_visitor_per_year = pd.to_numeric(df.nb_visitor_per_year, errors='coerce').fillna(0).astype(np.int64)
    return df

def fetch_data_from_remote_source_convert_to_data_frame():
    museumsUrl = "http://en.wikipedia.org/w/api.php?action=parse&format=json&page=List_of_most_visited_museums&utf8=1&section=1"

    response = requests.get(museumsUrl)

    data_table_html = response.json()["parse"]["text"]["*"]
    return data_table_html

def get_wikibase_id(city):
    url = "https://en.wikipedia.org/w/api.php?action=query&prop=pageprops&format=json&titles=" + city    
    response = requests.get(url)
    pageIdKey = next(iter(response.json()["query"]["pages"].keys()))
    return response.json()["query"]["pages"][pageIdKey]["pageprops"]["wikibase_item"]


def get_wikibase_property_int64(wikibase_id, prop_id, city):
    try:
        url = "https://www.wikidata.org/w/api.php?action=wbgetentities&languages=en&format=json&ids=" + wikibase_id    
        response = requests.get(url)
        return np.int64(response.json()["entities"][wikibase_id]["claims"][prop_id][0]["mainsnak"]["datavalue"]["value"]["amount"])
    except:
        return scrape_from_wikipedia(city)

def scrape_from_wikipedia(city):
        url = " http://en.wikipedia.org/w/api.php?action=query&prop=revisions&rvprop=content&format=json&rvsection=0&titles=" + city    
        response = requests.get(url)
        pageIdKey = next(iter(response.json()["query"]["pages"].keys()))
        content_text = response.json()["query"]["pages"][pageIdKey]["revisions"][0]["*"]
        pop = re.compile("population_total = ([\d|,]+)").search(content_text).group(1)
        return np.int64(pop.replace(',', ''))

def get_most_visited_museums(visitor_treshold = 2000000, year = 2017, force_reload = False):
    """
        Retrieve the most visited museums list from wikipedia, clean and filter the data.
    """
    file_name = 'most_visited_museums.csv'
    csv_file = Path('./' + file_name)
    if not force_reload and csv_file.is_file():
        df = pd.read_csv(file_name)
        return df
    else:     
        data_table_html = fetch_data_from_remote_source_convert_to_data_frame()
        df = clean_data(data_table_html)
        df = filter_data(df, visitor_treshold, year)
        df['wikibase_id'] = df.apply (lambda row: get_wikibase_id (row.city),axis=1)
        df['population'] = df.apply (lambda row: get_wikibase_property_int64 (row.wikibase_id, "P1082", row.city),axis=1)
        df.to_csv(file_name, index=False)
    return df

def estimate_coefficients(x ,y):
    n = np.size(x)

    mean_x, mean_y = np.mean(x), np.mean(y)

    SS_xy = np.sum(y*x - n*mean_y*mean_x)
    SS_xx = np.sum(x*x - n*mean_x*mean_x)

    b_1 = SS_xy / SS_xx
    b_0 = mean_y - b_1-mean_x

    return (b_0, b_1)

def plot_regression_line(x, y, b):
    plt.scatter(x, y, color = "m", marker = "o", s = 30)

    y_pred = b[0] + b[1]*x
    plt.plot(x, y_pred, color = "g")

    plt.xlabel('City Population')    
    plt.ylabel('Museum Visit')

    plt.show()


def main():
    museums_df = get_most_visited_museums(force_reload=False)
    city_pop = np.array(list(museums_df['population']))
    museum_visit = np.array(list(museums_df['nb_visitor_per_year']))

    b = estimate_coefficients(city_pop, museum_visit)

    print("Estimated coefficients:\nb_0 = {} \nb_1 ={}".format(b[0], b[1]))

    plot_regression_line(city_pop, museum_visit, b)

main()    
# cities_df = get_city_data(list(museums_df.city.unique()))

# cities_df.drop_duplicates(subset="city" ,inplace=True)
# print(get_wikibase_id("Vatican City"))
# print(get_wikibase_property_int64("Q90", "P1082"))
# print(get_wikibase_property_int64("Q237", "P1082"))
# citiesNoPop = list(museums_df[museums_df.population == -1].city)
# for c in citiesNoPop:
    # print(scrape_from_wikipedia(c))