import pandas as pd

def get_basic_info(data):
    return data.head(), data.info(), data.describe()

def top_nationalities(data):
    return data['nationality'].value_counts().head(10)

def player_counts_by_country(data):
    return data['nationality'].value_counts()
