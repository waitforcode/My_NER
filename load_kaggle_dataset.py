import pandas as pd
from ast import literal_eval


def tuple_from_df(df):
    data = []
    for _, row in df.iterrows():
        tokens = [token for token in zip(row['Sentence'].split(), row['POS'], row['Tag'])]
        data.append(tokens)
    return data


def df_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    df['POS'] = df['POS'].apply(literal_eval)
    df['Tag'] = df['Tag'].apply(literal_eval)
    return df


def tuple_from_csv(csv_path):
    df = df_from_csv(csv_path)
    data = []
    for _, row in df.iterrows():
        tokens = [token for token in zip(row['Sentence'].split(), row['POS'], row['Tag'])]
        data.append(tokens)
    del df
    return data
