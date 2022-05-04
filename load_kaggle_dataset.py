"""
Функции для загрузки и преобразования датасета

Формат входных данных: csv файл с колонками:
Sentences: предложения с разделителями пробелами между каждым токеном (словом, знаком...)
POS: список меток частей речи, выровненный по токенам в предожении
Tag: список меток сущностей, выровненный по токенам в предожении
"""

import pandas as pd
from ast import literal_eval
from typing import List, Tuple


def df_from_csv(csv_path: str) -> pd.DataFrame:
    """
    Чтение данных в DataFrame.
    Из-за того, что списки POS и Tag читаются, как строки, нужно бреобразовать их обратно

    :param csv_path: путь к csv файлу
    :return: pd.DataFrame
    """
    df = pd.read_csv(csv_path)
    df['POS'] = df['POS'].apply(literal_eval)
    df['Tag'] = df['Tag'].apply(literal_eval)
    return df


def tuple_from_csv(csv_path: str) -> List[List[Tuple[str, str, str]]]:
    """
    Чтение данных и преобразование их в матрицу вида:
    [[
        (word1, POS, Tag),
        (word2, POS, Tag),
        ...
    ],...]

    :param csv_path: путь к csv файлу
    """

    df = df_from_csv(csv_path)
    data = []
    for _, row in df.iterrows():
        tokens = [token for token in zip(row['Sentence'].split(), row['POS'], row['Tag'])]
        data.append(tokens)
    del df
    return data
