import pandas as pd
from ast import literal_eval
from collections import Counter
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict, Union, Any


"""
Методы работы с данными, получения фичей и разбиения на трейн и тест
"""

def get_word_to_idx(words_by_sentences: List[List[str]],
                    min_words: Union[int, float] = 0.0,
                    max_words: Union[int, float] = 1.0) -> Dict[str, int]:
    """
    Получение словаря для преобразования слов в индексы

    :param words_by_sentences: список слов по предложениям из входных данных
    :param min_words: минимальная частота слова
    :param max_words: максимальная частота слова
    """

    word_count = Counter([i for j in words_by_sentences for i in j]).most_common()

    max_count = word_count[0][1]
    if isinstance(min_words, float):
        min_words = max_count * min_words
    if isinstance(max_words, float):
        max_words = max_count * max_words

    all_words = [w[0] for w in word_count if max_words >= w[1] >= min_words]

    all_words = ['<pad>', '<unk>'] + all_words

    word_to_idx = {k: v for k, v in zip(all_words, range(0, len(all_words)))}
    return word_to_idx


def get_tag_to_idx(tag_list_by_sentences: List[List[str]]) -> Dict[str, int]:
    """
    Получение словаря для преобразования лейблов сущностей в индексы

    :param tag_list_by_sentences: список сущностей по предложениям из входных данных
    """

    all_tags = sorted(list({t for tags in tag_list_by_sentences for t in tags}))

    tag2idx = {}
    for idx, entity in enumerate(all_tags):
        tag2idx[f'{entity}'] = idx

    tag2idx['PAD'] = len(tag2idx)
    return tag2idx


def df_from_csv(csv_path: str) -> pd.DataFrame:
    """
    Чтение данных в DataFrame.
    Из-за того, что списки POS и Tag читаются, как строки, нужно преобразовать их в list

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


def get_word_features(words: List[Tuple[str, str, str]], i: int) -> Dict[str, Union[float, str, bool]]:
    """
    Метод для извлечения фичей для одного слова

    :param words: список слов предложения в виде
        [
            (word1, POS, Tag),
            (word2, POS, Tag),
            ...
        ]
    :param i: индекс слова
    """

    word = words[i][0]
    postag = words[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
    }
    if i > 0:
        word1 = words[i - 1][0]
        postag1 = words[i - 1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
        })
    else:
        features['BOS'] = True

    if i < len(words) - 1:
        word1 = words[i + 1][0]
        postag1 = words[i + 1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
        })
    else:
        features['EOS'] = True

    return features


def get_sent_features(sent: List[Tuple[str, str, str]]) -> List[Dict[str, Union[float, str, bool]]]:
    return [get_word_features(sent, i) for i in range(len(sent))]


def get_sent_labels(sent: List[Tuple[str, str, str]]) -> List[str]:
    return [label for token, postag, label in sent]


def get_sent_tokens(sent: List[Tuple[str, str, str]]) -> List[str]:
    return [token for token, postag, label in sent]


def get_train_test(data: List[List[Tuple[str, str, str]]], test_size: float = 0.25)\
        -> Tuple[List[List[Any]], List[List[str]], List[List[Any]], List[List[str]]]:
    """
    Разбиение на train/test и формирование X_train, y_train | X_test, y_test - векторов фичей и таргетов

    :param data: входные данные (список предложений) в виде:
    [[
            (word1, POS, Tag),
            (word2, POS, Tag),
            ...
    ],...]
    :param test_size: параметр разбиения - размер тестовой выборки
    """

    train_data, test_data = train_test_split(data, test_size=test_size)
    X_train = [get_sent_features(s) for s in train_data]
    y_train = [get_sent_labels(s) for s in train_data]
    X_test = [get_sent_features(s) for s in test_data]
    y_test = [get_sent_labels(s) for s in test_data]

    return X_train, y_train, X_test, y_test
