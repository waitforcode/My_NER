"""
Методы для получения фичей и разбиения на трейн и тест
"""

from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict, Union, Any


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
