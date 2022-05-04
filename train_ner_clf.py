"""
Обучение модели извлечения сущностей (NER) с использованием пакета sklearn_crfsuite

Этапы:
Чтение датасета в формате csv, файл должен содержать колонки - Sentences, POS, Tag
Выделение фичей crf и разбиение на train/test (в файле make_crf_features)
Создание, тренировка модели с параметрами из params и сохранение в файл
Рассчет метрик (f1) на тестовой выборке
"""

import joblib
import os

from crf_model import CrfModel
import make_crf_features
import load_kaggle_dataset


DATA_PATH = 'datasets/ner.csv'
test_size = 0.2
params = {
    'algorithm': 'lbfgs',
    'c1': 0.1,
    'c2': 0.1,
    'max_iterations': 10,
    'all_possible_transitions': True,
    'verbose': True,
}


tuple_data = load_kaggle_dataset.tuple_from_csv(DATA_PATH)
X_train, y_train, X_test, y_test = make_crf_features.get_train_test(tuple_data, test_size=test_size)

model = CrfModel()
model.init_model(params)
model.train_model(X_train, y_train)
params = model.crf.get_params()

if not os.path.exists('output'):
    os.mkdir('output')

model.save_model(f'output/crf_model_c1_{params["c1"]}_c2_{params["c2"]}.pkl')

all_ents_f1, ents_f1 = model.test_model(X_test, y_test)

print(f'F1 for all entities: {all_ents_f1}, \nF1 for real entities: {ents_f1}')
