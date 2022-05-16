"""
Обучение модели извлечения сущностей (NER) с использованием пакета sklearn_crfsuite

Этапы:
Чтение датасета в формате csv, файл должен содержать колонки - Sentences, POS, Tag
Выделение фичей crf и разбиение на train/test (в файле make_crf_features)
Создание, тренировка модели с параметрами из params и сохранение в файл
Рассчет метрик (f1) на тестовой выборке
"""

import os

from src.models.crf_model import CrfModel
from src.data.data_utils import get_train_test, tuple_from_csv


DATA_PATH = 'datasets/train.csv'
out_path = 'crf_models'
test_size = 0.2
params = {
    'algorithm': 'lbfgs',
    'c1': 0.1,
    'c2': 0.1,
    'max_iterations': 10,
    'all_possible_transitions': True,
    'verbose': True,
}


tuple_data = tuple_from_csv(DATA_PATH)
X_train, y_train, X_test, y_test = get_train_test(tuple_data, test_size=test_size)

model = CrfModel()
model.init_model(params)
model.train_model(X_train, y_train)
params = model.crf.get_params()
#
# if not os.path.exists(out_path):
#     os.mkdir(out_path)

model.save_model(f'{out_path}/crf_model_c1_{params["c1"]}_c2_{params["c2"]}.pkl')

all_ents_f1, ents_f1 = model.test_model(X_test, y_test)

print(f'F1 for all entities: {all_ents_f1}, \nF1 for real entities: {ents_f1}')
