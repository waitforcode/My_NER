import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn_crfsuite import scorers
import joblib
import os

from crf_model import CrfModel
import make_crf_features
import load_kaggle_dataset


DATA_PATH = 'datasets/ner.csv'


tuple_data = load_kaggle_dataset.tuple_from_csv(DATA_PATH)
X_train, y_train, X_test, y_test = make_crf_features.train_test(tuple_data)
params = {
    'algorithm': 'lbfgs',
    'c1': 0.1,
    'c2': 0.1,
    'max_iterations': 10,
    'all_possible_transitions': True,
    'verbose': True,
}
model = CrfModel()
model.init_model(params)
model.train_model(X_train, y_train)
params = model.crf.get_params()

if not os.path.exists('output'):
    os.mkdir('output')

joblib.dump(model.crf, f'output/crf_model_c1_{params["c1"]}_c2_{params["c2"]}.pkl')

all_ents_f1, ents_f1 = model.test_model(X_test, y_test)

print(f'F1 for all entities: {all_ents_f1}, \nF1 for real entities: {ents_f1}')
