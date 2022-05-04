from sklearn_crfsuite import CRF
from sklearn_crfsuite import metrics
import joblib
import os


class CrfModel:
    """
    Обертка над моделью CRF из sklearn_crfsuite, позволяющая обучать, тестировать, сохранять/загружать модель
    и делать predict
    """

    def __init__(self):
        self.crf = None

    def init_model(self, params=None):
        if params:
            self.crf = CRF(**params)
        else:
            self.crf = CRF(
                algorithm='lbfgs',
                c1=0.1,
                c2=0.1,
                max_iterations=10,
                all_possible_transitions=True,
                verbose=True,
            )

    def load_model(self, model_path):
        self.crf = joblib.load(model_path)

    def save_model(self, model_path):
        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(model_path))
        joblib.dump(self.crf, model_path)

    def train_model(self, X_train, y_train):
        self.crf.fit(X_train, y_train)

    def predict(self, X_test):
        return self.crf.predict(X_test)

    def extract_entities(self, features):
        pass

    def test_model(self, X_test, y_test):
        all_ents = list(self.crf.classes_)
        ents = [ent for ent in all_ents if ent != 'O']

        y_pred = self.predict(X_test)

        all_ents_f1 = metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=all_ents)
        ents_f1 = metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=ents)

        return all_ents_f1, ents_f1


