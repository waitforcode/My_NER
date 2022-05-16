# My_NER
### Описание проекта
Примеры модулей обучения моделей для решения задачи NER. Для примера взят датасет Annotated Corpus for Named Entity Recognition с kaggle. 
####Сущности:
- geo = Geographical Entity
- org = Organization
- per = Person
- gpe = Geopolitical Entity
- tim = Time indicator
- art = Artifact
- eve = Event
- nat = Natural Phenomenon
####Модели:
1. Обучение с использованием sklearn_crfsuite
2. Обучение bidirectional LSTM с эмбеддингами токенов из fasttext
3. Обучение модели на базе предобученной bert-base-cased

####Структура проекта:
```bash
.
├── train_crfner_clf.py
├── train_bilstm_clf.py
├── train_bert_clf.py
├── README.md
├── datasets
│   ├── ner.csv
│   ├── test.csv
│   ├── train.csv
├── Notebooks
│   ├── Test_kaggle_dataset.ipynb
│   ├── BERT_ner.ipynb
├── src
│   ├── data
│   ├── ├── BertNerDataset.py
│   ├── ├── Embedding.py
│   ├── ├── NERDataloader.py
│   ├── ├── data_utils.py
│   ├── models
│   ├── ├── BiLSTMModel.py
│   ├── ├── crf_model.py
│   ├── ├── f1_score.py
```
- `train_ner_clf.py` - запуск обучения CRF модели
- `train_bilstm_clf.py` - запуск обучения BiLSTM модели
- `train_bert_clf` - запуск обучения модели BERT
- `README.md` - общее описание проекта
- `datasets`
  - `ner.csv` - исходный датасет
  - `test.csv` - тестовая выборка
  - `train.csv` - обучающая выборка
- `Notebooks`
  - `Test_kaggle_dataset.ipynb` - загрузка датасета и тест моделей
  - `BERT_ner.ipynb` - тест модели BERT
- `src`
  - `data` - модули работы с данными
    - `BertNerDataset.py` - класс обертка датасета для модели BERT
    - `Embedding.py` - класс для получения эмбеддингов fasttext
    - `NERDataloader.py` - класс даталоадера для BiLSTM модели
    - `data_utils.py` - загрузка, обработка данных, получение фичей
  - `models` - классы моделей
    - `BiLSTMModel.py`
    - `crf_model.py`
    - `f1_score.py`
