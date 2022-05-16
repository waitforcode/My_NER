"""
Обучение модели извлечения сущностей (NER) с использованием предобученной модели Bert-base-cased

Этапы:
Чтение датасета в формате csv, файл должен содержать колонки - Sentences, Tag
Токенизация, создание датасета
Создание, тренировка модели с параметрами из params и сохранение в файл
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import transformers
from transformers import BertTokenizerFast
from tqdm import tqdm
import os
import json
from src.data.data_utils import get_word_to_idx, get_tag_to_idx, df_from_csv
from src.data.BertNerDataset import BertNerDataset
from src.models.f1_score import F1Score

# specify GPU
device = torch.device("cuda")
DATA_PATH = 'datasets/train.csv'
config = {
    'MAX_LEN': 512,
    'batch_size': 8,
    'sentence_col': 'Sentence',
    'tag_col': 'Tag',
    'model_name': 'kaggle_bert_base_cased',
    'out_path': 'bert_output',
    'test_size': 0.33
}

df = df_from_csv(DATA_PATH).sample(100)
tag_to_idx = get_tag_to_idx(df['Tag'].values.tolist())
word_to_idx = get_word_to_idx(df['Sentence'].apply(lambda x: x.split(' ')).values.tolist())


train_df, val_df = train_test_split(df, test_size=config['test_size'], random_state=42)

tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
bert_model = transformers.BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=len(tag_to_idx)-1)

train_dataset = BertNerDataset(ner_data=train_df[config['sentence_col']].values,
                               ner_tags=train_df[config['tag_col']].values,
                               tokenizer=tokenizer,
                               tag_to_idx=tag_to_idx,
                               data_type='train')

val_dataset = BertNerDataset(ner_data=val_df[config['sentence_col']].values,
                             ner_tags=val_df[config['tag_col']].values,
                             tokenizer=tokenizer,
                             tag_to_idx=tag_to_idx,
                             data_type='valid')

train_data_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=False)
val_data_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)


def train(data_loader, model, optimizer):
    train_loss = 0
    for idx, dataset in enumerate(tqdm(data_loader, total=len(data_loader))):
        batch_input_ids = dataset['input_ids'].to(device, dtype=torch.long)
        batch_att_mask = dataset['attention_mask'].to(device, dtype=torch.long)
        batch_target = dataset['target'].to(device, dtype=torch.long)

        output = model(batch_input_ids,
                       token_type_ids=None,
                       attention_mask=batch_att_mask,
                       labels=batch_target)

        step_loss = output[0]
        prediction = output[1]

        step_loss.sum().backward()
        optimizer.step()
        train_loss += step_loss
        optimizer.zero_grad()

    return train_loss.sum()


def evaluate(data_loader, model):
    model.eval()
    eval_loss = 0
    predictions = np.array([], dtype=np.int64).reshape(0, config['MAX_LEN'])
    true_labels = np.array([], dtype=np.int64).reshape(0, config['MAX_LEN'])

    with torch.no_grad():
        for idx, dataset in enumerate(tqdm(data_loader, total=len(data_loader))):
            batch_input_ids = dataset['input_ids'].to(device, dtype=torch.long)
            batch_att_mask = dataset['attention_mask'].to(device, dtype=torch.long)
            batch_target = dataset['target'].to(device, dtype=torch.long)

            output = model(batch_input_ids,
                           token_type_ids=None,
                           attention_mask=batch_att_mask,
                           labels=batch_target)

            step_loss = output[0]
            eval_prediction = output[1]

            eval_loss += step_loss

            eval_prediction = np.argmax(eval_prediction.detach().to('cpu').numpy(), axis=2)
            actual = batch_target.to('cpu').numpy()

            predictions = np.concatenate((predictions, eval_prediction), axis=0)
            true_labels = np.concatenate((true_labels, actual), axis=0)

            return eval_loss.sum(), predictions, true_labels


if not os.path.exists('bert_output'):
    os.mkdir('bert_output')

epoch = 3
bert_model.to(device)
optimizer = torch.optim.Adam(bert_model.parameters(), lr=3e-5)
metric = F1Score(average='weighted')

best_eval_loss = 1000000
for i in range(epoch):
    train_loss = train(data_loader=train_data_loader, model=bert_model, optimizer=optimizer)
    eval_loss, eval_predictions, true_labels = evaluate(data_loader=val_data_loader, model=bert_model)
    print(f"Epoch {i} , Train loss: {train_loss}, Eval loss: {eval_loss}")

    if eval_loss < best_eval_loss:
        best_eval_loss = eval_loss

        print("Saving the model")
        torch.save(bert_model.state_dict(), f'{config["out_path"]}/{config["model_name"]}')
