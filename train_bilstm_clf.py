import torch
from torch import nn
import numpy as np
import fasttext.util
import os

from sklearn.model_selection import train_test_split
from src.models.BiLSTMModel import BiLSTMCRFAtt
from src.models.f1_score import F1Score
from src.data.Embedding import FTEmbedding
from src.data.NERDataloader import NERDataloader
from src.data.data_utils import get_word_to_idx, get_tag_to_idx, df_from_csv


DATA_PATH = 'datasets/train.csv'
out_path = 'bilstm_models'
test_size = 0.33

df = df_from_csv(DATA_PATH)
tag_to_idx = get_tag_to_idx(df['Tag'].values.tolist())
word_to_idx = get_word_to_idx(df['Sentence'].apply(lambda x: x.split(' ')).values.tolist())

train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

train = NERDataloader(train_df['Sentence'].values.tolist(),
                      train_df['Tag'].values.tolist(),
                      word_to_idx,
                      tag_to_idx,
                      batch_size=128)
test = NERDataloader(test_df['Sentence'].values.tolist(),
                     test_df['Tag'].values.tolist(),
                     word_to_idx,
                     tag_to_idx,
                     batch_size=128)


train_loader = train.get_loader()
valid_loader = test.get_loader()

fasttext.util.download_model('en', if_exists='ignore')
ft = fasttext.load_model('cc.en.300.bin')
embedding_matrix, nb_words = FTEmbedding(ft).build_matrix(word_to_idx)
embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)
embedding = nn.Embedding.from_pretrained(embedding_matrix)
embedding.weight.requires_grad = False

model = BiLSTMCRFAtt(tag_to_idx, 100, 32, 0.2)
metric = F1Score(average='weighted')
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.cuda()
embedding.cuda()

train_f1 = []
valid_f1 = []
max_eval_f1 = 0
for epoch in range(20):
    model.train()
    epoch_train_f1 = []
    for i, batch in enumerate(train_loader):
        tokens, lens, labels = batch
        tokens, lens, labels = tokens.cuda(), lens.cuda(), labels.cuda()
        optimizer.zero_grad()

        tag_seq, loss = model(embedding(tokens), lens, labels)

        loss.backward()
        optimizer.step()

        labels = labels.flatten()
        labels = labels[labels != tag_to_idx['PAD']]
        f1_score = metric(tag_seq, labels).item()
        epoch_train_f1.append(f1_score)

    model.eval()
    epoch_valid_f1 = []
    with torch.no_grad():
        for i, batch in enumerate(valid_loader):
            tokens, lens, labels = batch
            tokens, lens, labels = tokens.cuda(), lens.cuda(), labels.cuda()

            tag_seq, loss = model(embedding(tokens), lens, labels)
            labels = labels.flatten()
            labels = labels[labels != tag_to_idx['PAD']]
            f1_score = metric(tag_seq, labels).item()
            epoch_valid_f1.append(f1_score)

    mean_epoch_train_f1 = np.mean(epoch_train_f1)
    mean_epoch_valid_f1 = np.mean(epoch_valid_f1)

    if mean_epoch_valid_f1 > max_eval_f1:
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        torch.save(model.state_dict(), f'{out_path}/model_epoch_{epoch}_f1_{mean_epoch_valid_f1}.pth')
        max_eval_f1 = mean_epoch_valid_f1

    train_f1.append(mean_epoch_train_f1)
    valid_f1.append(mean_epoch_valid_f1)
    print(f'{epoch=}. {mean_epoch_train_f1=:0.4f}. {mean_epoch_valid_f1=:0.4f}.')
