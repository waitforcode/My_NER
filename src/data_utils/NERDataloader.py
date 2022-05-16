import torch
from torch.utils.data import Dataset
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from typing import List, Dict, Tuple, Union


class NerDataset(Dataset):
    def __init__(self, ner_data: List, ner_tags: List, word_to_idx: Dict, tag_to_idx: Dict, **kwarg: Dict):
        self.ner_data = ner_data
        self.ner_tags = ner_tags
        self.word_to_idx = word_to_idx
        self.tag_to_idx = tag_to_idx

    def __len__(self) -> int:
        return len(self.ner_data)

    def __getitem__(self, idx: int) -> Tuple[np.array, int, np.array]:
        tokens = [self.word_to_idx[w] if w in self.word_to_idx else self.word_to_idx['<unk>'] for w in
                  self.ner_data[idx]]

        labels = [self.tag_to_idx[t] for t in self.ner_tags[idx]]

        return np.array(tokens), len(tokens), np.array(labels)


class Collator:
    def __init__(self, test=False, percentile=100, pad_value=0):
        self.test = test
        self.percentile = percentile
        self.pad_value = pad_value

    def __call__(self, batch):
        tokens, lens, labels = zip(*batch)
        lens = np.array(lens)

        max_len = min(int(np.percentile(lens, self.percentile)), 100)

        tokens = torch.tensor(
            pad_sequences(tokens, maxlen=max_len, padding='post', value=self.pad_value), dtype=torch.long
        )
        lens = torch.tensor([min(i, max_len) for i in lens], dtype=torch.long)
        labels = torch.tensor(
            pad_sequences(labels, maxlen=max_len, padding='post', value=self.pad_value), dtype=torch.long
        )

        return tokens, lens, labels


class NERDataloader:
    def __init__(self, X, y, word_to_idx: Dict, tag_to_idx: Dict, batch_size: int):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.dataset = NerDataset(self.X, self.y, word_to_idx, tag_to_idx)
        self.pad_value = tag_to_idx['PAD']

    def get_loader(self):
        pad_value = self.pad_value
        collator = Collator(percentile=100, pad_value=pad_value)
        loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=0,
            collate_fn=collator,
            shuffle=False
        )
        return loader
