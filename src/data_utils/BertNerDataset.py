import torch
from torch.utils.data import Dataset
from keras.preprocessing.sequence import pad_sequences
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
import numpy as np
from typing import List, Dict, Tuple, Union


def tokenize_and_align_labels(sentence, tags, tokenizer, tag2idx):
    tokenized = tokenizer(sentence.split(' '), truncation=True,
                          is_split_into_words=True)

    # align tokens and labels
    labels = []
    word_ids = tokenized.word_ids()
    prev_word_id = None
    for word_id in word_ids:
        if word_id is None:
            labels.append(-100)
        elif word_id != prev_word_id:
            labels.append(tag2idx[tags[word_id]])
        else:
            labels.append(-100)
        prev_word_id = word_id

    tokenized["labels"] = labels
    return tokenized


class BertNerDataset(Dataset):
    def __init__(self, ner_data, ner_tags, tokenizer: PreTrainedTokenizerBase, tag_to_idx,
                 data_type='test'):
        self.ner_data = ner_data
        self.ner_tags = ner_tags
        self.tokenizer = tokenizer
        self.tag_to_idx = tag_to_idx
        self.max_lenght = 512
        self.data_type = data_type

    def __len__(self):
        return len(self.ner_data)

    def __getitem__(self, item):
        tokenized = tokenize_and_align_labels(self.ner_data[item], self.ner_tags[item], self.tokenizer, self.tag_to_idx)
        input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']
        labels = tokenized['labels']

        # pad seq
        pad_len = self.max_lenght - len(input_ids)
        input_ids = input_ids + [2] * pad_len
        attention_mask = attention_mask + [0] * pad_len

        if self.data_type != 'test':
            labels = labels + self.tag_to_idx['PAD'] * pad_len
        else:
            labels = 1

        return {'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                'target': torch.tensor(labels, dtype=torch.long)
                }
