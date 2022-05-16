import fasttext.util
from typing import Dict, Tuple

import numpy as np


class FTEmbedding:
    def __init__(self, model):
        self.ft = model
        fasttext.util.reduce_model(self.ft, 100)

    def build_matrix(self, word_dict: Dict, max_features: int = 100000, embed_size: int = 100) \
            -> Tuple[np.array, int]:
        """
        Create embedding matrix

        Args:
            word_dict: tokenizer
            embedding_index: Fasttext embeddings
            max_features: max features to use
            embed_size: size of embeddings

        Returns:
            embedding matrix, number of words and the list of not found words
        """

        embedding_index = self.ft
        nb_words = min(max_features, len(word_dict))
        embedding_matrix = np.zeros((nb_words, embed_size))

        for word, i in word_dict.items():
            embedding_matrix[i] = embedding_index[word]
        return embedding_matrix, nb_words
