import torch
from torch import nn
from torchcrf import CRF
from typing import Tuple, Dict, Any


class SpatialDropout(nn.Module):
    """
    Spatial Dropout drops a certain percentage of dimensions from each word vector in the training sample
    implementation: https://discuss.pytorch.org/t/spatial-dropout-in-pytorch/21400
    explanation: https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/76883
    """

    def __init__(self, p: float):
        super(SpatialDropout, self).__init__()
        self.spatial_dropout = nn.Dropout2d(p=p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)  # convert to [batch, channels, time]
        x = self.spatial_dropout(x)
        x = x.permute(0, 2, 1)  # back to [batch, time, channels]
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    torch.nn.MultiHeadAttention wrapper to unify interface with other Attention classes
    Implementation of Dot-product Attention
    paper: https://arxiv.org/abs/1706.03762
    Time complexity: O(n^2)
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float, **kwargs: Dict[str, Any]):
        super(MultiHeadSelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, **kwargs)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.BoolTensor = None) -> torch.Tensor:
        x = x.transpose(0, 1)
        attn = self.attention(query=x, key=x, value=x, key_padding_mask=key_padding_mask)[0]
        attn = attn.transpose(0, 1)
        return attn


class LayerNorm(nn.Module):
    """
    Layer Normalization
    paper: https://arxiv.org/abs/1607.06450
    """

    def __init__(self, normalized_shape: int):
        super(LayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_norm(x)


class BiLSTMCRFAtt(nn.Module):
    """
    New model without nn.Embedding layer
    """

    def __init__(self, tag_to_idx: Dict, embeddings_dim: int = 100, hidden_dim: int = 4, spatial_dropout: float = 0.2):
        super().__init__()
        self.embedding_dim = embeddings_dim
        self.hidden_dim = hidden_dim
        self.tag_to_idx = tag_to_idx
        self.tagset_size = len(tag_to_idx.values())
        self.crf = CRF(self.tagset_size, batch_first=True)
        self.embedding_dropout = SpatialDropout(spatial_dropout)

        self.lstm = nn.LSTM(
            embeddings_dim, hidden_dim // 2, num_layers=2, bidirectional=True, batch_first=True, dropout=0.25
        )
        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, hidden_dim // 2)
        self.hidden2tag2 = nn.Linear(hidden_dim // 2, self.tagset_size)
        self.rnn_layer_norm = LayerNorm(hidden_dim)
        self.att = MultiHeadSelfAttention(embed_dim=hidden_dim, num_heads=2, dropout=0.25)

    def _get_lstm_features(self, embeds: torch.Tensor, lens: torch.Tensor, mask: bool) -> torch.Tensor:
        """
        LSTM forward

        Args:
            embeds: batch with embeddings
            lens: lengths of sequences
        """
        embeds = self.embedding_dropout(embeds)
        packed_embeds = torch.nn.utils.rnn.pack_padded_sequence(
            embeds, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        lstm_out, self.hidden = self.lstm(packed_embeds)
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        lstm_out = self.rnn_layer_norm(lstm_out)
        lstm_out = self.att(lstm_out, key_padding_mask=mask)
        lstm_feats = self.hidden2tag2(self.hidden2tag(lstm_out.reshape(embeds.shape[0], -1, self.hidden_dim)))
        return lstm_feats

    def forward(
        self, embeds: torch.Tensor, lens: torch.Tensor, tags: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward

        Args:
            embeds: batch with embeddings
            lens: lengths of sequences
            tags: list of tags (optional)
        """
        mask1 = tags == self.tag_to_idx['PAD']
        lstm_feats = self._get_lstm_features(embeds, lens, mask1)

        if tags is not None:
            mask = tags != self.tag_to_idx['PAD']
            loss: torch.Tensor = self.crf(lstm_feats, tags, mask=mask)
            tag_seq = self.crf.decode(emissions=lstm_feats, mask=torch.tensor(mask))  # type: ignore

        else:
            loss = torch.tensor(0)
            tag_seq = self.crf.decode(lstm_feats)

        pred: torch.Tensor = torch.tensor([i for j in tag_seq for i in j]).type_as(embeds)
        return pred, -loss
