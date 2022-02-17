import torch
import torch.nn as nn

from Layer import BertEmbedding, BertEncoder, BertPooler

class BertModel(nn.Module):
    def __init__(self, hidden_size, vocab_size, d_ff, n_layers=2, n_heads=2, max_position_embeddings=512, type_vocab_size=2, pad_idx=0, layer_norm_eps=1e-12, dropout_p=0.1):
        super().__init__()
        self.embedding = BertEmbedding(vocab_size, hidden_size, max_position_embeddings, type_vocab_size, pad_idx, layer_norm_eps, dropout_p)
        self.encoder = BertEncoder(hidden_size, d_ff, n_layers, n_heads, layer_norm_eps, dropout_p)
        self.pooler = BertPooler(hidden_size)

    def forward(self, input_ids, attention_mask, token_type_ids):
        """
        input_ids,attention_mask,token_type_ids = (batch_size, seq_len)
        return: seq_out = (batch_size, seq_len, hidden_size), pooled_out = (batch_size, hidden_size)
        pooled_out: 각 배치 내 cls 토큰의 의미 가짐
        """
        embedding_out = self.embedding(input_ids, token_type_ids)
        seq_out = self.encoder(embedding_out, attention_mask)
        pooled_out = self.pooler(seq_out)

        return (seq_out, pooled_out)
