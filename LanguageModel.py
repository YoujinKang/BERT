import torch 
import torch.nn as nn

class BertMLMHead(nn.Module):
    def __init__(self, hidden_size, vocab_size, layer_norm_eps=1e-12):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = torch.nn.functional.gelu
        self.layer_norm = nn.LayerNorm(hidden_size, layer_norm_eps)

        self.decoder = nn.Linear(hidden_size, vocab_size)
        
        self._init_weight()

    def _init_weight(self):
        nn.init.kaiming_uniform_(self.dense.weight, mode='fan_out', nonlinearity='relu')
        # self.dense.weight.data.uniform_(mean=0.0, std=0.02)
        self.dense.bias.data.zero_()
        self.layer_norm.bias.data.zero_()
        self.layer_norm.weight.data.fill_(1.0)
        nn.init.kaiming_uniform_(self.decoder.weight, mode='fan_out', nonlinearity='relu')
        # self.decoder.weight.data.uniform_(mean=0.0, std=0.02)
        self.decoder.bias.data.zero_()

    def forward(self, h_states):
        """
        h_states = (batch_size, seq_len, hidden_size)
        return = (batch_size, seq_len, vocab_size)
        """
        h_states = self.layer_norm(self.activation(self.dense(h_states)))
        return self.decoder(h_states)  


class BertNSPHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.seq_relation = nn.Linear(hidden_size, 2)
        
        self._init_weight()

    def _init_weight(self):
        nn.init.kaiming_uniform_(self.seq_relation.weight, mode='fan_out', nonlinearity='relu')
        # self.seq_relation.weight.data.uniform_(mean=0.0, std=0.02)
        self.seq_relation.bias.data.zero_()

    def forward(self, pooled_output):
        """
        pooled_output = (batch_size, hidden_size)
        """
        return self.seq_relation(pooled_output)


class BertPreTrainingHead(nn.Module):
    def __init__(self, model, hidden_size, vocab_size, layer_norm_eps=1e-12):
        super().__init__()
        self.model = model
        self.mlm = BertMLMHead(hidden_size, vocab_size, layer_norm_eps)
        self.nsp = BertNSPHead(hidden_size)

    def forward(self, input_ids, attention_mask, token_type_ids):
        seq_out, pooled_out = self.model(input_ids, attention_mask, token_type_ids)
        return self.mlm(seq_out), self.nsp(pooled_out)



