import torch
import torch.nn as nn

class BertEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position_embeddings=512, type_vocab_size=2, pad_idx=0, layer_norm_eps=1e-12, dropout_p=0.1):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_idx)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size+1, hidden_size, padding_idx=2)

        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout_p)

        self._init_weight(pad_idx)

        # 파라미터 업데이트 하지 않도록 설정
        self.register_buffer("position_ids", torch.arange(max_position_embeddings).expand((1, -1)))
        self.register_buffer("token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False, )

    def _init_weight(self, pad_idx=0):
        nn.init.normal_(self.word_embeddings.weight,mean=0.0, std=0.02)
        self.word_embeddings.weight.data[pad_idx].zero_()
        nn.init.normal_(self.position_embeddings.weight,mean=0.0, std=0.02)
        self.position_embeddings.weight.data[pad_idx].zero_()
        nn.init.normal_(self.token_type_embeddings.weight,mean=0.0, std=0.02)
        self.token_type_embeddings.weight.data[pad_idx].zero_()
        self.layer_norm.weight.data.fill_(1.0)
        self.layer_norm.bias.data.zero_()

    def forward(self, input_ids, token_type_ids):
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        assert seq_len == 512, f"seq_len: {seq_len}"

        input_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)

        embeddings = input_embeds + position_embeds + token_type_embeds
        return self.dropout(self.layer_norm(embeddings))   


class BertSelfAttention(nn.Module):
    def __init__(self, hidden_size, n_heads=2, dropout_p=0.1, is_decoder=False):
        super().__init__()
        assert hidden_size % n_heads == 0, "hidden_size must be a multiple of n_heads"
        self.n_heads = n_heads
        self.head_size = hidden_size // n_heads
        self.all_head_size = self.head_size * n_heads
        self.is_decoder = is_decoder

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(dropout_p)

        self._init_weight()

    def _init_weight(self):
        nn.init.normal_(self.query.weight,mean=0.0, std=0.02)
        self.query.bias.data.zero_()
        nn.init.normal_(self.key.weight,mean=0.0, std=0.02)
        self.key.bias.data.zero_()
        nn.init.normal_(self.value.weight,mean=0.0, std=0.02)
        self.value.bias.data.zero_()   
    
    def transpose_for_scores(self, x):
        # x = (batch_size, seq_len, hidden_size)
        new_x_shape = x.size()[:-1] + (self.n_heads, self.head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3) # (batch_size, n_heads, seq_len, head_size)

    def forward(self, h_states, attention_mask):
        """
        h_states = (batch_size, seq_len, hiddne_size)
        attention_mask = (batch_size, seq_len)
        """
        query_layer = self.transpose_for_scores(self.query(h_states)) 
        key_layer = self.transpose_for_scores(self.key(h_states))
        value_layer = self.transpose_for_scores(self.value(h_states))
            # (batch_size, n_heads, seq_len, head_size)
        
        attn_score = torch.matmul(query_layer, key_layer.transpose(-1, -2)) 
        # (batch_size, n_heads, seq_len, seq_len)
        attn_score = attn_score / (self.head_size ** 0.5) 
        
        # extending attention_mask
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  
        attention_mask = attention_mask.to(dtype=torch.float16)
        attention_mask = (1.0 - attention_mask) * -10000.0
        
        # adopt attention_mask
        attn_score += attention_mask
        # attn_score = self.dropout(nn.Softmax(dim=-1)(attn_score))
        attn_score = self.dropout(torch.softmax(attn_score, dim=-1))

        context_layer = torch.matmul(attn_score, value_layer)  # (batch_size, n_heads, seq_len, head_size,)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # (batch_size, seq_len, n_head, head_size)
        context_layer = context_layer.view(context_layer.size()[:-2] + (self.all_head_size, ))
        # (batch_size, seq_len, all_head_size)

        return context_layer

class BertSelfOutput(nn.Module):
    def __init__(self, hidden_size, layer_norm_eps=1e-12, dropout_p=0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        self._init_weight()

    def _init_weight(self):
        nn.init.normal_(self.dense.weight,mean=0.0, std=0.02)
        self.dense.bias.data.zero_()

    def forward(self, h_state):
        """
        h_state: output of BertSelfAttention, (batch_size, seq_len, hidden_size)
        """
        return self.dropout(self.dense(h_state)) # (batch_size, seq_len, hidden_size)


class BertAttention(nn.Module):
    def __init__(self, hidden_size, n_heads=2, layer_norm_eps=1e-12, dropout_p=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.selfattn = BertSelfAttention(hidden_size, n_heads, dropout_p)
        self.output = BertSelfOutput(hidden_size, layer_norm_eps, dropout_p)

        self._init_weight()

    def _init_weight(self):
        self.layer_norm.weight.data.fill_(1.0)
        self.layer_norm.bias.data.zero_()

    def forward(self, h_states, attention_mask):
        """
        h_states = (batch_size, seq_len, hidden_size)
        attention_mask = (batch_size, seq_len)
        enc_h_states = (batch_size, seq_len, hidden_size). If None, we get self-attention
        """
        self_out = self.selfattn(self.layer_norm(h_states), attention_mask)
        return self.output(self_out) + h_states  # (batch_size, seq_len, hidden_size)


class FFNN(nn.Module):
    def __init__(self, hidden_size, d_ff, layer_norm_eps=1e-12, dropout_p=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.ffnn1 = nn.Linear(hidden_size, d_ff)
        self.ffnn2 = nn.Linear(d_ff, hidden_size)

        self.gelu = nn.functional.gelu
        self.dropout = nn.Dropout(dropout_p)

        self._init_weight()
    
    def _init_weight(self):
        nn.init.normal_(self.ffnn1.weight, mean=0.0, std=0.02)
        self.ffnn1.bias.data.zero_()
        nn.init.normal_(self.ffnn2.weight, mean=0.0, std=0.02)
        self.ffnn2.bias.data.zero_()
        self.layer_norm.weight.data.fill_(1.0)
        self.layer_norm.bias.data.zero_()


    def forward(self, h_states):
        """
        h_states = (batch_size, seq_len, hidden_size)
        """
        next = self.gelu(self.ffnn1(self.layer_norm(h_states)))
        return self.dropout(self.ffnn2(next)) + h_states  #(batch_size, seq_len, hidden_size)

class BertLayer(nn.Module):
    def __init__(self, hidden_size, d_ff, n_heads=2, layer_norm_eps=1e-12, dropout_p=0.1):
        super().__init__()
        self.attn = BertAttention(hidden_size, n_heads, layer_norm_eps, dropout_p)
        self.cross_attention = BertAttention(hidden_size, n_heads, layer_norm_eps, dropout_p)
        self.ffnn = FFNN(hidden_size, d_ff, layer_norm_eps, dropout_p)

    def forward(self, h_states, attention_mask):
        """
        h_states: output of embedding layer, (batch_size, seq_len, hidden_size)
        attention_mask = (batch_size, seq_len)
        """
        attn_output = self.attn(h_states, attention_mask)
        ffnn_output = self.ffnn(attn_output)

        return ffnn_output  # (batch_size, seq_len, hidden_size)



class BertEncoder(nn.Module):
    def __init__(self,  hidden_size, d_ff, n_layers=2, n_heads=2, layer_norm_eps=1e-12, dropout_p=0.1):
        super().__init__()
        self.layers = nn.ModuleList([BertLayer(hidden_size, d_ff, n_heads, layer_norm_eps, dropout_p) for _ in range(n_layers)])
    
    def forward(self, h_states, attention_mask):
        for layer in self.layers:
            h_states = layer(h_states, attention_mask)

        return h_states

class BertPooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self._init_weight()

    def _init_weight(self):
        nn.init.normal_(self.dense.weight,mean=0.0, std=0.02)
        self.dense.bias.data.zero_()
    
    def forward(self, h_states):
        first_token = h_states[:, 0]  # 각 배치의 cls token 
        return self.activation(self.dense(first_token))  # (batch_size, hidden_size)
