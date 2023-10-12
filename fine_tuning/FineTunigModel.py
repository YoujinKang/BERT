import torch.nn as nn


class BertClassification(nn.Module):
    def __init__(self, model, hidden_size, num_labels, dropout_p=0.1):
        super().__init__()
        self.model = model
        self.dense = nn.Linear(hidden_size, hidden_size*4)
        self.classifier = nn.Linear(hidden_size*4, num_labels)
        self.dropout = nn.Dropout(dropout_p)
        
        self._init_weight()

    def _init_weight(self):
        nn.init.normal_(self.dense.weight,mean=0.0, std=0.02)
        self.dense.bias.data.zero_()
        nn.init.normal_(self.classifier.weight,mean=0.0, std=0.02)
        self.classifier.bias.data.zero_()

    def forward(self, input_ids, attention_mask, token_type_ids):
        _, pooled_out = self.model(input_ids, attention_mask, token_type_ids)
        pooled_out = nn.functional.gelu(self.dense(pooled_out))
        return self.classifier(self.dropout(pooled_out)) 
        
        
class BertNLI(nn.Module):
    def __init__(self, model, hidden_size, num_labels):
        super().__init__()
        self.model = model 
        self.classifier = nn.Sequential(
                nn.Linear(hidden_size, 4*hidden_size),
                nn.ReLU(),
                nn.Linear(4*hidden_size, num_labels)
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return self.classifier(output[0][:,0,:])  
