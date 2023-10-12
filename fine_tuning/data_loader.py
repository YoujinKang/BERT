from torch.utils.data import Dataset
import torch

from transformers import BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# use huggingfase tokenizer
class ClassificationDataset(Dataset):
    def __init__(self, data, max_len, tag, label):
        self.tag = tag
        if isinstance(tag, tuple):
            tag1, tag2 = tag
            self.sent1 = list(map(str, self.generator(data, f'{tag1}')))
            self.sent2 = list(map(str, self.generator(data, f'{tag2}')))
        else:
            self.sent = list(map(str, self.generator(data, f'{tag}')))
        self.label = list(map(int, self.generator(data, f'{label}')))
        self.max_len = max_len
    
    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        if isinstance(self.tag, tuple):
            inputs = tokenizer(self.sent1[idx], self.sent2[idx], padding='max_length', truncation=True, max_length=512, return_tensors="pt")
        else:
            inputs = tokenizer(self.sent[idx], padding='max_length', truncation=True, max_length=512, return_tensors="pt")
        inputs["token_type_ids"] = inputs["token_type_ids"]+(1-inputs["attention_mask"])*2
        label = torch.tensor(self.label[idx]).to(dtype=torch.long)

        return inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"], label

    def generator(self, data, name):
        for one_fold in data:
            yield one_fold[f'{name}']

# to check bert pretrained model from huggingface
class ClassificationDataset_bert(Dataset):
    def __init__(self, data, max_len, tag, label):
        self.tag = tag
        if isinstance(tag, tuple):
            tag1, tag2 = tag
            self.sent1 = list(map(str, self.generator(data, f'{tag1}')))
            self.sent2 = list(map(str, self.generator(data, f'{tag2}')))
        else:
            self.sent = list(map(str, self.generator(data, f'{tag}')))
        self.label = list(map(int, self.generator(data, f'{label}')))
        self.max_len = max_len
    
    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        if isinstance(self.tag, tuple):
            inputs = tokenizer(self.sent1[idx], self.sent2[idx], padding='max_length', truncation=True, max_length=512, return_tensors="pt")
        else:
            inputs = tokenizer(self.sent[idx], padding='max_length', truncation=True, max_length=512, return_tensors="pt")
        label = torch.tensor(self.label[idx]).to(dtype=torch.long)

        return inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"], label

    def generator(self, data, name):
        for one_fold in data:
            yield one_fold[f'{name}']