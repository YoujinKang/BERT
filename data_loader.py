from torch.utils.data import Dataset
import torch
import random
import unicodedata
from tqdm.auto import tqdm

from transformers import BertTokenizerFast, data

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
data_col = data.data_collator.DataCollatorForLanguageModeling(tokenizer)

# using sentencepiece
class LMDataset(Dataset):
    def __init__(self, data, vocab, sep_token, cls_token, mask_token, pad_token, 
                seq_len=512, mask_frac=0.15, nsp_prob=0.5):
        """
        data: list of tensors
        vocab: sentencepiece model
        mask_frac: masking rate per input
        nsp_prob: Probability for NSP
        """
        super().__init__()
        self.data = data
        self.vocab = vocab
        self.sep_id = vocab.piece_to_id(sep_token)
        self.cls_id = vocab.piece_to_id(cls_token)
        self.mask_id = vocab.piece_to_id(mask_token)
        self.pad_id = vocab.piece_to_id(pad_token)
        self.seq_len = seq_len
        self.mask_frac = mask_frac
        self.nsp_prob = nsp_prob

    def __getitem__(self, idx):
        seq1 = self.vocab.EncodeAsIds(self.data[idx].strip().lower()) # text->list of ids
        seq2_idx = idx+1 # initial setting
        
        # make next_sentence_label
        if random.random() > self.nsp_prob:  # NotNext
            next_seq_label = torch.tensor(1)
            while seq2_idx == idx+1:  # seq2_idx has not to be idx+1
                seq2_idx = random.randint(0, len(self.data)-1)
        else: # IsNext
            if seq2_idx >= len(self.data):  # if idx is the last value
                next_seq_label = torch.tensor(1)
                seq2_idx = idx-random.randint(1, idx-1)
            else:
                next_seq_label = torch.tensor(0) # seq2_idx is idx+1

        seq2 = self.vocab.EncodeAsIds(self.data[seq2_idx].strip().lower())

        # if length of tokens are more than 512
        if len(seq1) + len(seq2) >= self.seq_len - 3: # 1 [CLS], 2 [SEP]
            if len(seq2) >= len(seq1):
                id = self.seq_len - 3 - len(seq1)
                seq2 = seq2[:id]
            else:
                id = self.seq_len - 3 - len(seq2)
                seq1 = seq1[:id]
        
        targets = torch.tensor([self.cls_id] + seq1 + [self.sep_id] + seq2 + [self.sep_id] + [self.pad_id] * (self.seq_len - 3 - len(seq1) - len(seq2))).long().contiguous()
        
        attention_mask = len([self.cls_id] + seq1 + [self.sep_id] + seq2 + [self.sep_id]) * [1] + (self.seq_len - 3 - len(seq1) - len(seq2)) * [0]
        attention_mask = torch.tensor(attention_mask)

        token_type_ids = len([self.cls_id] + seq1 + [self.sep_id]) * [0] + len(seq2 + [self.sep_id]) * [1] 
        if self.seq_len - len(token_type_ids) > 0:
            token_type_ids = token_type_ids + (self.seq_len - len(token_type_ids)) * [2]
        token_type_ids = torch.tensor(token_type_ids)

        train_tokens = torch.cat([torch.tensor([self.cls_id]), self.masking(seq1), torch.tensor([self.sep_id]), self.masking(seq2), torch.tensor([self.sep_id])]).long().contiguous()
        inputs = torch.cat([train_tokens, torch.tensor([self.pad_id] * (self.seq_len - train_tokens.size(0)))]).long().contiguous()

        return inputs, targets, attention_mask, token_type_ids, next_seq_label

    def masking(self, token_ids):
        token_ids = torch.tensor(token_ids).long().contiguous()
        token_len = token_ids.size(0)

        ones_num = int(token_len * self.mask_frac * 0.9)  # number of masking tokens
        zeros_num = token_len - ones_num  
        lm_mask = torch.cat([torch.zeros(zeros_num), torch.ones(ones_num)])
        lm_mask = lm_mask[torch.randperm(token_len)]
        masked_token = token_ids.masked_fill(lm_mask.bool(), self.mask_id)

        ones_num2 = int(token_len * self.mask_frac * 0.1)  # number of masking tokens
        zeros_num2 = token_len - ones_num2  
        lm_mask2 = torch.cat([torch.zeros(zeros_num2), torch.ones(ones_num2)])
        lm_mask2 = lm_mask2[torch.randperm(token_len)]
        masked_token2 = masked_token.masked_fill(lm_mask2.bool(), random.randint(7, 30000))
            
        return masked_token2

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for x in self.data:
            yield x
        
    def get_vocab(self):
        return self.vocab
    
    def decode(self, token_ids):
        return self.vocab.DecodeIds(token_ids)


def load_dataset(train_file, valid_file):
    def read_text(file):

        with open(file, 'r') as f:
            data = f.readlines()
        return data
    
    train = read_text(train_file)
    valid = read_text(valid_file)

    return train, valid


# using huggingface tokenizer (much faster)
class CustomDataset(Dataset):
    def __init__(self, data, max_len):
        self.data = data
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq2_idx = idx+1
        if random.random() > 0.5:  # NotNext
            next_seq_label = torch.tensor(1)
            while seq2_idx == idx+1:  # seq2_idx has not to be idx+1
                seq2_idx = random.randint(0, len(self.data)-1)
        else: # IsNext
            if seq2_idx >= len(self.data):  # if idx is the last value
                next_seq_label = torch.tensor(1)
                seq2_idx = idx-random.randint(1, idx-1)
            else:
                next_seq_label = torch.tensor(0) # seq2_idx is idx+1

        inputs = tokenizer(self.data[idx], self.data[seq2_idx], padding='max_length', truncation=True, max_length=512, return_tensors="pt")
        _, inputs["label"] = data_col.torch_mask_tokens(inputs["input_ids"])
        inputs["next_seq_label"] = next_seq_label
        inputs["token_type_ids"] = inputs["token_type_ids"]+(1-inputs["attention_mask"])*2

        return inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"], inputs["label"], inputs["next_seq_label"]

