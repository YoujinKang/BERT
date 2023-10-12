import argparse
import json
import os
import math
import time
from tqdm.auto import tqdm
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, AdamW
from datasets import load_dataset, load_metric
from FineTunigModel import BertNLI
from data_loader import ClassificationDataset
import sys
# 상위 폴더 접근
sys.path.append(r'/data/user4/BERT')
from Model import BertModel


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default='config/config.json')
    parser.add_argument("--output_dir", type=str, default='saved_model')
    parser.add_argument("--device", type=str, default='gpu')
    parser.add_argument("--dataset", type=str, default='mnli')
    parser.add_argument("--num_labels", type=int, default=3)
    args = parser.parse_args()

    device = torch.device('cuda:0') if args.device is 'gpu' else 'cpu'
    print("device: ", device)

    with open(args.config_file) as f:
        config = json.load(f)

    log_dir = f'{args.output_dir}/{args.dataset}'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    pad_idx = 0

    data = load_dataset('glue', f'{args.dataset}').copy()

    train = ClassificationDataset(data['train'], config['model']['max_position_embeddings'], ('premise', 'hypothesis') ,'label')
    valid = ClassificationDataset(data['validation_matched'], config['model']['max_position_embeddings'], ('premise', 'hypothesis') ,'label')
 
    train_loader = DataLoader(train, batch_size=config['fine_tuning']['batch_size'], shuffle=False)
    valid_loader = DataLoader(valid, batch_size=config['fine_tuning']['batch_size'], shuffle=False)

    print("DataLoaders are prepared")

    model = BertModel(config['model']['hidden_size'], config['model']['vocab_size'],
                    config['model']['d_ff'], config['model']['n_layers'], config['model']['n_heads'],
                    config['model']['max_position_embeddings'], type_vocab_size=2, pad_idx=pad_idx, 
                    layer_norm_eps=config['model']['layer_norm_eps'], dropout_p=config['model']['dropout_p'])

    ckpt = torch.load(f'{args.output_dir}/pretraining3/model_final.ckpt', map_location=device)
    model.load_state_dict(ckpt['bert_model_state_dict'])
    print("Step_num: ", ckpt['step_num'])

    classification_head = BertNLI(model, config['model']['hidden_size'], args.num_labels).to(device)
    
    total_steps = math.ceil(config['fine_tuning']['max_epoch'] * len(data['train'])*1./config['fine_tuning']['batch_size'])
    warmup_steps = int(total_steps * 0.2)

    optimizer = AdamW(classification_head.parameters(), lr=2e-5, eps=1e-6, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    criterion = nn.CrossEntropyLoss().to(device)

    max_epoch = config['fine_tuning']['max_epoch']
    total_loss = 0
    total_dev_loss = 0
    total_s_time = time.time()
    lowest_loss = 100000

    glue_metric = load_metric('glue', f'{args.dataset}')     

    for epoch in range(0, max_epoch):
        s_time = time.time()
        print("******************** Epoch: {}/{} ********************".format(epoch+1, max_epoch))

        classification_head.train()
        for inputs, attention_mask, token_type_ids, labels  in tqdm(train_loader):
            inputs = inputs.squeeze(1).to(device)
            attention_mask = attention_mask.squeeze(1).to(device)
            token_type_ids = token_type_ids.squeeze(1).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            class_out = classification_head(inputs, attention_mask, token_type_ids)

            loss = criterion(class_out.view(-1, args.num_labels), labels.view(-1)) 
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(classification_head.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        classification_head.eval()
        correct_cnt = 0
        with torch.no_grad():
            for inputs, attention_mask, token_type_ids, labels  in tqdm(valid_loader):
                inputs = inputs.squeeze(1).to(device)
                attention_mask = attention_mask.squeeze(1).to(device)
                token_type_ids = token_type_ids.squeeze(1).to(device)
                labels = labels.to(device)

                class_out = classification_head(inputs, attention_mask, token_type_ids)

                loss = criterion(class_out.view(-1, args.num_labels), labels.view(-1))
                total_dev_loss += loss.item()

                _, top_pred = torch.topk(class_out, k=1, dim=-1)
                glue_metric.add_batch(predictions=top_pred.squeeze(), references=labels)

        
            print("Prediction: ", top_pred.squeeze().tolist()[:10])
            print("References: ", labels.tolist()[:10])
            score1 = glue_metric.compute()
            print(f"Score for mnli_m: {score1}")

            if total_dev_loss < lowest_loss:
                lowest_loss = total_dev_loss
                torch.save({"model_state_dict": classification_head.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()}, 
                log_dir+f"/model_ft.ckpt")

            total_dev_loss = 0

            