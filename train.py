import argparse
import json
import os
import time
import numpy as np
from tqdm.auto import tqdm
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset
from data_loader import CustomDataset
from Model import BertModel
from LanguageModel import BertPreTrainingHead


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default='config/config.json')
    parser.add_argument("--output_dir", type=str, default='saved_model')
    parser.add_argument("--device", type=str, default='gpu')
    args = parser.parse_args()

    device = torch.device('cuda:0') if args.device is 'gpu' else 'cpu'
    print("device: ", device)

    with open(args.config_file) as f:
        config = json.load(f)

    log_dir = args.output_dir + '/pretraining3'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    tb_writer = SummaryWriter(log_dir)

    pad_idx = 0

    bookcorpus = load_dataset('bookcorpus', split='train')['text'].copy()
    wikipedia = load_dataset('wikipedia', '20200501.en', split='train')['text'].copy()
    train = CustomDataset(bookcorpus+wikipedia, config['model']['max_position_embeddings'])
    
    train_loader = DataLoader(train, batch_size=config['train']['batch_size'], shuffle=True)
    print("DataLoaders are prepared")

    model = BertModel(config['model']['hidden_size'], config['model']['vocab_size'],
                    config['model']['d_ff'], config['model']['n_layers'], config['model']['n_heads'],
                    config['model']['max_position_embeddings'], type_vocab_size=2, pad_idx=pad_idx, 
                    layer_norm_eps=config['model']['layer_norm_eps'], dropout_p=config['model']['dropout_p'])
    
    pretraining_head = BertPreTrainingHead(model, config['model']['hidden_size'], config['model']['vocab_size'], layer_norm_eps=config['model']['layer_norm_eps']).to(device)

    optimizer = optim.Adam(pretraining_head.parameters(), lr=config['train']['lr'], betas=[config['train']['beta1'], config['train']['beta2']], weight_decay=config['train']['weight_decay'])
    criterion = nn.CrossEntropyLoss(ignore_index=-100).to(device)

    # ckpt = torch.load(f'{log_dir}/model.ckpt', map_location=device)
    # pretraining_head.load_state_dict(ckpt['pt_model_state_dict'])
    # optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    

    max_epoch = config['train']['max_epoch']
    total_loss = 0
    stack = 0
    step_num = 0
    # step_num = ckpt['step_num']
    print("Step_num: step_num")
    total_s_time = time.time()
    lowest_loss = 10000

    for epoch in range(0, max_epoch):
        s_time = time.time()
        print("*"*20 + "Epoch: {}/{}".format(epoch+1, max_epoch) + "*"*20)

        pretraining_head.train()
        for inputs, attention_mask, token_type_ids, labels, next_seq_label in tqdm(train_loader):
            inputs = inputs.squeeze(1).to(device)
            attention_mask = attention_mask.squeeze(1).to(device)
            token_type_ids = token_type_ids.squeeze(1).to(device)
            labels = labels.squeeze(1).to(device)
            next_seq_label = next_seq_label.to(device)

            optimizer.zero_grad()
            mlm_out, nsp_out = pretraining_head(inputs, attention_mask, token_type_ids)

            loss_mlm = criterion(mlm_out.view(-1, config['model']['vocab_size']), labels.view(-1))
            loss_nsp = criterion(nsp_out.view(-1, 2), next_seq_label.view(-1))
            loss = loss_mlm + loss_nsp    
            total_loss += loss.item()

            loss.backward()
            stack += 1

            if step_num == 1000000:
                break

            if stack % config['train']['step_batch'] == 0:
                step_num += 1
                optimizer.param_groups[0]['lr'] = config['model']['hidden_size'] ** (-0.5) * np.minimum(step_num ** (-0.5), step_num * (config['train']['warmup'] ** (-1.5)))
                optimizer.step()
                optimizer.zero_grad()


            if stack % config['train']['eval_interval'] == 1:
                elapsed = (time.time() - s_time)/60
                print("Step: %d | Loss: %f | Time:  %f [min]" %(step_num, total_loss, elapsed))
                
                tb_writer.add_scalar('loss/step', total_loss, step_num)
                tb_writer.add_scalar('lr/step', optimizer.param_groups[0]['lr'], step_num)
                tb_writer.flush()
                
                s_time = time.time()
                total_loss = 0
                torch.save({"pt_model_state_dict": pretraining_head.state_dict(),
                            "bert_model_state_dict": pretraining_head.model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "step_num": step_num}, log_dir+f"/model.ckpt")

            else:
                continue
        
        print(f"Time passed from training: {(time.time()-total_s_time)/3600} [h]")
        torch.save({"pt_model_state_dict": pretraining_head.state_dict(),
                    "bert_model_state_dict": pretraining_head.model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "step_num": step_num}, log_dir+f"/model_{epoch+1}.ckpt")
        print("Model is saved!")

    print(f"Time passed from training: {(time.time()-total_s_time)/3600} [h]")
    torch.save({"pt_model_state_dict": pretraining_head.state_dict(),
                "bert_model_state_dict": pretraining_head.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step_num": step_num}, log_dir+f"/model_final.ckpt")
    print("Final Model is saved!")