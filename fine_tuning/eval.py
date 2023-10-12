import argparse
import json
from tqdm.auto import tqdm
import torch 
from torch.utils.data import DataLoader
from datasets import load_dataset, load_metric
from FineTunigModel import BertClassification
from data_loader import ClassificationDataset
import sys
# 상위 폴더 접근
sys.path.append(r'D:\youjin\OneDrive - 고려대학교\Freshman\BERT')
from Model import BertModel
from LanguageModel import BertPreTrainingHead


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default='config/config.json')
    parser.add_argument("--output_dir", type=str, default='saved_model')
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--num_labels", type=int, required=True)
    args = parser.parse_args()

    device = torch.device('cuda:0') if args.device is 'gpu' else 'cpu'
    print("device: ", device)

    with open(args.config_file) as f:
        config = json.load(f)

    pad_idx = 0
    data = load_dataset('glue', f'{args.dataset}').copy()

    list0 = ['rte','mrpc','stsb']  # using sentence1, sentence2 notation
    list1 = ['qqp']  # using question1, question2 notation
    list2 = ['qnli'] # using question, sentence notation
    list3 = ['sst2', 'cola'] # using only one sentence 
    list4 = ['mnli_matched', 'mnli_mismatched'] # using premise, hypothesis notation
    tags = [('sentence1', 'sentence2'),('question1','question2'),('question','sentence'),'sentence',('premise','hypothsis')]
    if args.dataset in list0:
        tag = tags[0]
    elif args.dataset in list1:
        tag = tags[1]
    elif args.dataset in list2:
        tag = tags[2]
    elif args.dataset in list3:
        tag = tags[3]
    elif args.dataset in list4:
        tag = tags[4]

    test = ClassificationDataset(data['validation'], config['model']['max_position_embeddings'], tag ,'label')
     
    test_loader = DataLoader(test, batch_size=config['fine_tuning']['batch_size'], shuffle=True)
    print("DataLoaders are prepared")

    model = BertModel(config['model']['hidden_size'], config['model']['vocab_size'],
                    config['model']['d_ff'], config['model']['n_layers'], config['model']['n_heads'],
                    config['model']['max_position_embeddings'], type_vocab_size=2, pad_idx=pad_idx, 
                    layer_norm_eps=config['model']['layer_norm_eps'], dropout_p=config['model']['dropout_p'])

    pretraining_head = BertPreTrainingHead(model, config['model']['hidden_size'], config['model']['vocab_size'], layer_norm_eps=config['model']['layer_norm_eps'])
    

    classification_head = BertClassification(pretraining_head.model, config['model']['hidden_size'], args.num_labels, dropout_p=config['model']['dropout_p']).to(device)
    
    ckpt = torch.load(f'{args.output_dir}/{args.dataset}/model_ft.ckpt', map_location=device)    
    classification_head.load_state_dict(ckpt['model_state_dict'])

    prediction = []
    reference = []
    classification_head.eval()
    print("Evaluating ... ")
    glue_metric = load_metric('glue', f'{args.dataset}')     
    with torch.no_grad():
        for inputs, attention_mask, token_type_ids, labels  in tqdm(test_loader):
            inputs = inputs.squeeze(1).to(device)
            attention_mask = attention_mask.squeeze(1).to(device)
            token_type_ids = token_type_ids.squeeze(1).to(device)

            class_out = classification_head(inputs, attention_mask, token_type_ids)
            _, top_pred = torch.topk(class_out, k=1, dim=-1)
            glue_metric.add_batch(predictions=top_pred, references=labels)

        score = glue_metric.compute()
    print(f"Score: {score}")
           