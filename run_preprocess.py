from datasets import load_dataset
import sentencepiece as spm
from tqdm.auto import tqdm

def collect_data():
    bookcorpus = load_dataset("bookcorpus", split='train')['text'].copy()
    wikipedia = load_dataset('wikipedia', '20200501.en', split='train')['text'].copy()

    data = []
    
    for line in tqdm(bookcorpus):
        data.append(line.strip())
    for line in tqdm(wikipedia):
        data.append(line.strip())

    path = 'data/'
    with open(path+"train.txt", 'w') as f:
        for line in tqdm(data):
            f.write("%s\n" % line)
    
    return data


def make_vocab():
    parameter = '--input={} --model_prefix={} --vocab_size={} --user_defined_symbols=[MASK],[PAD],[CLS],[SEP] --model_type={} --character_coverage={} --hard_vocab_limit=false'
    train_input_file = 'data/train.txt'
    # we don't use <s> and </s> but they are automatically included
    vocab_size = 30002  
    prefix = "m"
    model_type = 'bpe'
    character_coverage = 1.0  # default

    cmd = parameter.format(train_input_file, prefix, vocab_size, model_type, character_coverage)

    print("Start making vocabulary...")
    spm.SentencePieceTrainer.Train(cmd)

if __name__=="__main__":
    data = collect_data()
    make_vocab()