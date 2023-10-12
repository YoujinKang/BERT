from easydict import EasyDict
import json


class JsonConfigFileManager:
    """
    json 설정 파일 관리
    """
    def __init__(self, file_path):
        self.values = EasyDict()
        if file_path:
            self.file_path = file_path
            self.reload()
    
    def reload(self):
        """
        설정 리셋, 설정파일 재로드
        """
        self.clear()
        if self.file_path:
            with open(self.file_path, 'r') as f:
                self.values.update(json.load(f))
        
    def clear(self):
        """
        설정 리셋
        """
        self.values.clear()
    
    def update(self, in_dict):
        """
        기존 설정에 새로운 설정 업데이트
        """
        for (k1, v1) in in_dict.items():
            self.values[k1] = v1
            if isinstance(v1, dict):
                for (k2, v2) in v1.items():
                    self.values[k1][k2] = v2

    def export(self, save_file_name):
        """
        설정값을 json 파일로 저장
        """
        if save_file_name:
            with open(save_file_name, 'w') as f:
                json.dump(dict(self.values), f, indent=4)




if __name__ == '__main__':
    conf = JsonConfigFileManager('config/config.json')
    print(conf.values)
    updates = {"model": { 
                    "hidden_size": 128,
                    "vocab_size": 30522,
                    "d_ff": 64,
                    "n_layers": 2, 
                    "n_heads": 2,
                    "max_position_embeddings": 512,
                    "type_vocab_size": 2,
                    "layer_norm_eps": 1e-12,
                    "dropout_p": 0.1}, 
                "train": {
                    "batch_size": 64, 
                    "step_batch": 4,
                    "max_epoch": 100,
                    "beta1": 0.9, 
                    "beta2": 0.999, 
                    "warmup": 4000, 
                    "eval_interval": 10000,  
                    "weight_decay": 0.01,
                    "lr": 1e-04,
                    "smoothing": 0.1},
                "fine_tuning": {
                    "batch_size": 16, 
                    "max_epoch": 2,
                    "lr": 2e-5,
                    "eval_interval": 100}}
    conf.update(updates)
    print(conf.values)
    conf.export('config/config.json')

    conf = JsonConfigFileManager('config/config_test.json')
    print(conf.values)
    updates = {"model": { \
                    "hidden_size": 6,
                    "vocab_size": 30522,
                    "d_ff": 3,
                    "n_layers": 2, 
                    "n_heads": 2,
                    "max_position_embeddings": 512,
                    "type_vocab_size": 2,
                    "layer_norm_eps": 1e-12,
                    "dropout_p": 0.1}, 
                "train": {
                    "batch_size": 2, 
                    "step_batch": 2,
                    "max_epoch": 3,
                    "beta1": 0.9, 
                    "beta2": 0.999, 
                    "warmup": 3, 
                    "eval_interval": 5,  
                    "weight_decay": 0.01,
                    "lr": 1e-04,
                    "smoothing": 0.1},
                "fine_tuning": {
                    "batch_size": 16, 
                    "max_epoch": 2,
                    "lr": 2e-5,
                    "eval_interval": 100}}
    conf.update(updates)
    print(conf.values)
    conf.export('config/config_test.json')
