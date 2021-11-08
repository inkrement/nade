import numpy as np
import fasttext
import json
import lightgbm as lgb
import regex as re
from . import __path__ as ROOT_PATH
from os.path import isfile

# hotfix: ignore warning
fasttext.FastText.eprint = lambda x: None

class Nade:
    def __init__(self, model = 'socialmedia_en', full_model=False):
        
        # set paths and check them
        self.model_paths = {
            'base': f'{ROOT_PATH[0]}/data/{model}',
            'emoji_index': f'{ROOT_PATH[0]}/data/{model}/emoji_frequencies.jsonl',
            'emoji_clf': f'{ROOT_PATH[0]}/data/{model}/nade_250k_hp.{"bin" if full_model else "ftz"}'
        }
        
        assert isfile(self.model_paths['emoji_index'])
        assert isfile(self.model_paths['emoji_clf'])
        
        # load emoji index
        self.emojis = dict()
        
        with open(self.model_paths['emoji_index']) as f:
            for l in f:
                dct = json.loads(l)
                self.emojis[dct['emoji']] = dct['hash']
        
        self.max_k = len(self.emojis)
        
        # load models (stage I & stage II)
        self.tm = fasttext.load_model(self.model_paths['emoji_clf'])
        
        self.labels = [
            'anger', 'anticipation', 'disgust', 
            'fear', 'joy', 'sadness', 'surprise', 
            'trust'
        ]
        
        for l in self.labels:
            assert isfile(f'{self.model_paths["base"]}/reg_{l}.txt')
        
        # load gradient boosting models (one per label)
        self.gb_reg = {
            l: lgb.Booster(
                model_file=f'{self.model_paths["base"]}/reg_{l}.txt'
            ) for l in self.labels 
        }
        
        self.elookup = { v:k for k, v in self.emojis.items() }
        
            
    def __emoji_prediction__(self, txt, k = 10):
        txt = Nade.preprocess(txt)
        
        l_raw, cred = self.tm.predict(txt, k = k)
        labels = [int(l[9:]) for l in l_raw]
        
        return zip(labels, cred)
    
    def predict_emojis(self, txt, k = 10, sort_by_key = False):
        if k < 0 or k > self.max_k:
            raise Exception(f'please select a k between 0 and {len(self.emojis)}')
        
        if sort_by_key:
            return {
                self.elookup[k]:v for k, v in 
                sorted(self.__emoji_prediction__(txt, k))
            }
            
        return { 
            self.elookup[k]:v 
            for k, v in sorted(
                self.__emoji_prediction__(txt, k), 
                key=lambda item: item[1], 
                reverse = True
            )
        }

    def reg_predict(self, emoji_scores):
        X = emoji_scores.reshape(1,-1)
        
        raw_reg = { 
            l : np.around(
                    np.clip(self.gb_reg[l].predict(X)[0], 
                        a_min=0, 
                        a_max=1
                    ), 
                decimals = 3
            )
            for l in self.labels
        }
        
        return raw_reg
    
    def predict(self, txt):
        raw_scores = np.fromiter(
            self.predict_emojis(txt, k=self.max_k, sort_by_key=True).values(),
            dtype=float
        )

        return self.reg_predict(raw_scores)
    
    @staticmethod
    def masked_rmse(y_true, y_pred):
        se = (y_true - y_pred)**2
        mask = (y_true > 0).astype(np.int)

        return np.sqrt((se*mask).mean())
    
    @staticmethod
    def preprocess(txt):
        txt = re.sub(r'\s*([\p{P}]+)\s*', ' \\1 ', txt)
        txt = re.sub(r'\s+', ' ', txt)
        txt = txt.lower()
        txt = txt.strip()
        return txt