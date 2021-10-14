import numpy as np
import fasttext
import json
from catboost import CatBoostRegressor
import re2
from . import __path__ as ROOT_PATH
from os.path import isfile

# hotfix: ignore warning
fasttext.FastText.eprint = lambda x: None

class Nade:
    def __init__(self, model = 'socialmedia_en', full_model=False):
        
        # set paths and check them
        self.model_paths = {
            'emoji_index': f'{ROOT_PATH[0]}/data/{model}/emoji_frequencies.jsonl',
            'emoji_clf': f'{ROOT_PATH[0]}/data/{model}/nade_250k_hp.{"bin" if full_model else "ftz"}',
            'emotion_reg': f'{ROOT_PATH[0]}/data/{model}/cb_reg.cbm'
        }
        
        assert isfile(self.model_paths['emoji_index'])
        assert isfile(self.model_paths['emoji_clf'])
        assert isfile(self.model_paths['emotion_reg'])
        
        
        # load emoji index
        self.emojis = dict()
        
        with open(self.model_paths['emoji_index']) as f:
            for l in f:
                dct = json.loads(l)
                self.emojis[dct['emoji']] = dct['hash']
        
        # load models (stage I & stage II)
        self.tm = fasttext.load_model(self.model_paths['emoji_clf'])
        self.cbreg = CatBoostRegressor()
        self.cbreg.load_model(self.model_paths['emotion_reg'])
        
        self.labels = [
            'anger', 'anticipation', 'disgust', 
            'fear', 'joy', 'sadness', 'surprise', 
            'trust'
        ]
        
        self.elookup = { v:k for k, v in self.emojis.items() }
        
            
    def __emoji_prediction__(self, txt, k = 10):
        txt = Nade.preprocess(txt)
        
        l_raw, cred = self.tm.predict(txt, k = k)
        labels = [int(l[9:]) for l in l_raw]
        
        return zip(labels, cred)
    
    def predict_emojis(self, txt, k = 10, sort_by_key = False):
        if k < 0 or k > len(self.emojis):
            raise Exception(f'please select a k between 0 and {len(self.emojis)}')
        
        if sort_by_key:
            return {self.elookup[k]:v for k, v in sorted(self.__emoji_prediction__(txt, k))}
            
        return { 
            self.elookup[k]:v 
            for k, v in sorted(self.__emoji_prediction__(txt, k), key=lambda item: item[1], reverse = True)
        }

    def reg_predict(self, emoji_scores):
        raw_reg = self.cbreg.predict(emoji_scores)
        format_reg = np.around(np.clip(raw_reg, a_min=0, a_max=1), decimals = 3)
        
        return format_reg
    
    def predict(self, txt):
        raw_scores = self.predict_emojis(txt, k=151, sort_by_key=True).values()
        emoji_scores = np.array(list(raw_scores))
        
        format_reg = self.reg_predict(emoji_scores)

        return dict(zip(self.labels, format_reg))
    
    @staticmethod
    def masked_rmse(y_true, y_pred):
        se = (y_true - y_pred)**2
        mask = (y_true > 0).astype(np.int)

        return np.sqrt((se*mask).mean())
    
    @staticmethod
    def preprocess(txt):
        txt = re2.sub('\s*([\p{P}]+)\s*', ' \\1 ', txt)
        txt = re2.sub('\s+', ' ', txt)
        txt = txt.lower()
        txt = txt.strip()
        return txt