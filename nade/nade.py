import numpy as np
import fasttext
import json
import pickle
import bz2
from . import __path__ as ROOT_PATH

# hotfix: ignore warning
fasttext.FastText.eprint = lambda x: None

class Nade:
    def __init__(self, model = 'socialmedia_en'):
        # TODO: check if model exists
        
        with open(f'{ROOT_PATH[0]}/data/{model}/emojis.json') as f:
            self.emojis = json.load(f)
        
        self.tm = fasttext.load_model(f'{ROOT_PATH[0]}/data/{model}/textmodel.ftz')
        
        with bz2.open(f'{ROOT_PATH[0]}/data/{model}/emotion_regression.pbz', 'rb') as f:
            self.etr = pickle.load(f)
                
        self.labels = [
            'anger', 'anticipation', 'disgust', 
            'fear', 'joy', 'sadness', 'surprise', 
            'trust'
        ]
        
        self.elookup = { v:k for k, v in self.emojis.items() }
        
            
    def emoji_predict(self, txt):
        l_raw, cred = self.tm.predict(txt, k = len(self.emojis))
        labels = [int(l[9:]) for l in l_raw]
        
        return dict(sorted(zip(labels, cred)))

    def reg_predict(self, emoji_predictions):
        np_features = np.fromiter(emoji_predictions.values(), dtype=float).reshape(1,-1)
        
        return dict(zip(self.labels, [ round(v,3) for v in self.etr.predict(np_features)[0].tolist() ]))
    
    def predict(self, txt, emoji_prop = False, top_k = 10):
        txt = Nade.preprocess(txt)
        e_cred = self.emoji_predict(txt)
        
        if emoji_prop:
            top_emojis = sorted(e_cred.items(), reverse=True, key=lambda item: item[1])
            
            topk_emojis = [ { 'emoji': self.elookup[eid] , 'probability': round(p, 3)} for (eid, p) in top_emojis[:top_k]]
            return topk_emojis, self.reg_predict(e_cred)
        
        return self.reg_predict(e_cred)
    
    @staticmethod
    def masked_rmse(y_true, y_pred):
        se = (y_true - y_pred)**2
        mask = (y_true > 0).astype(np.int)

        return np.sqrt((se*mask).mean())
    
    @staticmethod
    def preprocess(txt):
        txt = txt.lower().replace('\n', ' ')
        return txt