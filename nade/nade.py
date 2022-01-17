import numpy as np
import fasttext
import json
from . import __path__ as ROOT_PATH
from os.path import isfile
import pyarrow.compute as pcm
import pyarrow as pa

# hotfix: ignore warning
fasttext.FastText.eprint = lambda x: None

class Nade:
    '''
    load models & lookup tables
    '''
    def __init__(self, model = 'socialmedia_en', full_model=False, lleaves=False):
        
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
        self.gb_reg = {}
        
        if lleaves:
            import lleaves
            from pathlib import Path
            
            cache_folder = f'{Path.home()}/.nade'
            Path(cache_folder).mkdir(parents=True, exist_ok=True)
        else:
            import lightgbm as lgb
        
        for l in self.labels:
            path = f'{self.model_paths["base"]}/reg_{l}.txt'
            
            if lleaves:
                cache_file = f'{cache_folder}/{l}_lleaves.tmp'
                
                self.gb_reg[l] = lleaves.Model(model_file=path)
                self.gb_reg[l].compile(cache=cache_file)
                
            else:
                self.gb_reg[l] = lgb.Booster(model_file=path)
        
        self.elookup = { v:k for k, v in self.emojis.items() }
    
    '''
    predict emojis based on text (stage 1)
    '''
    def predict_emojis(self, txts, k = 10, sort_by_key = False):
        if k < 0 or k > self.max_k:
            raise Exception(f'please select a k between 0 and {len(self.emojis)}')
        
        txts = Nade.preprocess(txts)
        
        label_raw_np, cred_np = self.tm.predict(txts.tolist(), k = k)
        py_labels = label_raw_np

        label_raw_pa = map(
            lambda x: pa.array(
                [ int(i.lstrip('__label__')) for i in x], 
                type=pa.int16()
            ), 
            label_raw_np
        )
        cred_pa = [ pa.array(i) for i in cred_np ]
        preds = zip(cred_pa, label_raw_pa)
        
        if sort_by_key:
            preds = Nade.sort_predictions(preds)
            
        # convert labels into emojis
        cred_pa, labels = zip(*preds)
        labels = map(lambda x: pa.array(self.elookup[i] for i in x.tolist()), labels)
        
        return list(zip(cred_pa, labels))
    
    '''
    predict emotions based on emoji (stage 2)
    '''
    def predict(self, txts):
        ft_op = self.predict_emojis(txts, sort_by_key=True, k=151)
        X, _ = zip(*ft_op)
        X_np = np.stack(X, axis=0)
        
        raw_reg = { 
            l : np.around(
                    np.clip(self.gb_reg[l].predict(X_np), 
                        a_min=0, 
                        a_max=1
                    ), 
                decimals = 3
            )
            for l in self.labels
        }

        return raw_reg
    
    
    '''
    preprocess applies minimal preprocessing
    
     - add whitespace between punctuation and words
     - reduce multiple whitespaces to one
     - convert text to lower case
     - remove leading and trailing whitespace
    '''
    @staticmethod
    def preprocess(txts):
        
        # wrap if not a list
        if isinstance(txts, str):
            txts = pa.array([txts], type=pa.string())
            
        txts = pcm.replace_substring_regex(txts, pattern=r'\s*([\p{P}]+)\s*', replacement =  ' \\1 ')
        txts = pcm.replace_substring_regex(txts, pattern=r'\s+', replacement =  ' ')
        txts = pcm.utf8_lower(txts)
        txts = pcm.utf8_trim_whitespace(txts)

        return txts
    
    '''
    sort predictions based on index (fixes ordering)
    '''
    @staticmethod
    def sort_predictions(preds):
        def sort_single(x):
            c, l = x

            sorting = pcm.sort_indices(l)
            return (c.take(sorting), l.take(sorting))
    
        return map(sort_single, preds)