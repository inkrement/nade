from typing import List
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
    def __init__(self, model: str = 'socialmedia_en', lleaves: bool = False):
        root_pth = ROOT_PATH[0]

        # set paths and check them
        self.model_paths = {
            'base': f'{root_pth}/data/{model}',
            'emoji_index': f'{root_pth}/data/{model}/emoji_frequencies.jsonl',
            'emoji_clf': f'{ROOT_PATH[0]}/data/{model}/nade_250k_hp.ftz'
        }

        assert isfile(self.model_paths['emoji_index'])
        assert isfile(self.model_paths['emoji_clf'])

        # load emoji index
        self.emojis = dict()

        with open(self.model_paths['emoji_index']) as f:
            for line in f:
                dct = json.loads(line)
                self.emojis[dct['emoji']] = dct['hash']

        self.max_k = len(self.emojis)

        # load models (stage I & stage II)
        self.tm = fasttext.load_model(
            self.model_paths['emoji_clf']
        )

        self.labels = [
            'anger', 'anticipation', 'disgust',
            'fear', 'joy', 'sadness', 'surprise',
            'trust'
        ]

        # load gradient boosting models (one per label)
        self.gb_reg = {}

        if lleaves:
            import lleaves
            from pathlib import Path

            cache_folder = f'{Path.home()}/.nade'
            Path(cache_folder).mkdir(parents=True, exist_ok=True)
        else:
            import lightgbm as lgb

        for e_lbl in self.labels:
            path = f'{self.model_paths["base"]}/reg_{e_lbl}.txt'
            gz_path = f'{path}.gz'

            # unzip if not existing
            if not isfile(path) and isfile(gz_path):
                import gzip
                import shutil

                print(f'unpack {e_lbl} model')

                with gzip.open(gz_path, 'rb') as f_in:
                    with open(path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

            if lleaves:
                cache_file = f'{cache_folder}/{e_lbl}_lleaves.tmp'
                self.gb_reg[e_lbl] = lleaves.Model(model_file=path)
                self.gb_reg[e_lbl].compile(cache=cache_file)

            else:
                self.gb_reg[e_lbl] = lgb.Booster(model_file=path)

        self.elookup = {v: k for k, v in self.emojis.items()}

    '''
    predict emojis based on text (stage 1)
    '''
    def predict_emojis(
            self, txts: List[str], k: int = 10, sort_by_key: bool = False
            ):
        if k < 0 or k > self.max_k:
            raise Exception(
                f'please select a k between 0 and {len(self.emojis)}'
                )

        txts = Nade.preprocess(txts)
        label_raw_np, cred_np = self.tm.predict(txts.tolist(), k=k)

        label_raw_pa = map(
            lambda x: pa.array(
                [int(i.lstrip('__label__')) for i in x],
                type=pa.int16()
            ),
            label_raw_np
        )
        cred_pa = [pa.array(i) for i in cred_np]
        preds = zip(cred_pa, label_raw_pa)

        if sort_by_key:
            preds = Nade.sort_predictions(preds)

        # convert labels into emojis
        cred_pa, labels = zip(*preds)
        labels = map(
            lambda x: pa.array(self.elookup[i] for i in x.tolist()),
            labels
        )

        return list(zip(cred_pa, labels))

    '''
    predict emotions based on emoji (stage 2)
    '''
    def predict(self, txts: List[str]) -> List[str]:
        ft_op = self.predict_emojis(txts, sort_by_key=True, k=151)
        X, _ = zip(*ft_op)

        raw_reg = {
            lbl: pcm.around(
                Nade.clip(
                    self.gb_reg[lbl].predict(list(X)),
                    a_min=0,
                    a_max=1
                ),
                ndigits=3
            )
            for lbl in self.labels
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
    def preprocess(txts: List[str]) -> List[str]:

        # wrap if not a list
        if isinstance(txts, str):
            txts = pa.array([txts], type=pa.string())

        txts = pcm.replace_substring_regex(
            txts, pattern=r'\s*([\p{P}]+)\s*', replacement=' \\1 '
            )
        txts = pcm.replace_substring_regex(
            txts, pattern=r'\s+', replacement=' '
            )
        txts = pcm.utf8_lower(txts)
        txts = pcm.utf8_trim_whitespace(txts)

        return txts

    '''
    mimics np.clip using pyarrow

    clips arr into a range between a_min and a_max
    '''
    @staticmethod
    def clip(arr: List[float], a_min: float = 0.0, a_max: float = 1.0):
        return pcm.case_when(
            pcm.make_struct(
                pcm.greater(arr, a_max),
                pcm.less(arr, a_min)
            ),
            a_max, a_min, arr
        )

    '''
    sort predictions based on index (fixes ordering)
    '''
    @staticmethod
    def sort_predictions(preds):
        def sort_single(x):
            c, lbl = x

            sorting = pcm.sort_indices(lbl)
            return (c.take(sorting), lbl.take(sorting))

        return map(sort_single, preds)
