from typing import List, Optional, Union
import fasttext
import json
from . import __path__ as ROOT_PATH
from os.path import isfile
import pyarrow.compute as pcm
import pyarrow as pa
import warnings

# hotfix: ignore warning
fasttext.FastText.eprint = lambda x: None


class Nade:
    def __init__(self, model: str = 'socialmedia_en', lleaves: bool = False):
        """
        Loads the models and lookup tables.
        """
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

                warnings.warn(f'unpack {e_lbl} model')

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

    def predict_emojis(
            self, txts: List[str], k: int = 10, sort_by_key: bool = False
            ):
        """
        Predicts emojis based on input text.

        Args:
            txts: Input text.
            k: Number of emojis to predict.
            sort_by_key: If True, the predictions are sorted by their
                confidence.

        Returns:
            A list of tuples with the predicted emojis and their confidence.
        """

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

    def predict(
            self, input: Union[List[str], str],
            dimensions: Optional[List[str]] = None
            ) -> List[str]:
        """
        Predicts emojis, basic emotions, and vad based on input text.

        Args:
            input: Input text.
            dimensions: List of dimensions to predict. If None, all dimensions
                are predicted.

        Returns:
            A dictionary with the predicted values for each dimension.
        """
        dims_ = dimensions if dimensions is not None else self.labels
        txts = [input] if isinstance(input, str) else input

        ft_op = self.predict_emojis(txts, sort_by_key=True, k=151)
        X, _ = zip(*ft_op)

        raw_reg = {
            lbl: pcm.round(
                Nade.clip(
                    self.gb_reg[lbl].predict(list(X)),
                    a_min=0,
                    a_max=1
                ),
                ndigits=3
            )
            for lbl in dims_
        }

        return raw_reg

    @staticmethod
    def preprocess(txts: List[str]) -> List[str]:
        """
        Applies minimal preprocessing to the input text. This includes
        adding whitespace between punctuation and words, reducing multiple
        whitespaces to one, converting text to lower case, and removing
        leading and trailing whitespace.

        Args:
            txts: Input text.

        Returns:
            Preprocessed text.
        """

        # convert to pyarrow array
        txts = pa.array(txts, type=pa.string())

        txts = pcm.replace_substring_regex(
            txts, pattern=r'\s*([\p{P}]+)\s*', replacement=' \\1 '
            )
        txts = pcm.replace_substring_regex(
            txts, pattern=r'\s+', replacement=' '
            )
        txts = pcm.utf8_lower(txts)
        txts = pcm.utf8_trim_whitespace(txts)

        return txts

    @staticmethod
    def clip(arr: List[float], a_min: float = 0.0, a_max: float = 1.0):
        """
        Mimics np.clip using pyarrow.

        Clips arr into a range between a_min and a_max.

        Args:
            arr: Input array.
            a_min: Minimum value.
            a_max: Maximum value.

        Returns:
            Clipped array.
        """
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
            """
            Sorts a single prediction.

            Args:
                x: A tuple with the confidence and the label.

            Returns:
                A tuple with the sorted confidence and the sorted label.
            """
            c, lbl = x

            sorting = pcm.sort_indices(lbl)
            return (c.take(sorting), lbl.take(sorting))

        return map(sort_single, preds)
