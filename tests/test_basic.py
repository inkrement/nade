#from nade import Nade
import nade


def test_loading():
    n = nade.Nade()
    
    return True

def test_emojiPrediction():
    n = nade.Nade()
    cred_pa, labels = n.predict_emojis('test', k = n.max_k)[0]

    assert len(n.emojis) == len(cred_pa)
