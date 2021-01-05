#from nade import Nade
import nade


def test_loading():
    n = nade.Nade()
    
    return True

def test_emojiPrediction():
    n = nade.Nade()
    pred_emojis = n.emoji_predict('test')

    assert len(n.emojis) == len(pred_emojis)