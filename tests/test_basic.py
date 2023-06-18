import nade


def test_loading():
    n_init = nade.Nade()
    assert n_init is not None


def test_emojiPrediction():
    n = nade.Nade()
    cred_pa, labels = n.predict_emojis('test', k=n.max_k)[0]

    assert len(n.emojis) == len(cred_pa)
