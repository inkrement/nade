# README

```bash
pip install git+git://github.com/inkrement/nade.git
```


```python
from nade import Nade

n = Nade()
n.predict('I love this')
```

```json
{
 'anger': 0.004,
 'anticipation': 0.15,
 'disgust': 0.017,
 'fear': 0.027,
 'joy': 0.451,
 'sadness': 0.02,
 'surprise': 0.142,
 'trust': 0.242
}```
