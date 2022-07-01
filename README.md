# Readme

Natural affect detection allows to infer basic emotions from social media messages. 

![Architecture](https://raw.githubusercontent.com/inkrement/nade_py/main/docs/architecture.png)


## Usage
After installation, the module can be loaded and the predict method can be used for inference.

```python
from nade import Nade

n = Nade()
n.predict('I love this')
```

The method returns a dictionary containing the scores for all eight basic emotions.

```python
{
 'anger': [0.004],
 'anticipation': [0.15],
 'disgust': [0.017],
 'fear': [0.027],
 'joy': [0.451],
 'sadness': [0.02],
 'surprise': [0.142],
 'trust': [0.242]
}
```


## Installation

The package can be installed as follows:

```bash
pip install git+git://github.com/inkrement/nade.git
```

