# Readme

**try it: [https://nade.rds.wu.ac.at](https://nade.rds.wu.ac.at)**

Natural affect detection allows to infer basic emotions from social media messages. While human raters are often too resource-intensive, lexical approaches face challenges regarding incomplete vocabulary and the handling of informal language. Even advanced machine learning-based approaches require substantial resources (expert knowledge, programming skills, annotated data sets, extensive computational capabilities) and tend to gauge the mere presence, not the intensity, of emotion. This package (NADE) solves this issue by predicting a vast array of emojis based on the surrounding text, then reduces these predicted emojis to an established set of eight basic emotions.


![Architecture](https://raw.githubusercontent.com/inkrement/nade/main/docs/overview.png)


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

## Performance

The prediction method features a _lleaves_ option that provides much faster inference. However, you will have to install [lleaves](https://github.com/siboehm/lleaves) first.

## Links

 - [Nade Explorer](https://nade.rds.wu.ac.at)
 - Paper: coming soon
