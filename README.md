[![Downloads](https://static.pepy.tech/personalized-badge/gale-topo?period=total&units=international_system&left_color=grey&right_color=blue&left_text=Downloads)](https://pepy.tech/project/gale-topo) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/16JhR2nhm9J-9KtqmuxGsKMIGL_5c_42N?usp=sharing) [![Paper](https://img.shields.io/badge/read%20the-paper-blueviolet)](https://arxiv.org/pdf/2201.02155.pdf) [![Build](https://github.com/pnxenopoulos/gale/actions/workflows/build.yml/badge.svg)](https://github.com/pnxenopoulos/gale/actions/workflows/build.yml) [![Documentation Status](https://readthedocs.org/projects/gale/badge/?version=latest)](https://gale.readthedocs.io/en/latest/?badge=latest) [![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/pnxenopoulos/gale/blob/main/LICENSE)

# Globally Assessing Local Explanations (GALE)
GALE is a Python library used to assess the similarity of local explanations from methods such as LIME, SHAP or generalized additive models (GAMs). To do so, GALE models the relationship between the explanation space and the model predictions as a scalar function. Then, we compute the topological skeleton of this function. This topological skeleton acts as a signature, which we use to compare outputs from different explanation methods. 

### Install
GALE is easy to install and use. Simply run 

```shell
pip install gale-topo
```

and you're good to go!

### Usage
You can measure the similarity between sets of explanations in just a few lines of code, shown below

```python
from gale import create_mapper, bottleneck_distance

model_one_mapper = create_mapper(explanations_one, predictions_one, resolution=10, gain=0.3, dist_thresh=0.5)
model_two_mapper = create_mapper(explanations_two, predictions_two, resolution=10, gain=0.3, dist_thresh=0.5)

bottleneck_distance(model_one_mapper, model_two_mapper)  # This returns a float which represents the distance between the two Mapper outputs
```

### Need Help?
Need help? Open up an [issue](https://github.com/pnxenopoulos/gale/issues).

### Citation
GALE was published at TAGML 2022, a workshop at ICML 2022. If you use GALE in your work, please cite the following

```
latex citation goes here
```
