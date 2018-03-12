# Remote Sensing Machine Learning with Python

Here are a few algorithms I use whenever I am doing some remote sensing image processing. Most notably the `regression.py` file includes a SimpleR class that allows one to test out some of the noteable machine learning algorithms within the [scikit-learn](http://scikit-learn.org/stable/supervised_learning.html#supervised-learning) library. I would like to include more state-of-the-art algorithms in the future.

# Prerequisites

```
numpy==1.14.1
pandas==0.22.0
scikit-learn==0.19.1
scipy==1.0.0
```

# Installing

Create a new environment with Python 3.6:

```
conda create -n python=3.6
```

Install the requirements using pip from the `requirements.txt`:

```
pip install -r requirements.txt
```

# Acknowledgements

* IPL-UV: SimpleR Package - [github](https://github.com/IPL-UV/simpleR)
* IPL-UV: MLR for Ocean Parameters Retrieval - [github](https://github.com/IPL-UV/mlregocean)
