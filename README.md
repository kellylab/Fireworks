[![Build Status](https://travis-ci.org/kellylab/fireworks.svg?branch=master)](https://travis-ci.org/kellylab/fireworks)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/kellylab/Fireworks/mnistDemo?filepath=examples%2Fmnist.ipynb)
[![status](http://joss.theoj.org/papers/6156356281b38760548fdb14505d9ba1/status.svg)](http://joss.theoj.org/papers/6156356281b38760548fdb14505d9ba1)
[![DOI](https://zenodo.org/badge/135873580.svg)](https://zenodo.org/badge/latestdoi/135873580)


# Fireworks - PyTorch with DataFrames

Introduction
=====================================

This library provides an implementation of DataFrames which are compatible with PyTorch Tensors. This means that you can construct models that refer to their inputs by column name, which makes it easier to keep track of your variables when working with complex datasets. This also makes it easier to integrate those models into your existing Pandas-based data science workflow.
Additionally, we provide a complete machine learning framework built around DataFrames to facilitate model training, saving/loading, preprocessing, and hyperparameter optimization.

Overview
=====================================

Data is represented in Fireworks using an object called a Message, which generalizes the concept of a DataFrame to include PyTorch tensors (analogous to a TensorFrame in other frameworks). This data structure is popular in statistical research because of how well it organizes information while remaining flexible enough to facilitate any statistical analysis. By providing an implementation of a DataFrame that supports torch.Tensor objects, we can now use this data structure with PyTorch. You can easily to move columns in a Message to and from torch.Tensor and to and from the GPU all within one object.

We provide a set of abstract primitives (Pipes, Junctions, and Models) that can be be used to implement specific operations and can be stacked together to construct a data pipeline. Because of the standardization of input/output that these primitives expect, these components are reusable. Rather than constructing a new data pipeline in an ad-hoc manner for every project, you can modularly compose your pipeline using existing components provided by fireworks or that you have made previously.

This library also provides a set of tools built around these components. There are modules here for training machine learning models, reading and writing to databases using Messages, hyperparameter optimization, and saving/loading snapshots/logs of your data pipeline for re-usability and reproducibility.

# Getting Started
Installation
=====================================
You can install fireworks from PyPI:

    pip3 install fireworks-ml

Documentation
=====================================
See documentation at https://fireworks.readthedocs.io

# Contributing

Comments, questions, issues, and pull requests are always welcome! Feel free to open an issue with any feedback you have or reach out to me (smk508) directly with any questions. See our roadmap for an overview of features that we are looking to add (https://fireworks.readthedocs.io/en/development/Project.html#roadmap).

# Acknowledgements
Development
=====================================
fireworks was developed by Saad Khan, an MSTP student in the lab of Libusha Kelly at the Albert Einstein College of Medicine (https://www.kellylab.org/). We use this library to develop deep learning models to study the microbiome.

Funding
=====================================
Saad Khan is funded in part by an NIH MSTP training grant 6T32GM007288-46. This work was funded in part by a Peer Reviewed Cancer Research Program Career Development Award from the United States Department of Defense to Libusha Kelly (CA171019).
