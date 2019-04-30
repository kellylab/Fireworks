[![Build Status](https://travis-ci.org/kellylab/Fireworks.svg?branch=development)](https://travis-ci.org/kellylab/Fireworks)

# Fireworks
Introduction
=====================================

The goal of Fireworks is to make machine learning research more reproducible and consistent by providing tools to construct machine learning pipelines (the process of going from a raw dataset to a model to training that model, saving checkpoints, and analyzing its results) in a modular and structured manner.

Specifically, Fireworks is a batch-processing framework for Python and PyTorch. It is meant to provide an easy method to stream data from a dataset into a machine learning model while performing preprocessing steps such as randomization, train/test split, batch normalization, etc. along the way. Fireworks offers more flexibility and structure for constructing input pipelines than the built-in dataset modules in PyTorch, but is also meant to be easier to use than frameworks such as Apache Spark. In particular, Fireworks makes it easy to move objects to and from torch.Tensor and to and from the GPU, which is not natively possible in other batch-processing frameworks.

Overview
=====================================

Data is represented in Fireworks using an object called a Message, which generalizes the concept of a DataFrame to include PyTorch tensors (analogous to a TensorFrame in other frameworks). This data structure is popular in data science because of how well it organizes information while remaining flexible enough to facilitate any statistical analysis. By providing an implementation of a DataFrame that supports torch.Tensor objects, we can now use this data structure with PyTorch. You can easily to move columns in a Message to and from torch.Tensor and to and from the GPU all within one object.

Fireworks provides a set of abstract primitives (Pipes, Junctions, and Models) that can be be used to implement specific operations and then be stacked together to construct a data pipeline. Because of the standardization of input/output that these primitives expect, these components are reusable. Rather than constructing a new data pipeline in an ad-hoc manner for every project, you can modularly compose your pipeline using existing components provided by Fireworks or that you have made previously.

This library also provides a set of tools built around these components. There are modules here for training machine learning models, reading and writing to databases using Messages, hyperparameter optimization, and saving/loading snapshots/logs of your data pipeline for re-usability and reproducibility.

# Getting Started
Installation
=====================================
You can install Fireworks from PyPI:

    pip3 install fireworks-ml

Documentation
=====================================
See documentation at https://fireworks.readthedocs.io

# Contributing

Comments, questions, issues, and pull requests are always welcome! Feel free to open an issue with any feedback you have or reach out to me (smk508) directly with any questions. See our roadmap for an overview of features that we are looking to add (https://fireworks.readthedocs.io/en/development/Project.html#roadmap).

# Acknowledgements
Development
=====================================
Fireworks was developed by Saad Khan, an MSTP student in the lab of Libusha Kelly at the Albert Einstein College of Medicine (https://www.kellylab.org/). We use this library to develop deep learning models to study the microbiome.

Funding
=====================================
Saad Khan is funded in part by an NIH MSTP training grant 6T32GM007288-46. This work was funded in part by a Peer Reviewed Cancer Research Program Career Development Award from the United States Department of Defense to Libusha Kelly (CA171019).
