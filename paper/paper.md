---
title: "Fireworks: Reproducible Machine Learning and Preprocessing with PyTorch"
tags:
    - Python
    - PyTorch
    - batch processing
    - machine learning
authors:
    - name: Saad M. Khan
      orcid: 0000-0003-4686-4923
      affiliation: "1"
    - name: Libusha Kelly
      orcid:
      affiliation: "1"
affiliations:
    - name: Systems & Computational Biology, Albert Einstein College of Medicine
date: 23 April 2019
bibliography: paper.bib
---

# Summary

Here, we present a batch-processing library that facilitates pre- and post-processing operations common to machine learning pipelines. Steps in a pipeline are represented using objects that can be composed in a reusable manner with a standard input and output. These input/outputs are performed via a DataFrame-like object called a Message which can have PyTorch [@paszke2017automatic] Tensor-valued columns in addition to all of the functionality of a Pandas DataFrame [@mckinney-proc-scipy-2010]. Each of these pipeline objects can be serialized, enabling one to save checkpoints of the entire state of their workflow rather than just an individual model's. Additionally, we provide a suite of extensions and tools built on top of this library to facilitate common tasks such as relational database access, model training, hyperparameter optimization, saving experimental runs, and more.

A pain point in all deep learning frameworks is data entry; one has to take data in its original form and convert it into the form that the model expects (tensors, TFRecords [@tensorflow2015-whitepaper], etc.). The data conversion process can include acquiring the data from a database or an API call, formatting it, applying preprocessing transforms, and preparing mini batches for training. In each of these steps, one must be cognizant of memory, storage, bandwidth, and compute limitations. For example, if a dataset is too big to fit in memory, then it must be streamed or broken into chunks. In practice, dealing with these issues can take more time than developing the actual models. Moreover, ad-hoc solutions to these challenges often introduce biases. For example, if one chooses not to shuffle their dataset before sampling it due to its size, then the order of elements in that dataset becomes a source of bias. Ideally, proper statistical methodology should not have to be compromised for performance and memory considerations, but in practice, that often becomes the case.

We developed Fireworks to be an easy-to-use solution to these limitations. Fireworks is a python-first framework for performing the data processing steps of machine learning in a modular and reusable manner. Fireworks moves data between objects called ‘Pipes’. A Pipe can represent a file, a database, or a data transform. Each Pipe has an input Pipe and is itself an input to other Pipes, and as data flows from Pipe to Pipe, transforms are applied one at a time, creating a graph of data flow. Because each Pipe is independent, Pipes can be stacked and reused. Moreover, because Pipes are aware of their inputs, they can also call methods on their inputs, and this enables just-in-time evaluation of data transformations (a downstream Pipe could call a method implemented by an upstream Pipe. Lastly, communication between Pipes is represented by a Message object. A Message is essentially a Python dictionary of arrays, lists, vectors, tensors, etc. Messages generalize the functionality of a Pandas DataFrame to include the ability to store PyTorch tensors [@mckinney-proc-scipy-2010]. One can thus adapt traditional ML pipelines that use Pandas , because Messages behave like DataFrames. As a result, Fireworks is useful for any data processing task, and not just deep learning. It can be used for interacting with a database and constructing statistical models with Pandas as well.

[Figure 1: Illustration of the core primitives in Fireworks](Models.png)

Fireworks aims to enable reproducible and robust machine learning research by providing abstract primitives that can be used to represent the steps of a machine learning pipeline. All of these features are designed to improve the productivity of researchers and the reproducibility of their work by reducing the amount of time spent on these tasks and providing a standardized means to do so. We are currently using this library in our lab to implement neural network models for studying the human microbiome and for finding phage genes within bacterial genomes. In the future, we hope to implement more integration with other popular data science libraries in order to facilitate its inclusion in common data workflows. For example, distributed data processing systems such as Apache Spark [@spark] could be used to acquire initial data from a larger data-set that can then be pre-processed and mini-batched using Fireworks to train a PyTorch Model. The popular python library SciKit-Learn [@scikit-learn] also has numerous pre-processing tools and machine learning models that could be integrated with Fireworks so that they could be used within a Pipeline. There is also a growing interest in doing data science research on the cloud in a distributed environment. There are now libraries such as KubeFlow [@kubeflow_2019] which enable one to spawn experiments onto a cloud environment so that multiple runs can be performed and tracked simultaneously. In the future, we will look to integrate these libraries and features with Fireworks.

The overall design of this library is aimed at transparency and backwards-compatibility with respect to the underlying libraries. The goal is easy incorporation of Fireworks into a project without requiring one to change the other aspects of a workflow. Our library helps researchers implement cleaner machine learning pipelines in a more reproducible and reusable manner to enhance the scientific rigor of their work.

# References
