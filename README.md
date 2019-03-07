# Fireworks
Introduction
=====================================

Fireworks is a python-first batch-processing framework for performing the data processing steps of machine learning in a modular and reusable manner. It provides a set of core components (Message, Pipes, Junctions, and Models) that can be used to implement different operations and can be joined together to construct a data science workflow along with a set of tools built around these components. Data is represented using an object called a Message, which generalizes the concept of a DataFrame to include PyTorch tensors (analogous to a TensorFrame in other frameworks), and there are modules here for training machine learning models, reading and writing to databases using Messages, hyperparameter optimization, and saving/loading snapshots and logs of your data pipeline for re-usability and reproducibility.

Overview
=====================================

Fireworks consists of a number of modules that are designed to work together to facilitate an aspect of deep learning and data processing.

**Message**

 “A dictionary of vectors and tensors”. This is a generalization of the idea of a DataFrame to be able to include PyTorch Tensors (this has been called a TensorFrame in other frameworks). This class also standardizes the means of communication between objects in Fireworks. Standardizing the means of communication makes it easier to write models that are reusable, because the inputs and outputs are always the same format.

**Pipe**

A class that abstracts data access and transformation. Pipes can be linked together, allowing one to modularly construct a data pipeline. When data is accessed from such a pipeline, each Pipe will apply its own transformation one at a time and return the resulting output. There are numerous premade Pipes in Fireworks.toolbox which implement common preprocessing tasks.

**Junction**

Whereas pipes can only have one input and are meant to represent a linear flow of information, Junctions extend that functionality to allow multiple inputs, enabling more complex information flows. These inputs are called components, and each Junction can implement its own logic for how it uses its components to return data that is requested. There are premade Junctions in Fireworks.toolbox which implement common tasks related to combining or samplign from multiple data sources.

**Models**

Models represent parameterizable transformations. This differentiates them from Pipes, as there is additional functionality in place to facilitate saving, loading, and swapping around the parameters of a Model that go beyond what a typical Pipe would require. Additionally, Models have methods specifically relevant to the statistical models that they are meant to represent. In particular, PyTorch_Models are designed to represent machine learning models in the form of PyTorch Modules, and this allows one to insert the Modules into a pipeline.

**MessageCache**

This class addresses the particular challenge of dealing with datasets that won’t fit in memory. A MessageCache behaves like a python cache that supports insertions, deletions, and accessions, except the underlying data structure is a message. This enables one to hold a portion of a larger dataset in memory  while virtually representing the entire dataset as a message.

**Hyperparameter Optimization**

This module takes an approach similar to Ignite, except for hyperparameter optimization. It provides a Junction class called Factory that has methods corresponding to the events in a hyperparameter training process (train, evaluate, decide on new parameters, etc.) that can be provided by the developer. In addition, training runs are treated as independent processes, enabling one to spawn multiple training runs simultaneously to evaluate multiple hyperparameters at once.

**Relational Database Integration**

Fireworks.database has Pipes that can read from and write to a database. Because it is based on python's SQLalchemy library, it can be used to incorporate almost any relational database into a data analysis workflow.

**Collecting and Storing Experimental Runs / Metrics**
In order to make machine learning research reproducible, we have to be able to store metadata and outputs associated with experiments. Fireworks.experiment implements an Experiment class that creates a folder and can generate file handles and SQLite tables residing in that folder to save information to. It can also store user defined metadata, all in a given experiment’s folder. This folder can be reloaded at any time in order to access the results of that experiment, regenerate plots, perform additional analyses, and so on.
Additionally, it is possible to save and load the state of an entire computation graph using the Fireworks.scaffold module. Pipes, Junctions, and Models each implement methods for serializing and de-serializing their internal state. These methods can be customized by the user and are compatible with PyTorch's built-in save mechanisms. This allows you to save not just the internal parameters of a Model, but also a snapshot of the data used to train the model and the state of whatever preprocessing operations were performed, along with the outputs and logs that were produced. By saving the entire pipeline in this manner, it's easier to reason about reproducibility of experiments.


**Not Yet Implemented / Roadmap Objectives**

**Plotting**

Generating plots is the primary means for analyzing and communicating the results of an experiment. We want to generate plots in such a way that we can go back later on and change the formatting (color scheme, etc.) or generate new plots from the data. In order to do this, plots must be generated dynamically rather than as static images. In addition, we want to create a robust means for displaying plots using a dashboard framework such as Plotly Dash. Tools such as Visdom are great for displaying live metrics from an experiment, but they are not designed to present hundreds of plots at once or to display those plots in a pleasing manner (such as with dropdown menus).
My goal is to create a dashboard that can display all information associated with a chosen experiment. It should include an SQLite browser, a plotly dashboard, a Visdom instance, and a Tensorboard instance. This should allow one to have all of the common visualization tools for machine learning in a single place.

**Dry Runs**

Certain steps in a data processing pipeline can be time consuming one-time operations that make it annoying to repeatedly start an experiment over. We want to set up means to ‘dry run’ an experiment in order to identify and fix bugs before running it on the full dataset. Additionally, we want to be able to set up checkpoints that enable one to continue an experiment after pausing it or loading it from a database.

**Distributed and Parallel Data Processing**

The idea of representing the data processing pipeline as a graph can naturally scale to parallel and distributed environments. In order to do make this happen, we need to write a scheduler that can call methods on Sources in parallel and add support for asynchronous method calls on Sources. For distributed processing, we have to write a tool for containerizing Sources and having them communicate over a container orchestration framework. Argo is a framework for Kubernetes we can use that is designed for this task.

**Dynamic Optimization of Data Pipeline**

Many sources have to occasionally perform time consuming O(1) tasks (such as precomputing indices corresponding to minibatches). Ideally, these tasks should be performed asynchronously, and the timing of when to perform them should be communicated by downstream sources. Adding the ability to communicate such timings would allow the pipeline to dynamically optimize itself in creative ways. For example, a CachingSource could prefetch elements into its cache that are expected to be called in the future to speed up its operation.

**Static Performance Optimization**

Right now, the focus is on establishing the interface and abstractions associated with Fireworks. There are many places where operations can be optimized using better algorithms, cython implementations of important code sections, and eliminating redundant code.
