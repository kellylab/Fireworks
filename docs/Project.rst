Project
=================

History
----------
Saad Khan began working on Fireworks in June 2018. It was open sourced in January 2019.

Committers
----------
 - @smk508 (Saad Khan)

Resources
----------

    - `PyTorch <https://pytorch.org/>`_
    - `Ignite <https://pytorch.org/ignite/>`_


Roadmap
----------

    - Integration with Kubernetes to facilitate running experiments on the cloud. For example, a hyperparameter optimization routine could
      spawn multiple training runs on Kubernetes pods that each report back their results.
    - Performance improvements for caching components
        - Update the CachingPipe to use a B-tree datastructure for improved performance with range queries.
        - Add support for cache hinting. For example, a Pipe that randomly samples from a dataset could indicate to a downstream CachingPipe
          which elements are expected to be sampled next so that the CachingPipe can preload those elements and reduce Cache misses.
    - Abstracting Message object to support additional ml frameworks beyond PyTorch.
    - Refactoring library into smaller, separate packages to reduce installation size.
    - Integration with distributed streaming/computation frameworks such as Apache Spark and HDFS.
    
