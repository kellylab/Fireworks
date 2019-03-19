Tutorial
=================

Messages
------------------------------

      For the most part, you can use Messages like dataframes. That is, you can call Fireworks.Message() instead of pd.DataFrame(). There are a few key differences.
      First, Messages don't have a way to adjust their index. In Pandas, you can choose how the rows of a DataFrame are indexed. For example, the rows could refer
      to timestamps or dates. Here, the only way to index rows is by number (ie. the nth row is accessed by calling message[n]). This simplifies usage when feeding
      data to a statistical model which only cares about getting the next batch of data.
      Secondly, Messages can have torch.Tensor objects inside them. You can set a column of a Message to a torch.Tensor, and operations like append and indexing will work
      as you expect.

      .. code-block:: python

            a = torch.random(2,3)
            message = Message({'x': a})
            print(a)
            >>  Message with
            >>  Tensors:
            >>  TensorMessage: {'ok': tensor([[ 0.3087,  0.9619,  0.5176],
            >>         [ 0.2747,  0.6640,  0.2813]])}
            >>  Metadata:
            >>  Empty DataFrame
            >> Columns: []
            >> Index: []
            b = torch.random(4,3)
            message = message.append({'x', b})
            print(len(message))
            >> 6
            assert (message[0:2]['x'] == a).all()
            assert (message[2:6]['x'] == b).all()

    Internally, the Message stores torch.Tensors in an object called a TensorMessage, which is analogous to a DataFrame. All other types of data are stored inside
    a DataFrame. You can move data to and from these two formats as well.

    .. code-block:: python

            message = message.to_dataframe(['x']) # Leave blank to move all tensor columns
            print(type(message['x']))
            >> <class 'pandas.core.series.Series'>
            message = message.to_tensors(['x']) # Leave blank to move all dataframe columns
            print(type(message['x']))
            >> <class 'torch.Tensor'>

    You can also move tensors to and from different devices.

    .. code-block:: python

            message = message.cuda(device_num=0, keys=['x']) # Leave blank to default to device 0 and all tensor columns
            message = message.cpu(keys=['x']) # Leave blank to default to all tensor columns

Chaining Pipes and Junctions
------------------------------

    Pipes have a single input, whereas Junctions can have multiple. Having a single input makes recursive method calling well defined; a Pipe can simply refers to its one input.
    On the other hand, there is ambiguity when doing this for multiple inputs. Should all of the inputs be called or only one? What order should they be called? How should the outputs be combined?
    Because of this, Pipes have defined methods for recursive method calling which makes it easy to compose them. Junctions don't have these by default so that the logic can be implemented on
    an individual basis. Let's look at an example of composing Pipes:

    .. code-block:: python

        from Fireworks.toolbox.pipes import BatchingPipe, TensorPipe
        from Fireworks import Message
        import numpy as np

        message = Message({"x": np.random.rand(100)})
        shuffler = ShufflerPipe(input=message)
        minibatcher = BatchingPipe(shuffler, batch_size=25)
        train_set = TensorPipe(minibatcher)
        for batch in train_set:
            print(batch)

    In this example, we start off with some message as the first input. This could alternatively be some Pipe that reads in data from a database, file, etc.
    As we loop through the last pipe, train_set, the following steps occur at each loop:
        - train_set calls for the next element from its input, minibatcher.
        - minibatcher calls for the next 25 elements from its input, shuffler. This corresponds to the next batch.
        - shuffler return 25 elements from its input, message, to minibatcher. The elements are randomly chosen based on a precomputed shuffle that resets on every full loop through the dataset.
        - minibatcher returns the 25 elements to its output, train_set
        - train_set converts the columns of its 25 element batch to torch.Tensors and returns the batch. If cuda is installed and enabled, this also moves those tensors to the GPU.

    During each step of this process, the elements being returned are Messages. Because of this, the Pipes are decoupled and be re-used and re-composed. Pipes upstream in the pipeline can handle
    'formatting' tasks such as reading in the data and constructing batches, whereas more downstream pipes can perform preprocessing transformations on those batches, such as normalization, vectorization,
    and moving data to the GPU. Each successive Pipe adds an additional layer of abstraction, and from the perspective of a downstream Pipe, the input is the only thing that it needs to worry about.

    Lets look at a more involved example involving Junctions:

    .. code-block:: python

        from Fireworks.toolbox.pipes import LoopingPipe, CachingPipe, ShufflerPipe, BatchingPipe, TensorPipe
        from Fireworks.toolbox.junctions import RandomHubJunction
        import numpy as np
        from Fireworks import Message

        a = Message({'x': np.random.rand(100)})
        b = Message({'x': np.random.rand(200)})
        sampler = RandomHubJunction(components={'a':a, 'b':b})
        looper = LoopingPipe(sampler)
        cache = CachingPipe(looper, cache_size=1000)
        shuffler = ShufflerPipe(cache)
        minibatcher = BatchingPipe(shuffler, batch_size=30)
        train_set = TensorPipe(minibatcher)

        for batch in train_set:
            print(batch)

    The RandomHubJunction randomly samples elements from its multiple inputs during iteration. Here, we use this to combine two different data sets (a and b) into a single stream. The RandomHubJunction can only
    iterate through its inputs in a forward direction; you can't access items by index (eg. sampled[n]). The LoopingPipe creates the illusion of this functionality by moving the iteration forwards in order to get
    a requested element. For example, you can call looper[20], and this will return the element that is returned by iterating through sampler 20 times. The CachingPipe stores this information in memory as the name implies.
    This can be useful for working with extremely large datasets or inputs that are expensive to produce.


Using Databases
------------------------------

    It's often useful to use a database query as the starting point for a pipeline, or to write data in the form of a Message into a database.
    The database module is built on top of SQLalchemy and facilitates this. Let's say you have a SQL alchemy table (a subclass of declarative_base)
    which describes your schema and an engine object which can connect to the database. You can create a TablePipe which can query this database
    as follows:

    .. code-block:: python

       from Fireworks.extensions.database import TablePipe

       db = TablePipe(table, engine)
       query_all = db.query()
       for row in query_all:
           print(row) # This will print every column in the table as Messages
        query_some = db.query(['column_1','column_2'])
        for row in query_some:
           print(row) # This will print only 'column_1' and 'column_2'

      When you use the query method, the object returned is a DBPipe. This can serve as input to a pipeline, as it is iterable.
      Additionally, you can make your query more precise by applying filters which apply predicates. This lets you make
      queries of the form "SELECT a FROM b WHERE c" and so on.

      .. code-block:: python

         filtered = query_some.filter('column_n', 'between', 5,9) # "SELECT column_1, column_2 FROM table WHERE column_n BEWEEN 4 AND 8"
         filtered.all() # Returns the entire query as a single Message

      The allowed predicates correspond to the allowed filters in SQLalchemy (see https://docs.sqlalchemy.org/en/latest/orm/query.html#the-query-object for more information.)
      You can also insert Messages into this table, assuming the column names and data types align with the table's schema. Along with this,
      you can rollback and commit operations.

      .. code-block:: python

         db.insert(message)
         db.rollback() # The insertion will be undone
         db.insert(message)
         db.commit() # The transaction will be committed to the db



Saving and Loading
------------------------------

    All of the core structures in Fireworks have methods for serializing their state, which makes it straightforward to save and load Pipes, Junctions, and Models. On any of these objects, you can call the get_state()
    method to get a dictionary-serialized representation of their current state. You can also call set_state() to update this state.

    .. code-block:: python

    The returned state object consists of two dictionaries, internal and external.

    Not all pipes keep strict track of internal state. Thus, many Pipes may return an empty dictionary when get_state is called.

    - Experiment example
    - Scaffold example

Model Training
------------------------------



Hyperparameter Optimization
-------------------------------
    - Explain what this is
    - Factory example
