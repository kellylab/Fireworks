#%%

from fireworks import PyTorch_Model, Message, HookedPassThroughPipe, Experiment
from fireworks.toolbox import ShufflerPipe, TensorPipe, BatchingPipe, FunctionPipe
from fireworks.toolbox.preprocessing import train_test_split
from fireworks.extensions import IgniteJunction
from fireworks.core import PyTorch_Model
import pandas as pd
import torch
from torchvision.datasets.mnist import FashionMNIST
from os import environ as env
from itertools import count
import matplotlib.pyplot as plt 
import visdom

env_name = 'mnist_fashion'
# vis = visdom.Visdom(env=env_name) # If you have a running Visdom server, you can uncomment this to generate plots.
description = "Here, we will train a convolutional neural network on the Fashion MNIST dataset to demonstrate the usage of Fireworks."
experiment = Experiment(env_name, description=description)
#%%

mnist_dir = env.get('MNIST_DIR', './MNIST/')
print(mnist_dir)

# First, we download our dataset and plot one of its elements as an example.
mnist = FashionMNIST(mnist_dir, download=True)
dataset = Message({'examples': mnist.data, 'labels': mnist.targets})
example = dataset['examples'][0]
plt.imshow(example)
plt.show()

#%%
# Now we construct our training and test sets as a pipeline.
train, test = train_test_split(dataset, test=.1)

# We can compose pipes to create an input pipeline that will shuffle the training set on each iteration and produce minibatches formatted for our image classifier.
shuffler = ShufflerPipe(train)
minibatcher = BatchingPipe(shuffler, batch_size=100)
to_cuda = TensorPipe(minibatcher, columns=['examples', 'labels']) # By default, all columns will be moved to Cuda if possible, but you can explicitly specify which ones as well

def tensor_to_float(batch, column='examples'):
    """ This converts the images from bytes to floats which is the data type that torch.nn.Conv2d expects. """
    batch[column] = batch[column].float()
    return batch
 
def reshape_batch(batch, column='examples'):
    """ This reshapes the batch to have an extra dimension corresponding to the input channels so we can apply the torch.nn.Conv2d operation in our model. """
    shape = batch[column].shape
    new_shape = torch.Size([shape[0], 1, shape[1], shape[2]])
    batch[column] = batch[column].reshape(new_shape)
    return batch

def normalize_batch(batch, column='examples'):
    """ Normalizes pixel intensities to fall between 0 and 1. """
    batch[column] /= 255. 
    return batch

to_float = FunctionPipe(input=to_cuda, function=tensor_to_float)
normalized = FunctionPipe(input=to_float, function=normalize_batch)
training_set = FunctionPipe(input=normalized, function=reshape_batch)

# We can also compose a pipeline in one go like we do here for the test set.
test_set = \
    FunctionPipe(
        input=FunctionPipe(
            input=FunctionPipe(
                input=TensorPipe(
                    input=BatchingPipe(
                        input=test,
                        batch_size=100
                        ), 
                    columns=['examples', 'labels']
                ), 
                function=to_float
            ),
            function=normalize_batch
        ),
        function=reshape_batch
    )

#%%

# Construct Model
class mnistModel(PyTorch_Model):
    """ Embeds each image into a 10-dimensional vector. """
    required_components = ['in_column', 'out_column', 'conv1', 'pool1', 'conv2', 'pool2']

    def init_default_components(self):

        self.components['in_column'] = 'examples'
        self.components['out_column'] = 'embeddings'
        self.components['conv1'] = torch.nn.Conv2d(1, 64, 2, padding=1)
        self.components['pool1'] = torch.nn.MaxPool2d(2)
        self.components['conv2'] = torch.nn.Conv2d(64, 32, 2)
        self.components['pool2'] = torch.nn.MaxPool2d(2)
        self.components['nonlinearity'] = torch.nn.ELU()

    def forward(self, batch):
        
        embedding = batch[self.in_column]
        embedding = self.nonlinearity(self.conv1(embedding))
        embedding = self.pool1(embedding)
        embedding = self.nonlinearity(self.conv2(embedding))
        embedding = self.pool2(embedding)
        embedding = embedding.reshape(len(batch), 1152)
        batch[self.out_column] = embedding
        return batch

class Classifier(PyTorch_Model):
    """ Uses the input embedding to perform a classification. """    
    required_components = ['in_column', 'out_column', 'linear_layer']

    def init_default_components(self):
        self.components['in_column'] = 'embeddings'
        self.components['out_column'] = 'predictions'
        self.components['linear1'] = torch.nn.Linear(1152, 256)
        self.components['linear2'] = torch.nn.Linear(256, 10)
        self.components['nonlinearity'] = torch.nn.ELU()
        self.components['softmax'] = torch.nn.Softmax(dim=1)

    def forward(self, batch):
        
        predictions = batch[self.in_column]
        predictions = self.nonlinearity(self.linear1(predictions))
        predictions = self.softmax(self.linear2(predictions))        
        batch[self.out_column] = predictions
        return batch 

# All function calls to the classifier will call the embedder first
# ie. classifier(x) is equivalent to classifier.forward(embedder.forward(x))
embedder = mnistModel()
classifier = Classifier(input=embedder) 

if torch.cuda.is_available():
    embedder.cuda()
    classifier.cuda()
#%%

# Set up loss function and training loop
ce_loss = torch.nn.CrossEntropyLoss()
loss = lambda batch: ce_loss(batch['predictions'], batch['labels'])

# By default, this Junction applies a standard training closure of evaluating the model,
# computing gradients of the loss, and backpropagating using the chosen optimizer.
trainer = IgniteJunction(
    components={
        'model': classifier, 
        'dataset': training_set
        }, 
    loss=loss, optimizer='Adam', 
    lr=.0001, weight_decay=.001, 
    visdom=False, # If you have a running Visdom server, you can set this to true to plot training loss over time.
    environment=env_name
    )

trainer.run(max_epochs=10) # This will take almost 20 minutes on CPU and around 1 minute on GPU

#%%
# Now that we've trained our model, we can compute some metrics on the test set.
# Here, we construct a Pipe that will compute metrics such as sensitivity, specificity, f1, etc.
# on the test set.

classes = {i: class_name for i, class_name in zip(count(), mnist.classes)}
class Metrics(HookedPassThroughPipe):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.true_positives = {class_name: 0 for class_name in classes.values()}
        self.true_negatives = {class_name: 0 for class_name in classes.values()}
        self.false_positives = {class_name: 0 for class_name in classes.values()}
        self.false_negatives = {class_name: 0 for class_name in classes.values()}
        self.total_count = 0
        self.label_counts = {class_name: 0 for class_name in classes.values()}
        self.prediction_counts = {class_name: 0 for class_name in classes.values()}

    def _call_hook(self, batch):
        """ 
        This will get called every time the model is called. As a result, this pipe will continuously update
        itself as we iterate through the test set.
        """
        labels = batch['labels']
        predictions = torch.max(batch['predictions'],1)[1]
        correct_indices = (predictions == labels).nonzero().flatten().tolist()
        incorrect_indices = (predictions != labels).nonzero().flatten().tolist()
        for index, name in classes.items():
            self.label_counts[name] += int(sum(labels == index)) # How often the class showed up
            self.prediction_counts[name] += int(sum(predictions == index)) # How often the class was predicted
            self.true_positives[name] += int(sum(predictions[correct_indices] == index)) # How often the correct prediction was for thsi class
            self.true_negatives[name] += int(sum(predictions[correct_indices] != index)) # How often the correct prediction was not for the class; ie. how often the prediction was a true negative for this class
            self.false_positives[name] += int(sum(predictions[incorrect_indices] == index)) # How often a wrong prediction was for this class
            self.false_negatives[name] += int(sum(predictions[incorrect_indices] != index)) # How often a wrong prediction was for another class; ie. how often the prediction was a false negative for this class
        self.total_count += len(batch)
        return batch

    def compile_metrics(self):
        """ 
        After we have gone through the entire test set, we can call this method to compute the actual metrics.
        """
        class_names = classes.values()
        negative_counts = {name: sum(self.label_counts[other] for other in class_names if other != name) for name in class_names}
        self.sensitivity = {name: self.true_positives[name] / self.label_counts[name] for name in class_names}
        self.specificity = {name: self.true_negatives[name] / negative_counts[name] for name in class_names}
        negative_prediction_counts = {name: sum(self.prediction_counts[other] for other in class_names if other != name) for name in class_names}
        self.ppv = {name: self.true_positives[name] / self.prediction_counts[name] for name in class_names}
        self.npv = {name: self.true_negatives[name] / negative_prediction_counts[name] for name in class_names}
        self.f1 = {name: 2 / (1/self.ppv[name] + 1/self.sensitivity[name]) for name in class_names}
        self.accuracy = {name: (self.true_positives[name] + self.true_negatives[name]) / self.total_count for name in class_names}

    def get_metrics(self):
        """
        Lastly, we will use this method to return the computed metrics as a Pandas DataFrame.
        """
        columns = ['sensitivity', 'specificity', 'ppv', 'npv', 'f1', 'accuracy']
        df = pd.DataFrame(columns=columns, index=classes.values())
        for attribute in columns:
            value = getattr(self, attribute)
            df[attribute] = [value[key] for key in df.index]

        return df

# This class is implemented a a HookedPassThroughPipe, meaning that it's _call_hook method will be applied every time
# The class is called like a function, and this call will pass through to its input.
metrics_computer = Metrics(input=classifier)

for batch in test_set:
    # We can simply call this object repeatedly on batches in the test set
    # This operation is equivalent to metrics_computer._call_hook(classifier(batch))
    metrics_computer(batch)

metrics_computer.compile()
df = metrics_computer.get_metrics()
print(df)

# You can also convert this DataFrame to a Message.
m = Message(df)
print(m)

# Lastly, we can save our results from this experiment
# At it's simplest, the experiment object gives you a way of organizing files.
# You can uses its open() method to get a file handle or path string to a file 
# inside its experiment directory.
# Each time you create an experiment, a new experiment directory will be created automatically.
df.to_csv(experiment.open("metrics.csv", string_only=True))
# Since our models are still subclasses of torch.nn.module, we can save them using the standard torch.save feature
# but if we want, we can also save their parameters in other formats such as JSON
state = embedder.get_state()
Message.from_objects(state).to('json', path=experiment.open("embedder.json",string_only=True))
state = classifier.get_state()
Message.from_objects(state).to('json', path=experiment.open("classifier.json",string_only=True))
