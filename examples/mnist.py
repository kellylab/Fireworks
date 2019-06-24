#%%
from fireworks import PyTorch_Model, Message
from fireworks.toolbox import ShufflerPipe, TensorPipe, BatchingPipe, FunctionPipe
from fireworks.toolbox.preprocessing import train_test_split
from fireworks.extensions import IgniteJunction
from fireworks.core import PyTorch_Model
import torch
from torchvision.datasets.mnist import FashionMNIST
from os import environ as env
from itertools import count
import matplotlib.pyplot as plt 
import visdom

env_name = 'mnist_fashion'
vis = visdom.Visdom(env=env_name)
#%%
mnist_dir = env.get('MNIST_DIR', '/var/MNIST/')
print(mnist_dir)

mnist = FashionMNIST(mnist_dir, download=True)
dataset = Message({'examples': mnist.data, 'labels': mnist.targets})
example = dataset['examples'][0]
#plt.imshow(example)
#plt.show()

#%% 
# Annotate examples with class names
classes = {i: class_name for i, class_name in zip(count(), mnist.classes)}

#%%
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

# Construct Model
class mnistModel(PyTorch_Model):
    """ Embeds each image into a 10-dimensional vector. """
    required_components = ['in_column', 'out_column', 'conv1', 'pool1', 'conv2', 'pool2']

    def init_default_components(self):

        self.components['in_column'] = 'examples'
        self.components['out_column'] = 'embeddings'
        self.components['conv1'] = torch.nn.Conv2d(1, 64, 2, padding=1)
        self.components['pool1'] = torch.nn.MaxPool2d(2)
        #self.components['conv2'] = torch.nn.Conv2d(64, 32, 2)
        #self.components['pool2'] = torch.nn.MaxPool2d(3)
        self.components['nonlinearity'] = torch.nn.ELU()

    def forward(self, batch):
        
        embedding = batch[self.in_column]
        embedding = self.nonlinearity(self.conv1(embedding))
        embedding = self.pool1(embedding)
        #embedding = self.nonlinearity(self.conv2(embedding))
        #embedding = self.pool2(embedding)
        
        embedding = embedding.reshape(len(batch), 12544)
        batch[self.out_column] = embedding
        return batch

class Classifier(PyTorch_Model):
    """ Uses the input embedding to perform a classification. """    
    required_components = ['in_column', 'out_column', 'linear_layer']

    def init_default_components(self):
        self.components['in_column'] = 'embeddings'
        self.components['out_column'] = 'predictions'
        self.components['linear1'] = torch.nn.Linear(12544, 256)
        self.components['linear2'] = torch.nn.Linear(256, 10)
        self.components['nonlinearity'] = torch.nn.ELU()
        self.components['softmax'] = torch.nn.Softmax()

    def forward(self, batch):
        
        predictions = batch[self.in_column]
        predictions = self.nonlinearity(self.linear1(predictions))
        predictions = self.softmax(self.linear2(predictions))        
        batch[self.out_column] = predictions
        return batch 

embedder = mnistModel()
classifier = Classifier(input=embedder)

ce_loss = torch.nn.CrossEntropyLoss()

loss = lambda batch: ce_loss(batch['predictions'], batch['labels'])

trainer = IgniteJunction(components={'model': classifier, 'dataset': training_set}, loss=loss, optimizer='Adam', lr=.0001, weight_decay=.001, visdom=True, environment=env_name)

trainer.run(max_epochs=10)