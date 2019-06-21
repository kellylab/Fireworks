from Fireworks PyTorch_Model
import torch
from torch import nn 
from torchvision.models import vgg1

class ImageEmbedder(PyTorch_Model):
    
    def __init__(self, *args, **kwargs):
        required_components = ['widths', 'in_column', 'out_column']

        super().__init__(*args, **kwargs)
        for i in range(len(self.widths)-1):
            self.components['layer{0}'.format(i)] = nn.Linear(int(self.widths[i].tolist()), int(self.widths[i+1].tolist()))
        self.num_layers = len(self.widths)-1

    def init_default_components(self):
        self.components['vgg'] = vgg16(pretrained=True)
        for parameter in self.vgg.parameters():
            parameter.requires_grad = False
        self.components['in_column'] = 'image'
        self.components['out_column'] = 'image_embedding'

    def forward(self, batch):
        embedding = batch[self.in_column]
        embedding = self.vgg(embedding)
        for i in range(self.num_layers):
            layer = getattr(self, 'layer{0}'.format(i))
            embedding = layer(embedding)
            embedding = self.elu(embedding)
        batch[self.out_column]  = embedding
        return batch 

class QuestionEmbedder(PyTorch_Model):
    required_components = ['widths', 'in_column', 'out_column']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: Initialize the layers as LSTMs 

    def init_default_components(self):
        self.components['in_column'] = 'tokenized_question'
        self.components['out_column'] = 'question_embedding'

    def forward(self, batch):
        embedding = batch[self.in_column]
        for i in range(self.num_layers):
            layer = getattr(self, 'layer{0}'.format(i))
            embedding = layer(embedding)
            embedding = self.elu(embedding)
        batch[self.out_column] = embedding 
        return batch 

class QuestionImageEmbedder(PyTorch_Model):
    """ Combines an Image and Question embedding into a single joint embedding. """
    required_components = ['widths', 'question_column', 'image_column', 'out_column']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for i in range(len(self.widths)-1):
            self.components['layer{0}'.format(i)] = nn.Linear(int(self.widths[i].tolist()), int(self.widths[i+1].tolist()))
        self.num_layers = len(self.widths)-1

    def init_default_components(self):
        self.components['image_column'] = 'image_embedding'
        self.components['question_column'] = 'question_embedding'
        self.components['out_column'] = 'vqa_embedding'

    def forward(self, batch):
        image_embedding = batch[self.image_column]
        question_embedding = batch[self.question_embedding]
        # Get the hidden state at first EOS token.
        # This helps the training process by preventing excess EOS tokens
        # from clogging up the gradients.
        question_embedding = [ 
            question_embedding[i, batch['length'].data[i][0]]
            for i in range(batch['length'].size()[0])
        ]
        vqa = torch.cat((image_embedding, question_embedding), 1)
        for i in range(self.num_layers):
            layer = getattr(self, 'layer{0}'.format(i))
            vqa = layer(vqa)
            vqa = self.elu(vqa)
        batch[self.out_column] = vqa
        return batch

class QAClassifier(PyTorch_Model):
    
    required_components = ['in_width', 'out_width', 'in_column', 'out_column']

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.classification_layer = nn.Linear(self.in_width, self.out_width)
        self.activation = nn.Softmax()

    def init_default_components(self):

        self.components['in_column'] = 'vqa_embedding'
        self.components['out_column'] = 'classification'

    def forward(self, batch):

        output = batch[self.in_column]
        output = self.activation(output)
    
        batch[self.out_column] = output 
        return batch    