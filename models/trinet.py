from torchvision.models.resnet import ResNet
from torchvision.models.resnet import Bottleneck
from torchvision.models.resnet import model_urls
import torch.utils.model_zoo as model_zoo
import torch.nn as nn
model_parameters = {"dim": None}

class TriNet(ResNet):
    """TriNet implementation.

    Replaces the last layer of ResNet50 with two fully connected layers.

    First: 1024 units with batch normalization and ReLU
    Second: 128 units, final embedding.
    """
    
    def __init__(self, block, layers, dim=128, **kwargs):
        """Initializes original ResNet and overwrites fully connected layer."""

        super(TriNet, self).__init__(block, layers, 1) # 0 classes thows an error
        batch_norm = nn.BatchNorm1d(1024)
        self.avgpool = nn.AvgPool2d((4,4))
        self.fc = nn.Sequential(
            nn.Linear(512 * block.expansion, 1024),
            batch_norm,
            nn.ReLU(),
            nn.Linear(1024, dim)
        )
        batch_norm.weight.data.fill_(1)
        batch_norm.bias.data.zero_()
        self.dim = dim
        self.dimensions = {'emb': (self.dim, )}

    def forward(self, x, endpoints):
        x = super().forward(x)
        endpoints["emb"] = x
        return endpoints

def trinet(**kwargs):
    """Creates a TriNet network and loads the pretrained ResNet50 weights.
    
    https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/2
    """


    model = TriNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    #print(model)
    pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
    model_dict = model.state_dict()

    # filter out fully connected keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if not k.startswith("fc")}
    #for key, value in pretrained_dict.items():
    #    print(key)

    # overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # load the new state dict
    model.load_state_dict(model_dict)
    endpoints = {}
    endpoints["emb"] = None
    return model, endpoints
