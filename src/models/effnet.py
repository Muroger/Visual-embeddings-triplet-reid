from torchvision.models.resnet import ResNet
from torchvision.models.resnet import Bottleneck
from torchvision.models.resnet import model_urls
import torch.utils.model_zoo as model_zoo
import torch.nn as nn
import timm

model_parameters = {"dim": None}

class EffNet(nn.Module):
    """TriNet implementation.

    Replaces the last layer of ResNet50 with two fully connected layers.

    First: 1024 units with batch normalization and ReLU
    Second: 128 units, final embedding.
    """
    
    def __init__(self, dim=128, **kwargs):
        """Initializes original ResNet and overwrites fully connected layer."""

        super().__init__()
        self.model = timm.create_model(model_name='tf_efficientnet_b0_ns',
                                        pretrained=True)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(n_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, dim)
        )
        self.dim = dim
        self.dimensions = {'emb': (self.dim, )}
    def forward(self, x, endpoints):
        x = self.model(x)
        endpoints["emb"] = x
        return endpoints

def effnet(**kwargs):
    """Creates a TriNet network and loads the pretrained ResNet50 weights.
    
    https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/2
    """


    model = EffNet(**kwargs)
    #print(model)
    #pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
    #model_dict = model.state_dict()

    # filter out fully connected keys
    #pretrained_dict = {k: v for k, v in pretrained_dict.items() if not k.startswith("fc")}
    #for key, value in pretrained_dict.items():
    #    print(key)

    # overwrite entries in the existing state dict
    #model_dict.update(pretrained_dict)
    # load the new state dict
   # model.load_state_dict(model_dict)
    endpoints = {}
    endpoints["emb"] = None
    return model, endpoints
