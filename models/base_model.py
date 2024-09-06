'''
ResNet in PyTorch.
This code mainly adopted from:

<https://github.com/alinlab/CSI>

@inproceedings{tack2020csi,
  title={CSI: Novelty Detection via Contrastive Learning on Distributionally Shifted Instances},
  author={Jihoon Tack and Sangwoo Mo and Jongheon Jeong and Jinwoo Shin},
  booktitle={Advances in Neural Information Processing Systems},
  year={2020}
}
'''

from abc import *
import torch.nn as nn


class BaseModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, last_dim, num_classes=10):
        super(BaseModel, self).__init__()
        self.linear = nn.Linear(last_dim, num_classes)

    @abstractmethod
    def penultimate(self, inputs, all_features=False):
        pass

    def forward(self, inputs):

        features = self.penultimate(inputs)
        output = self.linear(features)

        return output
