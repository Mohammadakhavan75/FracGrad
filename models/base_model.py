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
    def __init__(self, num_classes=10):
        super(BaseModel, self).__init__()

    @abstractmethod
    def penultimate(self, inputs, all_features=False):
        pass

    def forward(self, inputs, penultimate=False):

        output, features_list = self.penultimate(inputs, penultimate)
        output = None
        if penultimate:
            return output, features_list
        else:
            return output
