"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import oneflow as flow
import oneflow.nn as nn

from typing import Any


__all__ = ["AlexNet", "alexnet", "AlexNet_parallel", "alexnet_parallel"]


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=96,kernel_size=(3,3),stride=1,padding=1),nn.ReLU(),nn.MaxPool2d(kernel_size=(3,3),stride=2),
            nn.Conv2d(in_channels=96,out_channels=256,kernel_size=(5,5),stride=1,padding=2),nn.ReLU(),nn.MaxPool2d(kernel_size=(3,3),stride=2),
            nn.Conv2d(in_channels=256,out_channels=384,kernel_size=(3,3),padding=1),nn.ReLU(),
            nn.Conv2d(in_channels=384,out_channels=384,kernel_size=(3,3),padding=1),nn.ReLU(),
            nn.Conv2d(in_channels=384,out_channels=256,kernel_size=(3,3),padding=1),nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3,3),stride=2)
                                    )
        self.layer2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*3*3, 1024), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 10)
        )

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x



def alexnet(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> AlexNet:
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    The required minimum input size of the model is 63x63.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(**kwargs)
    return model


class AlexNet_parallel(nn.Module):
    def __init__(self):
        super(AlexNet_parallel,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=96,kernel_size=(3,3),stride=1,padding=1),nn.ReLU(),nn.MaxPool2d(kernel_size=(3,3),stride=2),
            nn.Conv2d(in_channels=96,out_channels=256,kernel_size=(5,5),stride=1,padding=2),nn.ReLU(),nn.MaxPool2d(kernel_size=(3,3),stride=2),
            nn.Conv2d(in_channels=256,out_channels=384,kernel_size=(3,3),padding=1),nn.ReLU(),
            nn.Conv2d(in_channels=384,out_channels=384,kernel_size=(3,3),padding=1),nn.ReLU(),
            nn.Conv2d(in_channels=384,out_channels=256,kernel_size=(3,3),padding=1),nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3,3),stride=2)
                                    )
        self.layer2 = nn.Sequential(
            nn.Flatten(),
            linear(256*3*3, 1024), nn.ReLU(),
            nn.Dropout(p=0.5),
            linear(1024, 1024), nn.ReLU(),
            nn.Dropout(p=0.5),
            linear(1024, 10)
        )

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x



def alexnet_parallel(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> AlexNet:
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    The required minimum input size of the model is 63x63.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet_parallel(**kwargs)
    return model


def linear(in_features, out_features, bias=None):
    m = flow.nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
    m.weight = flow.nn.Parameter(
        m.weight.to_global(placement=flow.placement("cuda", [0,1]), sbp=flow.sbp.split(1))
    )
    if m.bias is not None:
        # bias is 1-d tensor
        m.bias = torch.nn.Parameter(placement=flow.placement("cuda", [0,1]), sbp=flow.sbp.split(1))

