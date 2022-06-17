#-*- codeing = utf-8 -*-
#@Time : 2022/3/30 20:09
#@Author : 鹿俊聪
#@File : __init__.py.py
#@Software : PyCharm
from __future__ import absolute_import
from .mobilev2 import MobileNetV2
from .shufflev2 import ShuffleNetV2
from .USmobilev2 import USMobileNetV2,USInvertedResidual
from .resnet import resnet50
from .baseblock import *
from .slimmableops import *
from .vgg import VGG