#-*- codeing = utf-8 -*-
#@Time : 2022/3/30 20:11
#@Author : 鹿俊聪
#@File : mobilev2.py
#@Software : PyCharm
import math
from modelsblock.baseblock import *

class MobileNetV2(nn.Module):
    def __init__(self,
                 n_class=10,
                 input_size=224,
                 width_mult=1.,
                 ):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 1],
            [6, 32, 3, 1],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        # 1280
        # self.zero_init_residual = zero_init_residual
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel

        self.features = [conv_bn_relu(3, input_channel, 3,1,1)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        self.features.append(conv_bn_relu(input_channel,self.last_channel, 1,1,0))
        self.features = nn.Sequential(*self.features)
        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x

    def get_featmap(self, x):
        # 只返回特征矩阵，不进行分类的意思吗，再返回一个对特征矩阵的平均
        x = self.features(x)
        return x, x.mean([2, 3])

    @property
    def feature_dim(self):
        # 取最后一个特征矩阵的维度
        return self.last_channel

def mobilenet_v2(pretrained=False, progress=True, **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2(**kwargs)
    if pretrained:
            # 下载权重文件，并加载到预训练模型上
        model.load_state_dict(torch.load('g:/biyesheji/AdaFocus-main/Experiments/model_best.pth.tar', map_location='cpu')['state_dict'])
    return model
