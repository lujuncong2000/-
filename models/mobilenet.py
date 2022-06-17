from torch import nn
from .utils import load_state_dict_from_url
from collections import OrderedDict

__all__ = ['MobileNetV2', 'mobilenet_v2']


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


def _make_divisible(v, divisor, min_value=None):
    """
    # v指输入特征深度，divisor指基数
    # 此函数的作用时讲v调整为指定divisor这个数的整数倍，将v调整为离8最近的整数倍的数值
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    """
    定义网络结构，每一个卷积操作
    group=1表示是普通卷积，group=2表示Depthwise(DW)卷积
    padding有kernel size大小决定
    """
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        # in_planes:输入通道深度 out_planes:输出通道深度 kernel_size:卷积核大小 stride：步距 groups：如果是一的话是普通卷积 是输入深度的话是DW卷积
        padding = (kernel_size - 1) // 2# 填充参数
        super(ConvBNReLU, self).__init__(
            # bias=false是因为还要输入BN层 所以偏置没有意义
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )

#反残差网络
class InvertedResidual(nn.Module):
    """
    倒残差结构
    1.expand_ratio表示扩展因子。
    2.hidden_channel = in_channel * expand_ratio中的hidden_channel表示输出深度也是卷积核数量。
    3.use_shortcut判断是否在正向传播过程中使用Mobile的捷径分支，bool值。
    4.stride == 1 and in_channel == out_channel判断使用捷径分支条件：stride == 1并且输入深度等于输出深度。
    5. if expand_ratio != 1 判断扩展因子是不是等于1，不等于1就添加一个1x1的卷积层。等于1的话就没有1x1的卷积层。
    forward()如果use_shortcut为TRUE的话使用捷径分支，返回卷积结果和捷径分支的和。如果为FALSE返回主分支卷积结果
    """
    def __init__(self, inp, oup, stride, expand_ratio):
        # inp：输入特征矩阵深度 oup：输出特征矩阵深度 stride：步距 expand_ratio:扩展因子，就是t
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        #隐藏层深度=输入层深度*扩展因子
        hidden_dim = int(round(inp * expand_ratio))
        # 使用捷径分支条件就是步距为一且输入深度=输出深度
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw 1x1 pointwise conv只有在扩展因子！=1时，才有第一层1*1的卷积层
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        # DW卷积输出特征矩阵深度与输入是相等的
        layers.extend([
            # dw 3x3 depthwise conv
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear 1x1 pointwise conv(linear)，线性激活并不需要用激活函数，所以conv2d和BN就够了
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        # 将上面的打包传在一起
        self.conv = nn.Sequential(*layers)


    def forward(self, x):
        # 是否使用捷径分支
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            通过这个数量来调整每一层通道的数量
            inverted_residual_setting: Network structure
            反向剩余设置：网络结构
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            四舍五入？将每层中的通道数四舍五入为该数字的倍数设置为1可关闭舍入
        """
        super(MobileNetV2, self).__init__()
        block = InvertedResidual  # 意思是block类等同于倒残差类
        input_channel = 32  # 输入的通道
        last_channel = 1280  # 最后一层的通道

        if inverted_residual_setting is None:
            # t是扩展因子，c是输出特征矩阵深度，n是bottleneck重复深度，s是步距
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]  # 添加第一个卷积层
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            # 调整输出通道数
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                # n代表重复几次倒残差结构
                stride = s if i == 0 else 1  # stride如果是第一层赋值为s如果不是第一层赋值为1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel  # 搭建每一个参差结构
        # building last several layers，搭建1*1的卷积层,1280
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential，特征提取层完毕
        self.features = nn.Sequential(*features)

        # building classifier，分类器部分
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization， 对所有子模块权重进行一个初始化
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
        # 没有用平均池化层是因为会损失位置信息，所以改成了平均值，对第二第三维度求平均
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
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model
