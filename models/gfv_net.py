
from PIL.Image import Image
from torch import autograd, nn
from ops.transforms import *
import torch.nn.functional as F
from torch.distributions import Categorical
from .resnet import resnet50
from .mobilenet import mobilenet_v2
from .utils import random_crop, get_patch
from .ppo import PPO, Memory
import torchvision

class GFV(nn.Module):
    """
    top class for adaptive inference on video
    """
    def __init__(self):
        super(GFV, self).__init__()
        # self.num_segments = args.num_segments
        # self.num_class = args.num_classes
        # self.rew = args.reward #random
        self.num_segments = 16
        self.num_class = 101
        self.rew = 'random'  # random
        # if args.dataset == 'fcvid':
        #     assert args.num_classes == 239
        self.glancer = None
        self.focuser = None
        self.classifier = None
        # self.input_size = args.input_size
        # self.batch_size = args.batch_size
        # self.patch_size = args.patch_size
        self.input_size = 256
        self.batch_size = 8
        self.patch_size = 128
        self.input_mean = [0.485, 0.456, 0.406]#imagenet的均值正则化 防止过拟合
        self.input_std = [0.229, 0.224, 0.225]#imagenet的标准差正则化
        # self.with_glancer = args.with_glancer#默认是true
        self.with_glancer = True
        self.glancer = Glancer(num_classes=self.num_class)#一个全局网络用来获得粗略的信息
        #状态维度==（特征图通道）1280*7*7，没看懂要我干嘛，感觉要改
        # state_dim = args.feature_map_channels * math.ceil(args.glance_size / 32) * math.ceil(args.glance_size / 32)
        state_dim = 1280 * math.ceil(224 / 32) * math.ceil(224 / 32)
        #策略网络所用参数，指focuser，感觉也有强化学习的事
        # policy_params = {
        #     'feature_dim': args.feature_map_channels,#特征矩阵通道数,1280
        #     'state_dim': state_dim,#状态维度1280*7*7
        #     'action_dim': args.action_dim,# 动作维度49
        #     'hidden_state_dim': args.hidden_state_dim,# 隐藏状态维度1024
        #     'policy_conv': args.policy_conv,# 策略卷积 true
        #     'gpu': args.gpu,
        #     'continuous': args.continuous,#gru
        #     'gamma': args.gamma,#0.7
        #     'policy_lr': args.policy_lr#0.0003
        # }
        policy_params = {
            'feature_dim': 1280,  # 特征矩阵通道数,1280
            'state_dim': 1280*7*7,  # 状态维度1280*7*7
            'action_dim': 49,  # 动作维度49
            'hidden_state_dim': 1024,  # 隐藏状态维度1024
            'policy_conv': True,  # 策略卷积 true
            'gpu': None,
            'continuous': False,  # gru
            'gamma': 0.7,  # 0.7
            'policy_lr': 0.0003  # 0.0003
        }
        # patch_size=128，random_patch=224
        # self.focuser = Focuser(args.patch_size, args.random_patch, policy_params, self.num_class)
        # self.dropout = nn.Dropout(p=args.dropout)
        self.focuser = Focuser(128, False, policy_params, self.num_class)
        self.dropout = nn.Dropout(p=0.8)
        if self.with_glancer:
            feat_dim = self.glancer.feature_dim + self.focuser.feature_dim # 3328
        else:
            feat_dim = self.focuser.feature_dim
        self.classifier = RecurrentClassifier(seq_len=16, input_dim=feat_dim, batch_size=self.batch_size,
                                              hidden_dim=1024, num_classes=self.num_class,
                                              dropout=0.8)
        self.down = torchvision.transforms.Resize((self.patch_size, self.patch_size), interpolation=Image.BILINEAR)
        # if args.consensus == 'gru':
        #     print('Using GRU-based Classifier!')
        #     self.classifier = RecurrentClassifier(seq_len=args.num_segments,input_dim = feat_dim, batch_size = self.batch_size,hidden_dim=args.hidden_dim, num_classes=args.num_classes, dropout=args.dropout)
        # elif args.consensus == 'fc':
        #     print('Using Linear Classifier!')
        #     self.classifier = LinearCLassifier(seq_len=args.num_segments, input_dim = feat_dim, batch_size= self.batch_size, hidden_dim=args.hidden_dim, num_classes=args.num_classes, dropout=args.dropout)
        # self.down = torchvision.transforms.Resize((args.patch_size, args.patch_size),interpolation=Image.BILINEAR)# 下采样，我想应该是截取部分输入图片，双线性
    
    def train(self, mode=True):
        super(GFV, self).train(mode)
        return

    def train_mode(self, args):
        if args.train_stage == 0:
            # 指的是训练模式
            self.train()
        elif args.train_stage == 1:
            # glancer进入评估阶段
            self.train()
            self.glancer.eval()
        elif args.train_stage == 2:
            self.eval()
            self.glancer.eval()
            self.focuser.eval()
            self.focuser.policy.policy.train()
            self.focuser.policy.policy_old.train()
        elif args.train_stage == 3:
            self.train()
            self.glancer.eval()
            self.focuser.eval()
            self.focuser.policy.policy.eval()
            self.focuser.policy.policy_old.eval()
        return

    def forward(self, *argv, **kwargs):
        if kwargs["backbone_pred"]:
            # input.shape = [16,48,224,224]
            input = kwargs["input"]
            _b, _tc, _h, _w = input.shape  # input (B, T*C, H, W)
            _t, _c = _tc // 3, 3
            # input_2d.shape = [256,3,224,224]
            input_2d = input.view(_b * _t, _c, _h, _w)
            if kwargs['glancer']:
                # 直接输入[256,3,224,224]并预测出结果,结果是[16,16,101]的矩阵大小
                pred = self.glancer.predict(input_2d).view(_b, _t, -1)
            else:
                # 输入[256,3,224,224]并预测出结果,结果是[8,16,101]
                pred = self.focuser.predict(input_2d).view(_b, _t, -1)
            return pred
        elif kwargs["one_step"]:
            # 阶段三走这条路径
            gpu = kwargs["gpu"]
            input = kwargs["input"]
            down_sampled = kwargs["scan"]
            _b, _tc, _h, _w = input.shape
            _t, _c = _tc // 3, 3
            input_2d = input.view(_b, _t, _c, _h, _w)

            with torch.no_grad():
                global_feat_map, global_feat = self.glance(down_sampled)
            outputs = []
            preds = []
            features = []
            if not self.focuser.random:
                # for s3 training
                for focus_time_step in range(_t):
                    img = input_2d[:, focus_time_step, :, :, :]
                    cur_global_feat_map = global_feat_map[:, focus_time_step, :, :, :]
                    cur_global_feat = global_feat[:, focus_time_step, :]
                    if self.with_glancer:
                        with torch.no_grad():
                            if focus_time_step == 0:
                                local_feat, patch_size_list = self.focuser(input=img, state=cur_global_feat_map, restart_batch=True, training=kwargs["training"])
                            else:
                                local_feat, patch_size_list = self.focuser(input=img, state=cur_global_feat_map, restart_batch=False, training=kwargs["training"])
                            local_feat = local_feat.view(_b, -1)
                            feature = torch.cat([cur_global_feat, local_feat], dim=1)
                            features.append(feature)
                    else:
                        with torch.no_grad():
                            if focus_time_step == 0:
                                local_feat, patch_size_list = self.focuser(input=img, state=cur_global_feat_map, restart_batch=True, training=kwargs["training"])
                            else:
                                local_feat, patch_size_list = self.focuser(input=img, state=cur_global_feat_map, restart_batch=False, training=kwargs["training"])
                            local_feat = local_feat.view(_b, -1)
                            feature = local_feat
                            features.append(feature)
                features = torch.stack(features, dim=1)
                outputs, pred = self.classifier(features)
                return (outputs, pred, patch_size_list)  # 更改过注意
        else:
            # for s1 training
            input = kwargs["input"]# images
            down_sampled = kwargs["scan"]# 下采样 input_prime
            _b, _tc, _h, _w = input.shape  # input (B, T*C, H, W)
            _t, _c = _tc // 3, 3
            input_2d = input.view(_b * _t, _c, _h, _w)
            _b, _tc, _h, _w = down_sampled.shape  # input (B, T*C, H, W)
            _t, _c = _tc // 3, 3
            # [256,3,224,224]
            downs_2d = down_sampled.view(_b * _t, _c, _h, _w)
            with torch.no_grad():
                # 全局特征矩阵[256,1280,7,7],7*7是mobilenetv2特征矩阵输出的大小 全局特征[256,1280]
                global_feat_map, global_feat = self.glancer(downs_2d)
            #input:[256,3,224,224],state:[256,1280,7,7],restart_batch=true,training=true, 返回矩阵[256,2048]
            local_feat = self.focuser(input=input_2d, state=global_feat_map, restart_batch=True, training=kwargs["training"])[0].view(_b*_t, -1)
            feature = torch.cat([global_feat, local_feat], dim=1)# 验证是否是[256,3328]
            feature = feature.view(_b, _t, -1)# [16,16,3328]
            return self.classifier(feature)

    def glance(self, input_prime):
        _b, _tc, _h, _w = input_prime.shape  # input (B, T*C, H, W),[32,48,224,224]
        _t, _c = _tc // 3, 3
        downs_2d = input_prime.view(_b * _t, _c, _h, _w)# [32*16,3,224,224]
        global_feat_map, global_feat = self.glancer(downs_2d)# global_feat_map:[512,1280,7,7], global_feat:[512,1280]
        _, _featc, _feath, _featw = global_feat_map.shape# 512,1280,7,7
        #global_feat_map.view(_b, _t, _featc, _feath, _featw):[32,16,1280,7,7], global_feat.view(_b, _t, -1):[32,16,1280]
        return global_feat_map.view(_b, _t, _featc, _feath, _featw), global_feat.view(_b, _t, -1)

    def one_step_act(self, img, global_feat_map, global_feat, restart_batch=False, training=True):
        _b, _c, _h, _w = img.shape#[32,3,224,224]
        #input:[32,3,224,224],state:[32,1,1280,7,7],restart_batch:true(0),false(>0),training:true;local_feat:[32,2048,1,1],pack:[None,[32,2]]
        local_feat, pack = self.focuser(input=img, state=global_feat_map, restart_batch=restart_batch, training=training)
        if pack is not None:
            patch_size_list, action_list = pack#none,[32,2]
        else:
            patch_size_list, action_list = None, None
        
        if self.with_glancer:
            # global_feat:[32,1280],local_feat.view(32,2048)
            feature = torch.cat([global_feat, local_feat.view(_b, -1)], dim=1)#[32,3328]
        else:
            feature = local_feat.view(_b, -1)
        feature = torch.unsqueeze(feature, 1) # (B, 1, feat)[32,1,3328]

        # for reward that contrast to random patching
        if self.rew == 'random':
            baseline_local_feature, pack = self.focuser.random_patching(img)#[32,3,224,224]
            if self.with_glancer:
                baseline_feature = torch.cat([global_feat, baseline_local_feature.view(_b, -1)], dim=1)
            else:
                baseline_feature = baseline_local_feature.view(_b, -1)
        elif self.rew == 'padding':
            # for reward that padding 0
            print('reward padding 0!')
            if self.with_glancer:
                baseline_feature = torch.cat([global_feat, torch.zeros(_b, self.focuser.feature_dim).cuda()], dim=1)
            else:
                baseline_feature = torch.zeros(_b, self.focuser.feature_dim).cuda()
        elif self.rew == 'prev':
            # bsl feat not used
            if self.with_glancer:
                baseline_feature = torch.cat([global_feat, torch.zeros(_b, self.focuser.feature_dim).cuda()], dim=1)
            else:
                baseline_feature = torch.zeros(_b, self.focuser.feature_dim).cuda()
        elif self.rew == 'conf':
            # bsl feat not used
            if self.with_glancer:
                baseline_feature = torch.cat([global_feat, torch.zeros(_b, self.focuser.feature_dim).cuda()], dim=1)
            else:
                baseline_feature = torch.zeros(_b, self.focuser.feature_dim).cuda()
        else:
            raise NotImplementedError

        baseline_feature = torch.unsqueeze(baseline_feature, 1)
        with torch.no_grad():
            baseline_logits, _ = self.classifier.test_single_forward(baseline_feature, reset=restart_batch)
            logits, last_out = self.classifier.single_forward(feature, reset=restart_batch)
        if training:
            return logits, last_out, patch_size_list, baseline_logits
        else:
            return logits, last_out, patch_size_list, action_list, baseline_logits

    @property
    def scale_size(self):
        return self.input_size * 256 // 224
    
    @property
    def crop_size(self):
        return self.input_size

    def get_augmentation(self, flip=True):
        if flip:
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]), GroupRandomHorizontalFlip(is_flow=False)])
        else:
            print('#' * 20, 'NO FLIP!!!')
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66])])

    def get_patch_augmentation(self):
        return torchvision.transforms.Compose([GroupScale(self.patch_size), GroupCenterCrop(self.patch_size)])


class Glancer(nn.Module):
    """
    Global network for glancing，粗略的扫描每一帧得到全局信息
    skip：？跳过？
    num_classes:数据集的种类个数
    """
    def __init__(self, skip=False, num_classes=200):
        super(Glancer, self).__init__()
        self.net = mobilenet_v2(pretrained=True)#需要预训练的mobilenet_2模型
        num_ftrs = self.net.last_channel#取最后一层特征层的深度
        # 将分类器改为有特定种类输出的分类器
        self.net.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_ftrs, num_classes),
        )
        self.skip = skip
    
    def forward(self, input):
        # 返回特征提取层的结果
        return self.net.get_featmap(input)

    def predict(self, input):
        # 返回预测结果
        return self.net(input)
    
    @property
    def feature_dim(self):
        # 返回特征提取层的维度
        return self.net.feature_dim

class Focuser(nn.Module):
    """
    Local network for focusing
    Resnet50
    """
    def __init__(self, size=96, random=True, policy_params: dict = None, num_classes=200):
        super(Focuser, self).__init__()
        self.net = resnet50(pretrained=True)#预训练过的resnet
        num_ftrs = self.net.fc.in_features#给我的感觉像是resnet50分类器的输入特征数量
        self.net.fc = nn.Linear(num_ftrs, num_classes)#分类器网络是一个全连接层，映射到101

        self.patch_size = size#截取区域的大小,128
        self.random = random#true
        self.patch_sampler = PatchSampler(self.patch_size, self.random)#随机截取方块大小的列表，大小是[C*T*B,H,W]
        self.policy = None #策略网络
        self.memory = Memory() #记忆因子，用来记录一些中间变量
        if not self.random:
            assert policy_params != None
            # 感觉是一个平均截取一张图片大小的东西
            self.standard_actions_set = {
                25: torch.Tensor([
                    [0, 0], [0, 1/4], [0, 2/4], [0, 3/4], [0, 1],
                    [1/4, 0], [1/4, 1/4], [1/4, 2/4], [1/4, 3/4], [1/4, 1],
                    [2/4, 0], [2/4, 1/4], [2/4, 2/4], [2/4, 3/4], [2/4, 1],
                    [3/4, 0], [3/4, 1/4], [3/4, 2/4], [3/4, 3/4], [3/4, 1],
                    [4/4, 0], [4/4, 1/4], [4/4, 2/4], [4/4, 3/4], [4/4, 1],
                ]).cuda(),
                36: torch.Tensor([
                    [0, 0], [0, 1/5], [0, 2/5], [0, 3/5], [0, 4/5], [0, 5/5],
                    [1/5, 0], [1/5, 1/5], [1/5, 2/5], [1/5, 3/5], [1/5, 4/5], [1/5, 5/5],
                    [2/5, 0], [2/5, 1/5], [2/5, 2/5], [2/5, 3/5], [2/5, 4/5], [2/5, 5/5],
                    [3/5, 0], [3/5, 1/5], [3/5, 2/5], [3/5, 3/5], [3/5, 4/5], [3/5, 5/5],
                    [4/5, 0], [4/5, 1/5], [4/5, 2/5], [4/5, 3/5], [4/5, 4/5], [4/5, 5/5],
                    [5/5, 0], [5/5, 1/5], [5/5, 2/5], [5/5, 3/5], [5/5, 4/5], [5/5, 5/5],
                ]).cuda(),
                49: torch.Tensor([
                    [0, 0], [0, 1/6], [0, 2/6], [0, 3/6], [0, 4/6], [0, 5/6], [0, 1],
                    [1/6, 0], [1/6, 1/6], [1/6, 2/6], [1/6, 3/6], [1/6, 4/6], [1/6, 5/6], [1/6, 1],
                    [2/6, 0], [2/6, 1/6], [2/6, 2/6], [2/6, 3/6], [2/6, 4/6], [2/6, 5/6], [2/6, 1],
                    [3/6, 0], [3/6, 1/6], [3/6, 2/6], [3/6, 3/6], [3/6, 4/6], [3/6, 5/6], [3/6, 1],
                    [4/6, 0], [4/6, 1/6], [4/6, 2/6], [4/6, 3/6], [4/6, 4/6], [4/6, 5/6], [4/6, 1],
                    [5/6, 0], [5/6, 1/6], [5/6, 2/6], [5/6, 3/6], [5/6, 4/6], [5/6, 5/6], [5/6, 1],
                    [6/6, 0], [6/6, 1/6], [6/6, 2/6], [6/6, 3/6], [6/6, 4/6], [6/6, 5/6], [6/6, 1],
                ]).cuda(),
                64: torch.Tensor([
                    [0, 0], [0, 1/7], [0, 2/7], [0, 3/7], [0, 4/7], [0, 5/7], [0, 6/7], [0, 7/7],
                    [1/7, 0], [1/7, 1/7], [1/7, 2/7], [1/7, 3/7], [1/7, 4/7], [1/7, 5/7], [1/7, 6/7], [1/7, 7/7],
                    [2/7, 0], [2/7, 1/7], [2/7, 2/7], [2/7, 3/7], [2/7, 4/7], [2/7, 5/7], [2/7, 6/7], [2/7, 7/7],
                    [3/7, 0], [3/7, 1/7], [3/7, 2/7], [3/7, 3/7], [3/7, 4/7], [3/7, 5/7], [3/7, 6/7], [3/7, 7/7],
                    [4/7, 0], [4/7, 1/7], [4/7, 2/7], [4/7, 3/7], [4/7, 4/7], [4/7, 5/7], [4/7, 6/7], [4/7, 7/7],
                    [5/7, 0], [5/7, 1/7], [5/7, 2/7], [5/7, 3/7], [5/7, 4/7], [5/7, 5/7], [5/7, 6/7], [5/7, 7/7],
                    [6/7, 0], [6/7, 1/7], [6/7, 2/7], [6/7, 3/7], [6/7, 4/7], [6/7, 5/7], [6/7, 6/7], [6/7, 7/7],
                    [7/7, 0], [7/7, 1/7], [7/7, 2/7], [7/7, 3/7], [7/7, 4/7], [7/7, 5/7], [7/7, 6/7], [7/7, 7/7],
                ]).cuda()
            }
            self.policy_feature_dim = policy_params['feature_dim']# 1280
            self.policy_state_dim = policy_params['state_dim']# 1280*7*7
            self.policy_action_dim = policy_params['action_dim']# 49
            self.policy_hidden_state_dim = policy_params['hidden_state_dim']# 1024
            self.policy_conv = policy_params['policy_conv']# true
            self.gpu = policy_params['gpu'] #for ddp
            self.policy = PPO(self.policy_feature_dim, self.policy_state_dim, self.policy_action_dim, self.policy_hidden_state_dim, self.policy_conv, self.gpu, gamma=policy_params['gamma'], lr=policy_params['policy_lr'])
    
    def forward(self, *argv, **kwargs):
        if self.random:
            standard_action = None
        else:
            # input:[32,3,224,224],state:[32,1280,7,7],restart_batch:true(0),false(>0),training:true;output:[32]
            action = self.policy.select_action(kwargs['state'], self.memory, kwargs['restart_batch'], kwargs['training'])
            standard_action, patch_size_list = self._get_standard_action(action)# [32,2],none

        # print('action:', standard_action)
        imgs = kwargs['input'] # input:[256,3,224,224]
        _b = imgs.shape[0]# 256
        if self.random:
            # patch:[256,3,128,128]
            patch = self.patch_sampler.sample(imgs, standard_action)# PatchSampler(self.patch_size, self.random)
            # 预训练阶段不执行forward,[256,2048,1,1]
            return self.net.get_featmap(patch, pooled=True), None
        else:
            patch = self.patch_sampler.sample(imgs, standard_action)# 根据standard_action选择图片的意思呗[32,3,128,128]
            return self.net.get_featmap(patch, pooled=True), (None, standard_action)
    

    def random_patching(self, imgs):
        patch = self.patch_sampler.random_sample(imgs)#[32,3,224,224]
        return self.net.get_featmap(patch, pooled=True), None

    def predict(self, input):
        # 输出[128,101], 预训练阶段只走预测
        return self.net(input)

    def update(self):
        self.policy.update(self.memory)
        self.memory.clear_memory()
    
    def _get_standard_action(self, action):
        # 取那个7*7,[32,2]
        standard_action = self.standard_actions_set[self.policy_action_dim]
        return standard_action[action], None
    
    @property
    def feature_dim(self):
        return self.net.feature_dim


class PatchSampler(nn.Module):
    """
    Sample patch over the whole image
    截取片段类
    """
    def __init__(self, size=96, random=True) -> None:
        super(PatchSampler, self).__init__()
        self.random = random# 是否随机截取
        self.size = size# 截取尺寸大小

    def sample(self, imgs, action = None):
        if self.random:
            # crop at random position
            batch = []
            # print(self.size)
            for img in imgs:
                batch.append(random_crop(img, self.size))
            # 返回batch的拼接，应该是[C*T*B,H,W]
            return torch.stack(batch)
        else:
            # crop at the position yielded by policy network，有一个[32,2]的方位列表
            assert action != None
            return get_patch(imgs, action, self.size)

    def random_sample(self, imgs):
        # crop at random position
        batch = []
        for img in imgs:
            batch.append(random_crop(img, self.size))
        return torch.stack(batch)

    def forward(self, *argv, **kwargs):
        raise NotImplementedError



class LinearCLassifier(nn.Module):
    def __init__(self, seq_len, input_dim, batch_size, hidden_dim, num_classes, dropout):
        super(LinearCLassifier, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.fc = nn.Linear(self.input_dim, self.num_classes)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, feature):
        _b, _t, _f = feature.shape
        out = self.dropout(feature)
        logits = self.fc(out.reshape(_b*_t, -1))
        softmax = self.softmax(logits).reshape(_b, _t, -1)
        avg = softmax.mean(dim=1, keepdim=False)
        log_softmax = torch.log(avg)
        return log_softmax, avg

class RecurrentClassifier(nn.Module):
    """
    GRU based classifier
    (seq_len=args.num_segments16,input_dim = feat_dim3328, batch_size = self.batch_size16,
    hidden_dim=args.hidden_dim1024, num_classes=args.num_classes101, dropout=args.dropout)
    """
    def __init__(self, seq_len, input_dim, batch_size, hidden_dim, num_classes, dropout, bias=True):
        super(RecurrentClassifier, self).__init__()
        self.seq_len = seq_len#16
        self.input_dim = input_dim#3328
        self.hidden_dim = hidden_dim#1024
        self.num_classes = num_classes#101
        self.batch_size = batch_size#16
        # input_size:3328 hidden_dim:1024
        self.gru = nn.GRU(input_size=self.input_dim, hidden_size=self.hidden_dim, bias=bias, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, self.num_classes)# 1024 101

        self.hx = None
        self.cx = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, feature):
        _b, _t, _f = feature.shape # 16 16 3328
        hx = torch.zeros(self.gru.num_layers, _b,  self.hidden_dim).cuda()#1,16,1024
        # hx = torch.zeros(self.gru.num_layers, _b, self.hidden_dim)
        self.gru.flatten_parameters()
        # 对GRU的小理解：输入的feature（特征矩阵）和hx（隐藏层向量，需要初始化）是第一个GRU cell的隐藏层状态;out（输出结果）和hn（隐藏层向量）是最后一个cell的隐藏层状态
        out, hn = self.gru(feature, hx) #out(_b, _t, hidden_size) with batch_first=true,out:[16,16,1024],hn:[1,16,1024]
        out = self.dropout(out) # [16,16,1024], dropout是专门用于训练的。在推理阶段，则需要把dropout关掉，而model.eval()就会做这个事情。
        logits = self.fc(out.reshape(_b*_t, -1)) # logits:[256,101],应该是记录每一步输出
        last_out = logits.reshape(_b, _t, -1)[:, -1, :].reshape(_b, -1) # last_out:[16,101]，直接取最后一步的意思应该是，这下每一帧对应一个预测
        return logits, last_out
    
    def single_forward(self, feature, reset=False, gpu=0):
        _b, _t, _f = feature.shape
        if reset:
            self.hx = torch.zeros(self.gru.num_layers, _b,  self.hidden_dim).cuda(gpu)
        self.gru.flatten_parameters()
        out, self.hx = self.gru(feature, self.hx) #out(_b, _t, hidden_size) with batch_first=true
        out = self.dropout(out)
        logits = self.fc(out.reshape(_b*_t, -1))
        last_out = logits.reshape(_b, _t, -1)[:, -1, :].reshape(_b, -1)
        return logits, last_out
    
    def test_single_forward(self, feature, reset=False, gpu=0):
        _b, _t, _f = feature.shape
        if reset:
            self.hx = torch.zeros(self.gru.num_layers, _b,  self.hidden_dim).cuda(gpu)
        self.gru.flatten_parameters()
        out, _ = self.gru(feature, self.hx) #out(_b, _t, hidden_size) with batch_first=true
        out = self.dropout(out)
        logits = self.fc(out.reshape(_b*_t, -1))
        last_out = logits.reshape(_b, _t, -1)[:, -1, :].reshape(_b, -1)
        return logits, last_out
