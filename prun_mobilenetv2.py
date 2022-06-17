#-*- codeing = utf-8 -*-
#@Time : 2022/3/30 19:59
#@Author : 鹿俊聪
#@File : prun_mobilenetv2.py
#@Software : PyCharm

import pruner
import os
import argparse
from modelsblock import *
import torch.optim as optim
from os.path import join
import json
from models.mobilenet import mobilenet_v2

from mythop import clever_format, profile
from ops import dataset_config
from ops.dataset import TSNDataSet
from ops.transforms import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='Mobilev2 Pruner')
parser.add_argument('--dataset', type=str, default='ucf101',
                    help='training dataset (default: ucf101)')
parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=16, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=160, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--finetunelr', type=float, default=0, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='checkpoints/model_best.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--save', default='checkpoints', type=str, metavar='PATH',
                    help='path to save prune model (default: current directory)')
parser.add_argument('--arch', default='MobileNetV2', type=str, choices=['USMobileNetV2', 'MobileNetV2','VGG',
                                                                        'ShuffleNetV2','resnet50'],
                    help='architecture to use')
parser.add_argument('--pruner', default='SlimmingPruner', type=str,
                    choices=['AutoSlimPruner', 'SlimmingPruner', 'l1normPruner'],
                    help='architecture to use')
parser.add_argument('--pruneratio', default=0.4, type=float,
                    help='architecture to use')
# parser.add_argument('--sr', dest='sr', action='store_true',
#                     help='train with channel sparsity regularization')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
savepath = '/home/UserData/ljc/AdaFocus/mobilenetv2_checkpoint.pth.tar'
args.savepath = savepath
kwargs = {'num_workers': 0, 'pin_memory': False} if args.cuda else {}


def get_augmentation(flip=True):
    if flip:
        return torchvision.transforms.Compose(
            [GroupMultiScaleCrop(224, [1, .875, .75, .66]), GroupRandomHorizontalFlip(is_flow=False)])
    else:
        print('#' * 20, 'NO FLIP!!!')
        return torchvision.transforms.Compose([GroupMultiScaleCrop(224, [1, .875, .75, .66])])


num_class, train_list, val_list, root_path, prefix = dataset_config.return_dataset(args.dataset,'/home/UserData/ljc/dataset')
normalize = GroupNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_augmentation = get_augmentation(flip=True)
train_dataset = TSNDataSet(root_path, list_file=train_list, num_segments=16,image_tmpl=prefix,
        transform=torchvision.transforms.Compose([
            train_augmentation,
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            normalize]),
        dense_sample=False,
        dataset=args.dataset,
        partial_fcvid_eval=False,
        partial_ratio=0.2,
        ada_reso_skip=False,
        reso_list=224,
        random_crop=False,
        center_crop=False,
        ada_crop_list=None,
        rescale_to=224,
        policy_input_offset=0,
        save_meta=False)
val_dataset = TSNDataSet(root_path, list_file=val_list,num_segments=16, image_tmpl=prefix,random_shift=False,
        transform=torchvision.transforms.Compose([
            GroupScale(256),
            GroupCenterCrop(224),
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            normalize]),
        dense_sample=False,
        dataset=args.dataset,
        partial_fcvid_eval=False,
        partial_ratio=0.2,
        ada_reso_skip=False,
        reso_list=224,
        random_crop=False,
        center_crop=False,
        ada_crop_list=None,
        rescale_to=224,
        policy_input_offset=0,
        save_meta=False)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False, drop_last=True)
test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)

model = mobilenet_v2(pretrained=True)
num_ftrs = model.last_channel#取最后一层特征层的深度
model.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(num_ftrs, 101),
    )
newmodel = mobilenet_v2(pretrained=True)
num_ftrs = newmodel.last_channel#取最后一层特征层的深度
newmodel.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(num_ftrs, 101),
    )
# model = eval(args.arch)(input_size=32)
# newmodel = eval(args.arch)(input_size=32)
# model.load_state_dict(torch.load('g:/biyesheji/AdaFocus-main/Experiments/model_best.pth.tar', map_location='cpu')['state_dict'])

sd = torch.load(savepath)
if 'state_dict' in sd:  # a checkpoint but not a state_dict
    sd = sd['state_dict']
sd = {k.replace('module.', ''): v for k, v in sd.items()}
model.load_state_dict(sd)
print("Best trained model loaded.")

if args.cuda:
    model.cuda().eval()
    newmodel.cuda().eval()
best_prec1 = -1
optimizer = optim.SGD(model.parameters(), lr=args.finetunelr, momentum=args.momentum, weight_decay=args.weight_decay)

if args.pruner == 'l1normPruner':
    kwargs = {'pruneratio': args.pruneratio}
elif args.pruner == 'SlimmingPruner':
    kwargs = {'pruneratio': args.pruneratio}
elif args.pruner == 'AutoSlimPruner':
    kwargs = {'prunestep': 16, 'constrain': 200e6}

pruner = pruner.__dict__[args.pruner](model=model, newmodel=newmodel, testset=test_loader, trainset=train_loader,
                                      optimizer=optimizer, args=args, **kwargs)
pruner.prune()
##---------count op
input = torch.randn(1, 3, 224, 224).cuda()
flops, params = profile(model, inputs=(input,), verbose=False)
flops, params = clever_format([flops, params], "%.3f")
flopsnew, paramsnew = profile(newmodel, inputs=(input,), verbose=False)
flopsnew, paramsnew = clever_format([flopsnew, paramsnew], "%.3f")
print("flops:{}->{}, params: {}->{}".format(flops, flopsnew, params, paramsnew))
accold = pruner.test(newmodel=False, cal_bn=False)
accpruned = pruner.test(newmodel=True)
accfinetune = pruner.finetune()

print("original performance:{}, pruned performance:{},finetuned:{}".format(accold, accpruned, accfinetune))

with open(join(savepath, '{}.json'.format(args.pruneratio)), 'w') as f:
    json.dump({
        'accuracy_original': accold,
        'accuracy_pruned': accpruned,
        'accuracy_finetune': accfinetune,
    }, f)