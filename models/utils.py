
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

import torchvision
import torch
import numpy as np

def prep_a_net(model_name, shall_pretrain):
    model = getattr(torchvision.models, model_name)(shall_pretrain)
    if "resnet" in model_name:
        model.last_layer_name = 'fc'
    elif "mobilenet_v2" in model_name:
        model.last_layer_name = 'classifier'
    return model

def zero_pad(im, pad_size):
    """Performs zero padding (CHW format)."""
    pad_width = ((0, 0), (pad_size, pad_size), (pad_size, pad_size))
    return np.pad(im, pad_width, mode="constant")

def random_crop(im, size, pad_size=0):
    """
    Performs random crop (CHW format).
    随机截取函数， 返回截取图片部分
    """
    if pad_size > 0:
        im = zero_pad(im=im, pad_size=pad_size)
    h, w = im.shape[1:]
    if size == h:
        # 如果要裁取的大小与图片大小相同，则直接返回图片
        return im
    y = np.random.randint(0, h - size)# 随机返回一个y坐标
    x = np.random.randint(0, w - size)# 随即返回一个x坐标
    im_crop = im[:, y : (y + size), x : (x + size)]# [C,H,W]
    assert im_crop.shape[1:] == (size, size)
    return im_crop

def get_patch(images, action_sequence, patch_size):
    """Get small patch of the original image"""
    batch_size = images.size(0)#32
    image_size = images.size(2)#224

    patch_coordinate = torch.floor(action_sequence * (image_size - patch_size)).int()#边界，防止取过了
    patches = []
    for i in range(batch_size):
        #取到每一帧图片
        per_patch = images[i, :,
                    (patch_coordinate[i, 0].item()): ((patch_coordinate[i, 0] + patch_size).item()),
                    (patch_coordinate[i, 1].item()): ((patch_coordinate[i, 1] + patch_size).item())]
        # 更改后尺寸[1,b,h,w]
        patches.append(per_patch.view(1, per_patch.size(0), per_patch.size(1), per_patch.size(2)))
    # 为啥要补0
    return torch.cat(patches, 0)