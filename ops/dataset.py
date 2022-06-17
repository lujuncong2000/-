import torch.utils.data as data
import torch

from PIL import Image
import os
import numpy as np
from numpy.random import randint


class VideoRecord(object):
    """
    提供简单的封装，用来返回关于数据的一些信息（帧路径、帧数量、帧标签）
    /home/UserData/ljc/dataset/frames/PoleVault/v_PoleVault_g17_c07 tensor([67, -1, -1])
    /home/UserData/ljc/dataset/frames/Typing/v_Typing_g23_c02 tensor([94, -1, -1])
    """
    def __init__(self, row):
        #_data：获取每一行数据
        self._data = row
        # 初始化，_labels = tensor（[-1,-1,-1]）
        self._labels = torch.tensor([-1, -1, -1])
        #大概意思是生成一个1的长度的列表，我感觉是labels = [0]
        labels = sorted(list(set([int(x) for x in self._data[2:]])))
        #？？？不能理解_label[0]=0,_label[1]=1
        for i, l in enumerate(labels):
            self._labels[i] = l

    @property
    def path(self):
        # 直接返回帧路径
        return self._data[0]

    @property
    def num_frames(self):
        # 直接返回对应的帧数量
        return int(self._data[1])

    @property
    def label(self):
        # _labels[-2]是最后一个值，只要有分类的帧
        if self._labels[-2] > -1:
            # 中间那一联有东西的话，就把他按照大小随机搞乱
            if self._labels[-1] > -1:
                return self._labels[torch.randperm(self._labels.shape[0])]
            else:
                # 随机一个数，大于0.5的话，就返回[[0,1,2]]，否则返回[[1,0,2]]
                if torch.rand(1) > 0.5:
                    return self._labels[[0,1,2]]
                else:
                    return self.label[[1,0,2]]
        else:
            return self._labels

#重点看这个！！！
class TSNDataSet(data.Dataset):
    """
    __init__参数：root_path：帧文件夹路径 list_file：形成的训练或者测试帧路径文件
                transform：转换函数 random_shift：布尔型，当设置为True时对训练集进行采样，设置为False时对验证集进行采样
                test_mode：布尔型，默认为False，当设置为True时即即对测试集进行采样remove_missing：布尔型，默认为False。与test_mode在同一个判断条件下，对数据进行读取。
                dense_sample：布尔型，设置为True时进行密集采样twice_sample：布尔型，设置为True时进行二次采样
                dataset：partial_fcvid_eval：
                partial_ratio：部分比率ada_reso_skip：
                reso_list：random_crop：
                center_crop：ada_crop_list：
                rescale_to：policy_input_offset：
                save_meta：
    重点还是用来加载数据
    """
    def __init__(self, root_path, list_file,
                 num_segments=3, image_tmpl='img_{:05d}.jpg', transform=None,
                 random_shift=True, test_mode=False,
                 remove_missing=False, dense_sample=False, twice_sample=False,
                 dataset=None, partial_fcvid_eval=False, partial_ratio=None,
                 ada_reso_skip=False, reso_list=None, random_crop=False, center_crop=False, ada_crop_list=None,
                 rescale_to=224, policy_input_offset=None, save_meta=False):

        self.root_path = root_path

        self.list_file = \
            ".".join(list_file.split(".")[:-1]) + "." + list_file.split(".")[-1]  # TODO
        self.num_segments = num_segments
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.remove_missing = remove_missing
        self.dense_sample = dense_sample  # using dense sample as I3D
        self.twice_sample = twice_sample  # twice sample for more validation

        # TODO(yue)
        self.dataset = dataset
        self.partial_fcvid_eval = partial_fcvid_eval
        self.partial_ratio = partial_ratio
        self.ada_reso_skip = ada_reso_skip
        self.reso_list = reso_list
        self.random_crop = random_crop
        self.center_crop = center_crop
        self.ada_crop_list = ada_crop_list
        self.rescale_to = rescale_to
        self.policy_input_offset = policy_input_offset
        self.save_meta = save_meta

        if self.dense_sample:
            print('=> Using dense sample for the dataset...')
        if self.twice_sample:
            print('=> Using twice sample for the dataset...')

        self._parse_list()

    def _load_image(self, directory, idx):
        """
        :param directory: 大概是帧文件夹
        :param idx:
        :return:PIL类型的图片，是RGB类型，三通道图片，大小大概是[3,]
        """
        try:
            return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert('RGB')]
        except Exception:
            print('error loading image:', os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
            return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')]

    def _parse_list(self):
        """
        self.video_list是一个长度为训练数据数量的列表。每个值都是VIDEORecord对象，
        包含一个列表和3个属性，列表长度为3，用空格键分割，分别为帧路径、该视频含有多少帧和帧标签。
        其中有一个判断机制，保证帧数需大于3。
        :return: 
        """# check the frame number is large >3:分隔符号一般都是“ ”""""""
        splitter = "," if self.dataset in ["actnet", "fcvid"] else " "
        if self.dataset == "kinetics":
            splitter = ";"
        #将每一行分割为三个值
        tmp = [x.strip().split(splitter) for x in open(self.list_file)]

        if any(len(items) >= 3 for items in tmp) and self.dataset == "minik":
            tmp = [[splitter.join(x[:-2]), x[-2], x[-1]] for x in tmp]

        if self.dataset == "kinetics":
            tmp = [[x[0], x[-2], x[-1]] for x in tmp]

        # 就是取每一行的值
        if not self.test_mode or self.remove_missing:
            tmp = [item for item in tmp if int(item[1]) >= 3]

        if self.partial_fcvid_eval and self.dataset == "fcvid":
            tmp = tmp[:int(len(tmp) * self.partial_ratio)]

        # video_list就是将每一行视频帧信息都换成VideoRecord类型的对象，组成一个序列
        self.video_list = [VideoRecord(item) for item in tmp]

        if self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
            for v in self.video_list:
                v._data[1] = int(v._data[1]) / 2
        print('video number:%d' % (len(self.video_list)))

    def _sample_indices(self, record):
        """
        _sample_indices函数功能在于实现TSN的密集采样或者稀疏采样，返回的是采样的帧数列表
        :param record: VideoRecord
        :return: list
        """
        if self.dense_sample:  # i3d dense sample密集随机采样
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        else:  # normal sample稀疏随机采样
            average_duration = record.num_frames // self.num_segments
            if average_duration > 0:
                offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                                                                                  size=self.num_segments)#[0,1,2]*50+
            elif record.num_frames > self.num_segments:
                offsets = np.sort(randint(record.num_frames, size=self.num_segments))
            else:
                offsets = np.array(
                    list(range(record.num_frames)) + [record.num_frames - 1] * (self.num_segments - record.num_frames))
            return offsets + 1

    def _get_val_indices(self, record):
        #大概是用来对训练帧的采样
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        else:
            if record.num_frames > self.num_segments:
                # 如果视频帧比16大，则每16帧选一帧，选中间一帧，返回选帧列表
                tick = record.num_frames / float(self.num_segments)
                offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            else:
                offsets = np.array(
                    list(range(record.num_frames)) + [record.num_frames - 1] * (self.num_segments - record.num_frames))
            return offsets + 1

    def _get_test_indices(self, record):
        # 大概用来对测试帧的采样，返回的是帧数列表
        if self.dense_sample:
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_list = np.linspace(0, sample_pos - 1, num=10, dtype=int)
            offsets = []
            for start_idx in start_list.tolist():
                offsets += [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        elif self.twice_sample:
            tick = record.num_frames / float(self.num_segments)

            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)] +
                               [int(tick * x) for x in range(self.num_segments)])

            return offsets + 1
        else:
            tick = record.num_frames / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            return offsets + 1

    def __getitem__(self, index):
        """
        该函数会在TSNDataSet初始化之后执行，功能在于调用执行采样函数,并且调用get方法，得到TSNDataSet的返回

        :param index:
        :return: 返回一个[48,224,224]的tensor数组，和label标签
        """
        # record变量读取的是video_list的第index个数据，包含该视频所在的文件地址、视频包含的帧数和视频所属的分类。
        record = self.video_list[index]
        # check this is a legit video folder
        if self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
            file_name = self.image_tmpl.format(int(record.path), 'x', 1)
            full_path = os.path.join(self.root_path, '{:06d}'.format(int(record.path)), file_name)
        else:
            file_name = self.image_tmpl.format(1)
            full_path = os.path.join(self.root_path, record.path, file_name)

        err_cnt = 0
        while not os.path.exists(full_path):
            err_cnt += 1
            if err_cnt > 3:
                exit("Sth wrong with the dataloader to get items. Check your data path. Exit...")
            print('################## Not Found:', os.path.join(self.root_path, record.path, file_name))
            index = np.random.randint(len(self.video_list))
            record = self.video_list[index]
            if self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
                file_name = self.image_tmpl.format(int(record.path), 'x', 1)
                full_path = os.path.join(self.root_path, '{:06d}'.format(int(record.path)), file_name)
            else:
                file_name = self.image_tmpl.format(1)
                full_path = os.path.join(self.root_path, record.path, file_name)
        # 大概是看是否是训练还是测试数据集，选择密集采样还是稀疏采样，将采样得到的结果保存到segment_indices中
        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)

        return self.get(record, segment_indices)

    def get(self, record, indices):
        """
         大概就是得到一个图片，对提取到的帧序号列表进行遍历，找到每一个帧对应的图片，添加到Images列表中。
         之后对提到的images进行数据集变形，返回变形后的数据集和对应的类型标签。
        """
        images = list()
        for seg_ind in indices:
            images.extend(self._load_image(record.path, int(seg_ind)))

        # process_data的大小为[C*N,H,W], N=16, [48,224,224]
        process_data = self.transform(images)
        if self.ada_reso_skip:
            return_items = [process_data]
            if self.random_crop:
                rescaled = [self.random_crop_proc(process_data, (x, x)) for x in self.reso_list[1:]]
            elif self.center_crop:
                rescaled = [self.center_crop_proc(process_data, (x, x)) for x in self.reso_list[1:]]
            else:
                rescaled = [self.rescale_proc(process_data, (x, x)) for x in self.reso_list[1:]]
            return_items = return_items + rescaled
            if self.save_meta:
                return_items = return_items + [record.path] + [indices]  # [torch.tensor(indices)]
            return_items = return_items + [record.label]

            return tuple(return_items)
        else:
            if self.rescale_to == 224:
                rescaled = process_data
            else:
                x = self.rescale_to
                if self.random_crop:
                    rescaled = self.random_crop_proc(process_data, (x, x))
                elif self.center_crop:
                    rescaled = self.center_crop_proc(process_data, (x, x))
                else:
                    rescaled = self.rescale_proc(process_data, (x, x))

            return rescaled, record.label

    # TODO(yue)
    # (NC, H, W)->(NC, H', W')
    def rescale_proc(self, input_data, size):
        return torch.nn.functional.interpolate(input_data.unsqueeze(1), size=size, mode='nearest').squeeze(1)

    def center_crop_proc(self, input_data, size):
        h = input_data.shape[1] // 2
        w = input_data.shape[2] // 2
        return input_data[:, h - size[0] // 2:h + size[0] // 2, w - size[1] // 2:w + size[1] // 2]

    def random_crop_proc(self, input_data, size):
        H = input_data.shape[1]
        W = input_data.shape[2]
        input_data_nchw = input_data.view(-1, 3, H, W)
        batchsize = input_data_nchw.shape[0]
        return_list = []
        hs0 = np.random.randint(0, H - size[0], batchsize)
        ws0 = np.random.randint(0, W - size[1], batchsize)
        for i in range(batchsize):
            return_list.append(input_data_nchw[i, :, hs0[i]:hs0[i] + size[0], ws0[i]:ws0[i] + size[1]])
        return torch.stack(return_list).view(batchsize * 3, size[0], size[1])

    def __len__(self):
        return len(self.video_list)