from os.path import join as ospj

def return_actnet(data_dir):
    filename_categories = ospj(data_dir, 'classInd.txt')
    root_data = data_dir + "/frames"
    filename_imglist_train = ospj(data_dir, 'actnet_train_split.txt')
    filename_imglist_val = ospj(data_dir, 'actnet_val_split.txt')
    prefix = 'image_{:05d}.jpg'

    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_fcvid(data_dir):
    filename_categories = ospj(data_dir, 'classInd.txt')
    root_data = data_dir + "/frames"
    filename_imglist_train = ospj(data_dir, 'fcvid_train_split.txt')
    filename_imglist_val = ospj(data_dir, 'fcvid_val_split.txt')
    prefix = 'image_{:05d}.jpg'

    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_minik(data_dir):
    filename_categories = ospj(data_dir, 'minik_classInd.txt')
    root_data = data_dir + "/frames"
    filename_imglist_train = ospj(data_dir, 'mini_train_videofolder.txt')
    filename_imglist_val = ospj(data_dir, 'mini_val_videofolder.txt')
    prefix = 'image_{:05d}.jpg'

    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_ucf101(data_dir):
    filename_categories = ospj(data_dir, 'ucf101_classInd.txt')
    root_data = data_dir + "/frames"
    filename_imglist_train = ospj(data_dir, 'ucf101_train_videofolder_1.txt')
    filename_imglist_val = ospj(data_dir, 'ucf101_val_videofolder_1.txt')
    prefix = 'image_{:05d}.jpg' #前缀

    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_dataset(dataset, data_dir):
    dict_single = {'actnet': return_actnet, 'fcvid': return_fcvid, 'minik': return_minik, 'ucf101': return_ucf101}
    if dataset in dict_single:
        file_categories, file_imglist_train, file_imglist_val, root_data, prefix = dict_single[dataset](data_dir)
    else:
        raise ValueError('Unknown dataset ' + dataset)

    #类别文件读取，rstrip()函数用来删除 string 字符串末尾的指定字符,默认为空白符，从而得到类别列表
    if isinstance(file_categories, str):
        with open(file_categories) as f:
            lines = f.readlines()
        categories = [item.rstrip() for item in lines]
    else:  # number of categories
        categories = [None] * file_categories
    #得到一共多少类别
    n_class = len(categories)
    print('{}: {} classes'.format(dataset, n_class))
    return n_class, file_imglist_train, file_imglist_val, root_data, prefix
