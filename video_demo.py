import torch_pruning as pruning
import os
import torch.distributed as dist

from ops.transforms import *
import hydra
import cv2
from PIL import Image
from models.gfv_net import GFV, RecurrentClassifier

catigories = ["ApplyEyeMakeup","ApplyLipstick","Archery","BabyCrawling","BalanceBeam","BandMarching","BaseballPitch",
              "Basketball","BasketballDunk","BenchPress","Biking","Billiards","BlowDryHair","BlowingCandles","BodyWeightSquats",
              "Bowling","BoxingPunchingBag","BoxingSpeedBag","BreastStroke","BrushingTeeth","CleanAndJerk","CliffDiving",
              "CricketBowling","CricketShot","CuttingInKitchen","Diving","Drumming","Fencing","FieldHockeyPenalty",
              "FloorGymnastics","FrisbeeCatch","FrontCrawl","GolfSwing","Haircut","Hammering","HammerThrow",
              "HandstandPushups","HandstandWalking","HeadMassage","HighJump","HorseRace","HorseRiding","HulaHoop",
              "IceDancing","JavelinThrow","JugglingBalls","JumpingJack","JumpRope","Kayaking","Knitting","LongJump",
              "Lunges","MilitaryParade","Mixing","MoppingFloor","Nunchucks","ParallelBars","PizzaTossing","PlayingCello",
              "PlayingDaf","PlayingDhol","PlayingFlute","PlayingGuitar","PlayingPiano","PlayingSitar","PlayingTabla",
              "PlayingViolin","PoleVault","PommelHorse","PullUps","Punch","PushUps","Rafting","RockClimbingIndoor",
              "RopeClimbing","Rowing","SalsaSpin","ShavingBeard","Shotput","SkateBoarding","Skiing","Skijet","SkyDiving",
              "SoccerJuggling","SoccerPenalty","StillRings","SumoWrestling","Surfing","Swing","TableTennisShot",
              "TaiChi","TennisSwing","ThrowDiscus","TrampolineJumping","Typing","UnevenBars","VolleyballSpiking",
              "WalkingWithDog","WallPushups","WritingOnBoard","YoYo"]

def prune_model_mixed(model):

    model.cpu()
    slim = pruning.Autoslim(model, inputs=torch.randn(
        1,3,224,224), compression_ratio=0.75)
    config = {
        'layer_compression_ratio': None,
        'norm_rate': 1.0, 'prune_shortcut': 1,
        'dist_type': 'l1', 'pruning_func': 'fpgm'
    }
    slim = pruning.Autoslim(model, inputs=torch.randn(
        1,3,224,224), compression_ratio=0.75)
    slim.base_prunging(config)
    return model

@hydra.main(config_path="conf", config_name="default")
def main(args):
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    # Simply call main_worker function
    main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    print("curent thread rank:", args.rank)
    args.num_classes = 101
    # load model
    model = GFV()
    start_epoch = 0
    resume_ckpt = torch.load('g:/biyesheji/AdaFocus-main/Experiments/origin_resnet50.pth.tar', map_location='cpu')
    print('best preformance: ', resume_ckpt['best_acc'])
    model.focuser.load_state_dict(resume_ckpt['focuser'], strict=False)

    model.focuser.net = prune_model_mixed(model.focuser.net)

    feat_dim = model.glancer.feature_dim + model.focuser.feature_dim # 3328
    model.classifier = RecurrentClassifier(seq_len=16, input_dim=feat_dim, batch_size=args.batch_size,
                                          hidden_dim=args.hidden_dim, num_classes=args.num_classes,
                                          dropout=args.dropout)

    resume_ckpt = torch.load('g:/biyesheji/AdaFocus-main/Experiments/prun75_checkpoint.pth.tar', map_location='cpu')
    print('best preformance: ', resume_ckpt['best_acc'])
    model.glancer.load_state_dict(resume_ckpt['glancer'])
    model.focuser.load_state_dict(resume_ckpt['focuser'], strict=False)
    model.classifier.load_state_dict(resume_ckpt['fc'])
    if args.train_stage == 3:
        model.focuser.policy.policy.load_state_dict(resume_ckpt['policy'])
        model.focuser.policy.policy_old.load_state_dict(resume_ckpt['policy'])
    scale_size = model.scale_size  # input_size * 256 // 224, input_size默认是224，规模大小
    crop_size = model.crop_size  # input_size大小是224，裁剪大小
    input_mean = model.input_mean
    input_std = model.input_std  # 输入的均值和方差，降低过拟合

    # if not torch.cuda.is_available():
    #     print('using CPU, this will be slow')
    # elif args.distributed:
    #     # For multiprocessing distributed, DistributedDataParallel constructor
    #     # should always set the single device scope, otherwise,
    #     # DistributedDataParallel will use all available devices.
    #     if args.gpu is not None:
    #         torch.cuda.set_device(args.gpu)
    #         model.cuda(args.gpu)
    #         # When using a single GPU per process and per单机单卡
    #         # DistributedDataParallel, we need to divide the batch size
    #         # ourselves based on the total number of GPUs we have
    #         args.batch_size = int(args.batch_size / ngpus_per_node)
    #         args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
    #         model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    #         print("Using DDP with specific GPU!")
    #     else:
    #         model.cuda()
    #         # DistributedDataParallel will divide and allocate batch_size to all
    #         # available GPUs if device_ids are not set多机多卡
    #         model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    #         print("Using DDP with all GPUs!")
    # elif args.gpu is not None:
    #     torch.cuda.set_device(args.gpu)
    #     model = model.cuda(args.gpu)
    # else:
    #     #为什么训练阶段是2的时候不需要分布式？
    #     if args.train_stage != 2:
    #         model = torch.nn.DataParallel(model).cuda()
    #         print('Using DP with GPUs')
    #     else:
    #         #将模型加载到GPU上去
    #         model = model.cuda()

    # optimizer = torch.optim.SGD([
    #     {'params': model.module.classifier.parameters()}
    #     ], lr=0,
    #     momentum=args.momentum,
    #     weight_decay=args.weight_decay)
    # cudnn.benchmark = True
    # 需要测试时添加进去
    model.eval()

    num_segments = 8
    normalize = GroupNormalize(input_mean, input_std)
    transform = torchvision.transforms.Compose([
            GroupScale(int(scale_size)),
            GroupCenterCrop(crop_size),
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            normalize])
    video_path = 'g:/biyesheji/AdaFocus-main/Experiments/v_Fencing_g02_c04.avi'
    pil_img_list = list()
    cls_text = ['noFencing', 'Fencing']# 有待商榷
    cls_color = [(0, 255, 0), (0, 0, 255)]

    import time

    cap = cv2.VideoCapture(video_path)
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)# 这一句是可以随意伸缩视频框框大小
    # cv2.resizeWindow("frame", 640, 480)# 这一句是改变窗口大小
    start_time = time.time()
    counter = 0
    frame_numbers = 0
    training_fps = 30
    training_time = 1
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps < 1:
        fps = 30
    duaring = int(fps * training_time / num_segments)
    print(duaring)

    state = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame_numbers += 1
            print(frame_numbers)
            # print(len(pil_img_list))
            # if frame_numbers % duaring == 0 and len(pil_img_list) < 1:
            #     frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            #     pil_img_list.extend([frame_pil])
            # if frame_numbers % duaring == 0 and len(pil_img_list) == 1:
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pil_img_list.extend([frame_pil])
            # pil_img_list.pop(0)
            # pil_img_list.extend([frame_pil])
            input = transform(pil_img_list)
            input = input.unsqueeze(0).cuda()
            input_prime = torch.nn.functional.interpolate(input, (args.glance_size, args.glance_size))
            input_prime = input_prime.cuda()
            # out = model(input)
            # 取得输出与预测值
            out, pred = model(input=input, scan=input_prime, training=False, backbone_pred=False, one_step=True, gpu=args.gpu)

            pil_img_list.pop(0)
            output_index = int(torch.argmax(pred).cpu())
            if output_index == 27:
                state = 1
                print("pred: ", output_index, cls_text[1])
            else:
                state = 0
                print("pred: ", output_index, cls_text[0])
                # output_index = int(torch.argmax(out).cpu())
                # state = output_index

            # 键盘输入空格暂停，输入q退出
            key = cv2.waitKey(1) & 0xff
            if key == ord(" "):
                cv2.waitKey(0)
            if key == ord("q"):
                break
            counter += 1  # 计算帧数
            if (time.time() - start_time) != 0:  # 实时显示帧数
                # 添加的文字 位置 字体 字体大小 字体颜色 字体粗细
                cv2.putText(frame,
                            "{0} {1}".format((cls_text[state]), float('%.1f' % (counter / (time.time() - start_time)))),
                            (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, cls_color[state], 2)
                cv2.imshow('frame', frame)

                counter = 0
                start_time = time.time()
            time.sleep(1 / fps)  # 按原帧率播放
            # time.sleep(2/fps)# observe the output
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()




