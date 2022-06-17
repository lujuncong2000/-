import os
import torch.distributed as dist

from ops.transforms import *
import hydra
import cv2
from PIL import Image
from models.gfv_net import GFV

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

def main():
    # load model
    model = GFV()
    print(model)
    # resume from checkpoint of previous training stage
    resume_ckpt = torch.load('g:/biyesheji/AdaFocus-main/Experiments/checkpoint.pth.tar', map_location='cpu')
    print('best preformance: ',resume_ckpt['best_acc'])
    model.glancer.load_state_dict(resume_ckpt['glancer'])
    model.focuser.load_state_dict(resume_ckpt['focuser'], strict=False)
    model.classifier.load_state_dict(resume_ckpt['fc'])
    model.focuser.policy.policy.load_state_dict(resume_ckpt['policy'])
    model.focuser.policy.policy_old.load_state_dict(resume_ckpt['policy'])

    scale_size = 256  # input_size * 256 // 224, input_size默认是224，规模大小
    crop_size = 224  # input_size大小是224，裁剪大小
    input_mean = [0.485, 0.456, 0.406]
    input_std = [0.229, 0.224, 0.225]  # 输入的均值和方差，降低过拟合

    model = torch.nn.DataParallel(model).cuda()
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
            input_prime = torch.nn.functional.interpolate(input, (224, 224))
            input_prime = input_prime.cuda()
            # out = model(input)
            # 取得输出与预测值
            out, pred = model(input=input, scan=input_prime, training=False, backbone_pred=False, one_step=True, gpu=None)

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




