from __future__ import print_function, division
import os
import time
import subprocess
from tqdm import tqdm
import argparse
from multiprocessing import Pool

parser = argparse.ArgumentParser(description="Dataset processor: Video->Frames")
parser.add_argument("dir_path", type=str, help="original dataset path")#存放视频的路径
parser.add_argument("dst_dir_path", type=str, help="dest path to save the frames")#存放视频帧的路径
parser.add_argument("--prefix", type=str, default="image_%05d.jpg", help="output image type")#输出图片前缀
parser.add_argument("--accepted_formats", type=str, default=[".mp4", ".mkv", ".webm", ".avi"], nargs="+",
                    help="list of input video formats")#接受的视频文件格式类型
parser.add_argument("--begin", type=int, default=0)#从第几个视频开始？
parser.add_argument("--end", type=int, default=666666666)#从第几个视频结束？
parser.add_argument("--file_list", type=str, default="")
parser.add_argument("--frame_rate", type=int, default=-1)#应该是提取视频帧的比率吧
parser.add_argument("--num_workers", type=int, default=16)#同时并行工作的区间
parser.add_argument("--dry_run", action="store_true")
parser.add_argument("--parallel", action="store_true")#并行跑，应该是
args = parser.parse_args()


def par_job(command):
    if args.dry_run:
        print(command)
    else:
        subprocess.call(command, shell=True)


if __name__ == "__main__":
    t0 = time.time()
    dir_path = args.dir_path # 存放视频目录总路径
    dst_dir_path = args.dst_dir_path# 保存帧的目标地址

    for class_name in os.listdir(dir_path):
        class_path = os.path.join(dir_path, class_name)
        if not os.path.isdir(class_path):
            print("没这东西")
            exit()
        dst_class_path = os.path.join(dst_dir_path, class_name)
        if not os.path.exists(dst_class_path):
            os.mkdir(dst_class_path)

        cmd_list = []

        # 取相关的类里边的视频
        for file_name in os.listdir(class_path):

            name, ext = os.path.splitext(file_name)  # 分离文件名与扩展名
            dst_directory_path = os.path.join(dst_class_path, name)  # 合成输出帧的所属视频文件夹名

            # 找到视频文件路径
            video_file_path = os.path.join(class_path, file_name)
            # 如果输出文件夹的路径不存在就创造一个文件夹
            if not os.path.exists(dst_directory_path):
                os.makedirs(dst_directory_path, exist_ok=True)
            # 添加帧的比率
            if args.frame_rate > 0:
                frame_rate_str = "-r %d" % args.frame_rate
            else:
                frame_rate_str = ""
            cmd = 'ffmpeg -nostats -loglevel 0 -i {} -vf scale=-1:360 {} {}/{}'.format(video_file_path, frame_rate_str,
                                                                                       dst_directory_path, args.prefix)
            if not args.parallel:
                if args.dry_run:
                    print(cmd)
                else:
                    subprocess.call(cmd, shell=True)
            cmd_list.append(cmd)

            # #对视频目录生成一个文件名字列表
            # if args.file_list == "":
            #     file_names = sorted(os.listdir(file_name))
            # else:
            #     file_names = [x.strip() for x in open(args.file_list).readlines()]
            #
            # #感觉是视频名称列表中有没在默认视频列表的视频类型就把它添加到删除列表中
            # del_list = []
            # for i, cld_file_name in enumerate(file_names):
            #     if not any([x in cld_file_name for x in args.accepted_formats]):
            #         del_list.append(i)
            #
            # #整理出不在删除列表且需要转化成帧的视频帧
            # file_names = [x for i, x in enumerate(file_names) if i not in del_list]
            # file_names = file_names[args.begin:args.end + 1]
            # print("%d videos to handle (after %d being removed)" % (len(file_names), len(del_list)))
            #
            # cmd_list = []
            # #tqdm是加循环条
            # for file_name in tqdm(file_names):
            #
            #     name, ext = os.path.splitext(file_name)#分离文件名与扩展名
            #     dst_directory_path = os.path.join(dst_class_path, name)#合成输出帧的所属视频文件夹名
            #
            #     #找到视频文件路径
            #     video_file_path = os.path.join(class_path, file_name)
            #     #如果输出文件夹的路径不存在就创造一个文件夹
            #     if not os.path.exists(dst_directory_path):
            #         os.makedirs(dst_directory_path, exist_ok=True)
            #     #添加帧的比率
            #     if args.frame_rate > 0:
            #         frame_rate_str = "-r %d" % args.frame_rate
            #     else:
            #         frame_rate_str = ""
            #     cmd = 'ffmpeg -nostats -loglevel 0 -i {} -vf scale=-1:360 {} {}/{}'.format(video_file_path, frame_rate_str,
            #                                                                                dst_directory_path, args.prefix)
            #     if not args.parallel:
            #         if args.dry_run:
            #             print(cmd)
            #         else:
            #             subprocess.call(cmd, shell=True)
            #     cmd_list.append(cmd)

            #并行操作
            if args.parallel:
                with Pool(processes=args.num_workers) as pool:
                    with tqdm(total=len(cmd_list)) as pbar:
                        for _ in tqdm(pool.imap_unordered(par_job, cmd_list)):
                            pbar.update()
            t1 = time.time()
            print("Finished in %.4f seconds" % (t1 - t0))
            os.system("stty sane")
