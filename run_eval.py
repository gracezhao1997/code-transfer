import os
from tool.my_utils import available_devices, format_devices, set_logger
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import torch
import csv
from transformers import CLIPProcessor, CLIPTokenizer, CLIPModel
from tool.score_clip import get_t2m, get_m2m
from tool.score_stucture import calculate_l2_given_paths, calculate_ssim, calculate_mse, calculate_psnr, calculate_lpips


_ = torch.manual_seed(42)
import jsonlines
import numpy as np
import logging

version = "openai/clip-vit-large-patch14"
device = "cuda"
processor = CLIPProcessor.from_pretrained(version)
tokenizer = CLIPTokenizer.from_pretrained(version)
clip = CLIPModel.from_pretrained(version).to(device)

def cal_score(type, text_list, translate_list, source_list):
    '''

    Args:
        text_list: list of str
        translate_list: list of dir
        source_list: list of dir

    Returns:

    '''
    assert len(text_list) == len(translate_list) == len(source_list)
    score = []

    for (text, translate_dir, source_dir) in zip(text_list, translate_list, source_list):
        # 若这个video未应用这种方法，则跳过
        if not os.path.exists(translate_dir):
            score.append(0)
            continue

        img_paths = []
        txt_inputs = []
        for name in os.listdir(translate_dir):    # '0.png', '1.png', ..., '7.png'
            img_paths.append(os.path.join(translate_dir, name))       # './outputs/baseline1/car10-a red car/results/fate/0.png'
            txt_inputs.append(text)
        print('Sample Cnt: {}'.format(len(img_paths)))

        if type == 'clip-text':
            mean_score = get_t2m(img_paths, txt_inputs, 1)
        elif type == 'clip-time':
            mean_score = get_m2m(img_paths)
        elif type == 'l2':
            mean_score = calculate_l2_given_paths(translate_dir, source_dir)
        elif type == 'mse':
            mean_score = calculate_mse(translate_dir, source_dir)
        elif type == 'ssim':
            mean_score = calculate_ssim(translate_dir, source_dir)
        elif type == 'psnr':
            mean_score = calculate_psnr(translate_dir, source_dir)
        elif type == 'lpips':
            mean_score = calculate_lpips(translate_dir, source_dir)
        score.append(mean_score)

        logging.info(f'{type} SCORE-video{i}-{text}: {mean_score}')
        print(f'{type} SCORE-video{i}-{text}: {mean_score}')
    return score

if __name__ == '__main__':

    # 之前保存结果的路径
    out_root = '/workspace/home/zhaomin/cephfs-thu/zhaomin/projects/text-to-image/Controlvideo_project_code/outputs/baseline_3'

    # 待测video_list
    video_list = 'videos_final.jsonl'
    with jsonlines.open(video_list, 'r') as reader:  # The reader and can be used directly. It contains the lines in the json file.
        videos = [video for video in reader]  # a list of dict
    reader.close()

    # 待测试的方法
    methods = ['control_temp_self-80','control_temp_self-300','control_temp_self-500','control_temp_self-1000' ]

    type_scores = ['clip-text']

    # for saving scv
    save_tables = np.zeros(shape=(len(type_scores), len(videos),1))

    # 对每种方法在video_list上计算得分，并保存结果
    # 首先提取三个list，包含每个视频对应的信息：
    text_list = []        # text_list: 每个视频对应的prompt
    translate_list = []        # translate_list: 每个视频对应的输出结果的目录
    source_list = []        # source_list: 每个视频对应的原视频目录
    for i, video in enumerate(videos):
        text_list.append(video['target'][0])
        if video['type'] == "canny":
            method = methods[0]
        elif video['type'] == "hed":
            method = methods[1]
        elif video['type'] == "depth":
            method = methods[2]
        elif video['type'] == "pose":
            method = methods[3]

        translate_list.append(os.path.join(out_root, f"{video['name']}-{video['target'][0]}", f"results/{method}"))
        source_list.append(os.path.join(out_root, f"{video['name']}-{video['target'][0]}", f"results/origin"))

    set_logger(os.path.join(out_root, "ours_score"), f'ours.txt')
    for j, type_score in enumerate(type_scores):
        save_tables[j, :, 0] = cal_score(type_score, text_list, translate_list, source_list)

    # 存储csv
    methods.append('ours')
    for idx, type_score in enumerate(type_scores): # 计算ours列 & 并对每种control取平均
        save_table = save_tables[idx, :, :]
        average_score = np.mean(save_table, axis=0).reshape(1, -1)
        save_table = np.vstack((save_table, average_score))
        print(f'The average clip text score across dataset is {float(average_score):.3f}')
        with open(os.path.join(out_root, f'ours_score/{type_score}.csv'), "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(methods)                # 先写入columns_name
            writer.writerows(save_table)        # 写入多行用writerows
