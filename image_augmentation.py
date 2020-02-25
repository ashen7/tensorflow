import random
from functools import reduce

import cv2 as cv
import numpy as np
from imgaug import augmenters as iaa  #数据增强

def image_augment(image):
    # 颜色通道随机改变
    seq_channel_shuffle = iaa.Sequential([
        iaa.ChannelShuffle()
    ])

    # 添加高斯噪声 让高频部分失真(避免过拟合)
    # scale: 噪声比例
    # per_channel: 是否每一个通道
    seq_gaussian_noise = iaa.Sequential([
        iaa.AdditiveGaussianNoise(scale=0.2*255, per_channel=True)
    ])

    # 随机丢失一部分像素 像素置0
    seq_dropout = iaa.Sequential([
        iaa.Dropout(0.2, per_channel=True)
    ])

    # 上下翻转 p 每张图片翻转的概率
    seq_flip_ud = iaa.Sequential([
        iaa.Flipud(1)
    ])

    # 左右翻转 p 每张图片翻转的概率
    seq_flip_lr = iaa.Sequential([
        iaa.Fliplr(1)
    ])

    # 子通道随机加减像素值
    # channels 那些通道要进行操作
    # children 选出来的通道进行什么操作
    seq_channel = iaa.WithChannels(
        channels = [0, 1],
        children=iaa.Add((random.randint(-60, -20), random.randint(20, 60)))
    )

    # 增加雪地效果
    seq_fast_snowy_land_scape = iaa.Sequential([
        iaa.FastSnowyLandscape(lightness_threshold=200)
    ])

    # 向左旋转90度
    seq_rot90l = iaa.Sequential([
        iaa.Rot90(k=1)
    ])

    # 向右旋转90度
    seq_rot90r = iaa.Sequential([
        iaa.Rot90(k=3)
    ])

    image_list = list()
    for i in range(10):
        if i == 0:
            image_list.append(image)
        elif i == 1:
            image_list.append(seq_channel_shuffle.augment_image(image))
        elif i == 2:
            image_list.append(seq_gaussian_noise.augment_image(image))
        elif i == 3:
            image_list.append(seq_dropout.augment_image(image))
        elif i == 4:
            image_list.append(seq_flip_ud.augment_image(image))
        elif i == 5:
            image_list.append(seq_flip_lr.augment_image(image))
        elif i == 6:
            image_list.append(seq_channel.augment_image(image))
        elif i == 7:
            image_list.append(seq_fast_snowy_land_scape.augment_image(image))
        elif i == 8:
            image_list.append(seq_rot90l.augment_image(image))
        elif i == 9:
            image_list.append(seq_rot90r.augment_image(image))

    # image_list = list(map(lambda img: cv.resize(img.astype(np.uint8), (400,200)), image_list))
    # a = reduce(lambda a, b: np.hstack((a, b)), image_list[:2])
    # b = reduce(lambda a, b: np.hstack((a, b)), image_list[2:4])
    # c = reduce(lambda a, b: np.hstack((a, b)), image_list[4:6])
    # d = reduce(lambda a, b: np.hstack((a, b)), image_list[6:8])
    # e = reduce(lambda a, b: np.hstack((a, b)), image_list[8:])
    # a = np.vstack((a, b))
    # a = np.vstack((a, c))
    # a = np.vstack((a, d))
    # a = np.vstack((a, e))
    # cv.imshow('a',a)
    # cv.waitKey(0)
    # exit(0)

    return image_list

def image_augment_batch(image_list, image_num):
    assert len(image_list) > 0
    assert image_num == len(image_list)
    # 颜色通道随机改变
    seq_channel_shuffle = iaa.Sequential([
        iaa.ChannelShuffle()
    ])

    # 添加高斯噪声 让高频部分失真(避免过拟合)
    # scale: 噪声比例
    # per_channel: 是否每一个通道
    seq_gaussian_noise = iaa.Sequential([
        iaa.AdditiveGaussianNoise(scale=0.2*255, per_channel=True)
    ])

    # 随机丢失一部分像素 像素置0
    seq_dropout = iaa.Sequential([
        iaa.Dropout(0.2, per_channel=True)
    ])

    # 上下翻转 p 每张图片翻转的概率
    seq_flip_ud = iaa.Sequential([
        iaa.Flipud(1)
    ])

    # 左右翻转 p 每张图片翻转的概率
    seq_flip_lr = iaa.Sequential([
        iaa.Fliplr(1)
    ])

    # 子通道随机加减像素值
    # channels 那些通道要进行操作
    # children 选出来的通道进行什么操作
    seq_channel = iaa.WithChannels(
        channels = [0, 1],
        children=iaa.Add((random.randint(-60, -20), random.randint(20, 60)))
    )

    # 增加雪地效果
    seq_fast_snowy_land_scape = iaa.Sequential([
        iaa.FastSnowyLandscape(lightness_threshold=200)
    ])

    # 向左旋转90度
    seq_rot90l = iaa.Sequential([
        iaa.Rot90(k=1)
    ])

    # 向右旋转90度
    seq_rot90r = iaa.Sequential([
        iaa.Rot90(k=3)
    ])

    new_image_list = list()
    for n in range(image_num):
        for i in range(10):
            if i == 0:
                new_image_list.append(image_list[n])
            elif i == 1:
                new_image_list.append(seq_channel_shuffle.augment_image(image_list[n]))
            elif i == 2:
                new_image_list.append(seq_gaussian_noise.augment_image(image_list[n]))
            elif i == 3:
                new_image_list.append(seq_dropout.augment_image(image_list[n]))
            elif i == 4:
                new_image_list.append(seq_flip_ud.augment_image(image_list[n]))
            elif i == 5:
                new_image_list.append(seq_flip_lr.augment_image(image_list[n]))
            elif i == 6:
                new_image_list.append(seq_channel.augment_image(image_list[n]))
            elif i == 7:
                new_image_list.append(seq_fast_snowy_land_scape.augment_image(image_list[n]))
            elif i == 8:
                new_image_list.append(seq_rot90l.augment_image(image_list[n]))
            elif i == 9:
                new_image_list.append(seq_rot90r.augment_image(image_list[n]))

    return new_image_list

