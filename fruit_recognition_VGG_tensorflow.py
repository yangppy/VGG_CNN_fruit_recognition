#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：简历项目
@File    ：fruit_recognition_VGG_tensorflow.py
@IDE     ：PyCharm
@Author  ：xiaoyang
@Date    ：2023/3/18 21:43
'''

import tensorflow as tf
import numpy as np

# 卷积块的参数
conv_arch = ((2, 64), (2, 128), (3, 256), (3, 512), (3, 512))
# 数据集目录
data_root_path = "fruits/"


# 文件读取器
def data_load(data_dir, img_height, img_width, batch_size):
    # 加载训练集
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        validation_split=0.1,  # 测试集占10%
        seed=123,
        subset='training',
        image_size=(img_height, img_width),
        batch_size=batch_size)
    # 加载测试集
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        validation_split=0.1,  # 测试集占10%
        seed=123,
        subset='validation',
        image_size=(img_height, img_width),
        batch_size=batch_size)
    # 返回处理之后的训练集、验证集和类名
    return train_ds, val_ds


# VGG 块创建
def vgg_block(num_conv, num_filters):
    # 序列模型
    blk = tf.keras.models.Sequential()
    for _ in range(num_conv):
        # 设置卷积层
        blk.add(tf.keras.layers.Conv2D(num_filters, kernel_size=3, padding='same', activation="relu"))
    blk.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    return blk


# 构建模型
def vgg(conv_arch):
    # 序列模型
    net = tf.keras.models.Sequential()
    # 生成卷积池化部分
    for (num_convs, num_filters) in conv_arch:
        net.add(vgg_block(num_convs, num_filters))
    # 全连接层
    net.add(tf.keras.models.Sequential([
        # 展平
        tf.keras.layers.Flatten(),
        # 全连接
        tf.keras.layers.Dense(4096, activation="relu"),
        # 随机失活
        tf.keras.layers.Dropout(0.5),
        # 全连接层
        tf.keras.layers.Dense(4096, activation="relu"),
        # 随机失活
        tf.keras.layers.Dropout(0.5),
        # 输出层
        tf.keras.layers.Dense(10, activation="softmax")
    ]))
    return net


def train():
    net = vgg(conv_arch)
    # 加载数据集
    train_ds, test_ds = data_load(data_root_path, 224, 224, 128)
    # 指定优化器，损失函数和评价指标
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0)

    net.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy",
                metrics=['accuracy'])
    net.fit(train_ds, batch_size=128, epochs=3, verbose=1, validation_data=test_ds)


if __name__ == '__main__':
    train()
