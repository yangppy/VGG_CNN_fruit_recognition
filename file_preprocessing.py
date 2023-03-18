#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：简历项目 
@File    ：file_preprocessing.py
@IDE     ：PyCharm 
@Author  ：xiaoyang
@Date    ：2023/3/18 9:46 
'''

import os

name_dict = {"apple": 0, "banana": 1, "grape": 2, "orange": 3, "pear": 4}
data_root_path = "fruits/"  # 数据所在的根目录
test_file_path = data_root_path + "test.txt"  # 测试文件目录
train_file_path = data_root_path + "train.txt"  # 训练文件目录
name_data_list = {}  # 记录每个类别有哪些图片 key：水果名称 value：图片路径构成的列表


# 将图片路径存入name_data_list字典中
def save_train_test_file(path, name):
    if name not in name_data_list:  # 该类水果不在字典中，则新建一个列表插入字典
        img_list = []
        img_list.append(path)  # 将图片路径存入列表
        name_data_list[name] = img_list  # 将图片列表插入字典
    else:  # 该类水果在字典中，直接添加到列表
        name_data_list[name].append(path)


def run():
    # 遍历数据集下面每个子目录，将图片路径写入上面的字典
    dirs = os.listdir(data_root_path)  # 列出数据集目录下所有的文件和子目录
    for d in dirs:
        full_path = data_root_path + d  # 拼接完整路径
        if os.path.isdir(full_path):  # 是一个子目录
            imgs = os.listdir(full_path)  # 列出子目录下所有的图片文件
            for img in imgs:
                save_train_test_file(full_path + '/' + img, d)  # 拼接图片完整路径
        else:  # 文件不予处理
            pass

    # 将name_data_list字典中的内容写入文件
    # 清空训练集文件和测试集文件
    with open(test_file_path, 'wb') as f:
        pass

    with open(train_file_path, 'wb') as f:
        pass

    # 遍历字典，将字典中的内容写入训练集和测试集
    for name, img_list in name_data_list.items():
        i = 0
        num = len(img_list)  # 获取每个类别的图片数量
        print("%s:%d张" % (name, num))
        for img in img_list:
            if i % 10 == 0:  # 每隔十个写入一次测试集文件
                with open(test_file_path, "a") as f:  # 一追加模式打开测试集文件
                    line = "%s\t%d\n" % (img, name_dict[name])  # 拼一行
                    f.write(line)  # 写入文件
            else:  # 写入训练集
                with open(train_file_path, "a") as f:  # 以追加模式打开测试集文件
                    line = "%s\t%d\n" % (img, name_dict[name])  # 拼成一行
                    f.write(line)  # 写入文件
            i += 1

    print("训练集测试集分类完成！")


if __name__ == '__main__':
    run()
