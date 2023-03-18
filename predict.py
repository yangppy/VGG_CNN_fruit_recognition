#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：简历项目 
@File    ：predict.py
@IDE     ：PyCharm 
@Author  ：xiaoyang
@Date    ：2023/3/18 20:56 
'''

from PIL import Image
import paddle
import paddle.fluid as fluid
import numpy
import matplotlib.pyplot as plt

# 定义执行器
place = fluid.CPUPlace()
infer_exe = fluid.Executor(place)
model_save_dir = 'fruits/'  # 模型保存路径
name_dict = {"apple": 0, "banana": 1, "grape": 2, "orange": 3, "pear": 4}


# 加载数据
def load_img(path):
    img = paddle.dataset.image.load_and_transform(path, 128, 128, False).astype("float32")
    img = img / 255.0
    return img


infer_imgs = []  # 存放要预测图像数据
test_img = "./fruits/apple0.png"  # 带预测的图片
infer_imgs.append(load_img((test_img)))  # 加载图片，并且将图片数据添加到带预测列表
infer_imgs = numpy.array(infer_imgs)  # 转成数组

# 加载模型
infer_program, feed_target_names, fetch_targets = fluid.io.load_inference_model(model_save_dir, infer_exe)

# 执行预测
results = infer_exe.run(infer_program,  # 执行预测program
                        feed={feed_target_names[0]: infer_imgs},  # 传入带预测图像数据
                        fetch_list=fetch_targets)  # 返回结果
print(results)

result = numpy.argmax(results[0])  # 取出预测结果中概率最大的元素索引值
for k, v in name_dict.items():
    if result == v:
        print("预测结果:", k)

# 显示待预测的图片
img = Image.open(test_img)
plt.imshow(img)
plt.show()
