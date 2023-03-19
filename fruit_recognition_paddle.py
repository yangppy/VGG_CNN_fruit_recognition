#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：简历项目 
@File    ：fruit_recognition.py
@IDE     ：PyCharm 
@Author  ：xiaoyang
@Date    ：2023/3/18 11:01 
'''
import paddle
import paddle.fluid as fluid
import numpy
import sys
import os
from multiprocessing import cpu_count
import time
import matplotlib.pyplot as plt


def train_mapper(sample):
    """
    根据传入的样本数据（一行文本）读取图片数据并返回
    :param sample: 元组，格式为（图片路径，类别）
    :return: 返回图像数据，类别
    """
    img, label = sample  # img为路径，label为类别
    if not os.path.exists(img):
        print(img, "图片不存在")

    # 读取图片内容
    img = paddle.dataset.image.load_image(img)
    # 对图片数据进行简单变换，设置为固定大小
    img = paddle.dataset.image.simple_transform(im=img,  # 原始图像
                                                resize_size=128,  # 图像要设置的大小
                                                crop_size=128,  # 剪裁图像大小
                                                is_color=True,  # 彩色图像
                                                is_train=True)  # 随机剪裁
    # 归一化处理，将每个像素值转换到0~1
    img = img.astype("float32") / 255.0
    return img, label  # 返回图像， 类别


# 从训练集中读取数据
def get_train_data(train_list, buffered_size=1024):
    def reader():
        with open(train_list, "r") as f:
            lines = [line.strip() for line in f]  # 读取所有的行去掉空格
            for line in lines:
                # 去掉一行数据的换行符，并按tab键拆分，存入两个变量
                img_path, lab = line.replace("\n", "").split("\t")
                yield img_path, int(lab)  # 返回图片路径，类别(整数)

    return paddle.reader.xmap_readers(train_mapper,  # 将reader读取的数进一步处理
                                      reader,  # reader读取到的数据传递给train_mapper
                                      cpu_count(),  # 线程数量
                                      buffered_size)  # 缓冲区大小


# 搭建CNN函数
# 结构：输入层--> 卷积/激活/池化/dropout-->卷积/激活/池化/dropout-->卷积/激活/池化/dropout-->fc-->dropout-->fc(softmax)
def convolution_neural_network(image, type_size):
    """
    创建CNN
    :param image: 图像数据
    :param type_size: 输出类别数量
    :return: 分类概率
    """
    # 第一组卷积/激活/池化/dropout
    conv_pool_1 = fluid.nets.simple_img_conv_pool(input=image,  # 原始图像数据
                                                  filter_size=3,  # 卷积核大小
                                                  num_filters=32,  # 卷积核数量
                                                  pool_size=2,  # 2*2的区域池化
                                                  pool_stride=2,  # 池化步长
                                                  act='relu')  # 激活函数
    drop = fluid.layers.dropout(x=conv_pool_1, dropout_prob=0.5)

    # 第二组
    conv_pool_2 = fluid.nets.simple_img_conv_pool(input=drop,  # 原始图像数据
                                                  filter_size=3,  # 卷积核大小
                                                  num_filters=64,  # 卷积核数量
                                                  pool_size=2,  # 2*2的区域池化
                                                  pool_stride=2,  # 池化步长
                                                  act='relu')  # 激活函数
    drop = fluid.layers.dropout(x=conv_pool_2, dropout_prob=0.5)

    # 第三组
    conv_pool_3 = fluid.nets.simple_img_conv_pool(input=drop,  # 原始图像数据
                                                  filter_size=3,  # 卷积核大小
                                                  num_filters=64,  # 卷积核数量
                                                  pool_size=2,  # 2*2的区域池化
                                                  pool_stride=2,  # 池化步长
                                                  act='relu')  # 激活函数
    drop = fluid.layers.dropout(x=conv_pool_3, dropout_prob=0.5)

    # 全连接层
    fc = fluid.layers.fc(input=drop, size=512, act="relu")
    # dropout
    drop = fluid.layers.dropout(x=fc, dropout_prob=0.5)
    # 输出层（fc）
    predict = fluid.layers.fc(input=drop,  # 输入
                              size=type_size,  # 输出值的个数（5个类别）
                              act='softmax')  # 采用softmax作为激活函数
    return predict


# 定义reader
BATCH_SIZE = 32  # 批次大小
data_root_path = "fruits/"  # 数据所在的根目录
test_file_path = data_root_path + "test.txt"  # 测试文件目录
train_file_path = data_root_path + "train.txt"  # 训练文件目录
trainer_reader = get_train_data(train_list=train_file_path)  # 原始reader
random_train_reader = paddle.reader.shuffle(reader=trainer_reader,
                                            buf_size=1300)  # 包装成随机读取器
batch_train_reader = paddle.batch(random_train_reader,
                                  batch_size=BATCH_SIZE)  # 批量读取器

# 变量
image = fluid.layers.data(name="image", shape=[3, 128, 128], dtype="float32")
label = fluid.layers.data(name="label", shape=[1], dtype="int64")

# 调用函数，创建CNN
predict = convolution_neural_network(image=image, type_size=5)
# 损失函数：交叉熵
cost = fluid.layers.cross_entropy(input=predict,  # 预测结果
                                  label=label)  # 真实结果

avg_cost = fluid.layers.mean(cost)

# 计算准确率
accuracy = fluid.layers.accuracy(input=predict,  # 预测结果
                                 label=label)  # 真实结果

# 优化器
optimizer = fluid.optimizer.Adam(learning_rate=0.001)
optimizer.minimize(avg_cost)  # 将损失函数值优化到最小

# 执行器
place = fluid.CPUPlace()  # CPU训练
# place = fluid.CUDAPlace(0)  # GPU训练
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
# feeder
feeder = fluid.DataFeeder(feed_list=[image, label],  # 指定要喂入数据
                          place=place)
model_save_dir = "model/"  # 模型保存路径
costs = []  # 记录损失值
accs = []  # 记录准确率
times = 0
batches = []  # 迭代次数

# 开始训练
for pass_id in range(40):
    train_cost = 0  # 临时变量，记录每次训练的损失值
    for batch_id, data in enumerate(batch_train_reader()):  # 循环读取样本，执行训练
        times += 1
        train_cost, train_acc = exe.run(program=fluid.default_startup_program(),
                                        feed=feeder.feed(data),  # 喂入参数
                                        fetch_list=[avg_cost, accuracy])  # 获取损失值，准确率
        if batch_id % 20 == 0:
            print("pass_id:%d, step:%d, cost:%f, acc:%f" % (pass_id, batch_id, train_cost[0], train_acc[0]))
            accs.append(train_acc[0])  # 记录准确率
            costs.append(train_cost[0])  # 记录损失值
            batches.append(times)  # 记录迭代次数

# 训练结束后，保存模型
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)
fluid.io.save_inference_model(dirname=model_save_dir,
                              feeded_var_names=["image"],
                              target_vars=[predict],
                              executor=exe)
print("模型训练保存完毕！")

# 训练过程可视化
plt.title("training", fontsize=24)
plt.xlabel("iter", fonsize=20)
plt.ylabel("cost/acc", fontsize=20)
plt.plot(batches, costs, color='red', label="Training Cost")
plt.plot(batches, accs, color='green', label="Training ACC")
plt.legend()
plt.grid()
plt.savefig("train.png")
plt.show()
