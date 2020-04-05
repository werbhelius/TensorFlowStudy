# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# 训练一个分类机器学习模型，对衣服的类型作出预测
def main():
    fashion_mnist = keras.datasets.fashion_mnist

    # 样本数据，包含特征和标签，分别表示输入和输出
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # 样本的输出通过 0...9 标示，这里对数字对应的类型描述
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # 预处理数据
    # 第0张图片的原始数据，未处理的图片的颜色值范围为[0,255]
    # plt.figure()
    # plt.imshow(train_images[0])
    # plt.colorbar()
    # plt.grid(False)
    # plt.show()

    # 将图片归一化，即将颜色范围缩小到[0,1]
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # 验证数据格式是否正确
    # plt.figure(figsize=(10, 10))
    # for i in range(25):
    #     plt.subplot(5, 5, i + 1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid(False)
    #     plt.imshow(train_images[i], cmap=plt.cm.binary)
    #     plt.xlabel(class_names[train_labels[i]])
    # plt.show()

    # 搭建神经网络
    # 第一层将图像格式从二维数组转化成一维数组
    # 第二层全连接层有128个神经元
    # 第二层也全连接层有10个神经元标示输出层
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10)
    ])

    # 编译模型，指定训练的优化器、损失函数、训练评估指标
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # 训练模型：训练是一个将权重和偏差值调整为最佳值的过程，权重和偏差值为所有神经网络层中的内部变量
    model.fit(train_images, train_labels, epochs=10)

    # 评估模型：通过准确性和丢失率来评估
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print('\nTest accuracy:', test_acc)

    # 至此模型以及训练完成，需要对真实数据作出预测
    # 在之前模型的基础上增加新的输出层 Softmax，Softmax 针对于分类模型生成的原始（非标准化）预测向量将其转化为标准化函数
    probability_model = tf.keras.Sequential([model,
                                             tf.keras.layers.Softmax()])

    predictions = probability_model.predict(test_images)
    # 输出第0个的预测分类
    print(predictions[0])
    # 输出预测分类最高的类型
    print(np.argmax(predictions[0]))
    # 与标签对比检查预测是否正确
    print(test_labels[0])

    # 可视化
    def plot_image(i, predictions_array, true_label, img):
        predictions_array, true_label, img = predictions_array, true_label[i], img[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])

        plt.imshow(img, cmap=plt.cm.binary)

        predicted_label = np.argmax(predictions_array)
        if predicted_label == true_label:
            color = 'blue'
        else:
            color = 'red'

        plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                             100 * np.max(predictions_array),
                                             class_names[true_label]),
                   color=color)

    def plot_value_array(i, predictions_array, true_label):
        predictions_array, true_label = predictions_array, true_label[i]
        plt.grid(False)
        plt.xticks(range(10))
        plt.yticks([])
        thisplot = plt.bar(range(10), predictions_array, color="#777777")
        plt.ylim([0, 1])
        predicted_label = np.argmax(predictions_array)

        thisplot[predicted_label].set_color('red')
        thisplot[true_label].set_color('blue')

    i = 3
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(1, 2, 2)
    plot_value_array(i, predictions[i], test_labels)
    plt.show()

    i = 12
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(1,2,2)
    plot_value_array(i, predictions[i],  test_labels)
    plt.show()

    num_rows = 5
    num_cols = 3
    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(i, predictions[i], test_labels, test_images)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, predictions[i], test_labels)
    plt.tight_layout()
    plt.show()

    # 使用训练模型验证单个图片
    img = test_images[1]
    print(img.shape)

    img = (np.expand_dims(img, 0))
    print(img.shape)

    predictions_single = probability_model.predict(img)
    print(predictions_single)

    plot_value_array(1, predictions_single[0], test_labels)
    _ = plt.xticks(range(10), class_names, rotation=45)

    print(np.argmax(predictions_single[0]))


if __name__ == '__main__':
    main()
