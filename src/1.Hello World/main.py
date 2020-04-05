import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# 一个最简单的 TensorFlow 例子
def main():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # 把样本从整数转化为浮点数
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # 为模型定义建神经网络、顺序模型
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # 编译模型，指定训练的优化器、损失函数、训练评估指标
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # 训练模型
    model.fit(x_train, y_train, epochs=5)

    # 评估模型
    model.evaluate(x_test, y_test, verbose=2)


if __name__ == '__main__':
    main()
