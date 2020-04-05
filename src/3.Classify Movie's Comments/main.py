import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")


def main():
    # 样本数据：5w 条评论，其中 1.5w 训练数据、1w 验证数据、2.5w 测试样本
    train_data, validation_data, test_data = tfds.load(
        name="imdb_reviews",
        split=('train[:60%]', 'train[60%:]', 'test'),
        as_supervised=True)

    # 了解数据格式
    train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
    print(train_examples_batch)
    print(train_labels_batch)

    # 搭建模型，需要考虑几个问题
    # - 如何表示文字？：通过现有的预训练模型将文字转化为 embeddings vectors，这里用到了迁移学习
    # - 需要建立多少层？
    # - 每一层有多少个神经元？

    embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
    hub_layer = hub.KerasLayer(embedding, input_shape=[],
                               dtype=tf.string, trainable=True)
    print(hub_layer(train_examples_batch[:3]))

    # 第一层是预训练模型层，对原始文本数据做了处理，映射为 embeddings vectors
    model = tf.keras.Sequential()
    model.add(hub_layer)
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(1))

    model.summary()

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # 训练
    history = model.fit(train_data.shuffle(10000).batch(512),
                        epochs=20,
                        validation_data=validation_data.batch(512),
                        verbose=1)

    # 验证
    results = model.evaluate(test_data.batch(512), verbose=2)
    for name, value in zip(model.metrics_names, results):
        print("%s: %.3f" % (name, value))

    probability_model = tf.keras.Sequential([model,
                                             tf.keras.layers.Softmax()])
    data = test_data.batch(512)
    predictions = probability_model.predict(data)
    predict_examples_batch, predict_labels_batch = next(iter(data))
    print(predict_examples_batch[0])
    print("\n")
    print(predict_labels_batch[0])
    print("\n")
    print(predictions[0])


if __name__ == '__main__':
    main()
