import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import os
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():
    # 样本数据，返回 train、test、info
    (train_data, test_data), info = tfds.load(
        # Use the version pre-encoded with an ~8k vocabulary.
        'imdb_reviews/subwords8k',
        # Return the train/test datasets as a tuple.
        split=(tfds.Split.TRAIN, tfds.Split.TEST),
        # Return (example, label) pairs from the dataset (instead of a dictionary).
        as_supervised=True,
        # Also return the `info` structure.
        with_info=True)

    # info 中包含一个 text encoder tfds.features.text.SubwordTextEncoder 这个编码器包含 8185 个词汇
    encoder = info.features['text'].encoder
    print('Vocabulary size: {}'.format(encoder.vocab_size))

    sample_string = 'Hello TensorFlow.'

    encoded_string = encoder.encode(sample_string)
    print('Encoded string is {}'.format(encoded_string))

    original_string = encoder.decode(encoded_string)
    print('The original string: "{}"'.format(original_string))

    assert original_string == sample_string

    # 通过分割词汇进行编码，如果单词不在字典中，则使用其分割为子单词
    for ts in encoded_string:
        print('{} ----> {}'.format(ts, encoder.decode([ts])))

    # 取第一个，查看该文本编码后的 text 和其对应的标签
    for train_example, train_label in train_data.take(1):
        print('Encoded text:', train_example[:10].numpy())
        print('Label:', train_label.numpy())
        print(encoder.decode(train_example))

    BUFFER_SIZE = 1000

    train_batches = (
        train_data.shuffle(BUFFER_SIZE).padded_batch(32, padded_shapes=([None], [])))

    test_batches = (
        test_data.padded_batch(32, padded_shapes=([None], [])))

    for example_batch, label_batch in train_batches.take(2):
        print("Batch shape:", example_batch.shape)
        print("label shape:", label_batch.shape)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(encoder.vocab_size, 16),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(1)])

    model.summary()

    model.compile(optimizer='adam',
                  loss=tf.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(train_batches,
                        epochs=10,
                        validation_data=test_batches,
                        validation_steps=30)

    loss, accuracy = model.evaluate(test_batches)

    print("Loss: ", loss)
    print("Accuracy: ", accuracy)
    #
    # history_dict = history.history
    # history_dict.keys()
    #
    # acc = history_dict['accuracy']
    # val_acc = history_dict['val_accuracy']
    # loss = history_dict['loss']
    # val_loss = history_dict['val_loss']
    #
    # epochs = range(1, len(acc) + 1)
    #
    # # "bo" is for "blue dot"
    # plt.plot(epochs, loss, 'bo', label='Training loss')
    # # b is for "solid blue line"
    # plt.plot(epochs, val_loss, 'b', label='Validation loss')
    # plt.title('Training and validation loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    #
    # plt.show()
    #
    # plt.clf()  # clear figure
    #
    # plt.plot(epochs, acc, 'bo', label='Training acc')
    # plt.plot(epochs, val_acc, 'b', label='Validation acc')
    # plt.title('Training and validation accuracy')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.legend(loc='lower right')
    #
    # plt.show()


if __name__ == '__main__':
    main()
