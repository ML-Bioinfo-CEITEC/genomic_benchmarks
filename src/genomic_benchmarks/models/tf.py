import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    MaxPooling1D,
)
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

character_split_fn = lambda x: tf.strings.unicode_split(x, "UTF-8")
vectorize_layer = TextVectorization(output_mode="int", split=character_split_fn)

# one-hot encoding

def get_basic_cnn_model_v0(num_classes, vocab_size):

    if num_classes == 2:
        num_classes = 1

    onehot_layer = tf.keras.layers.Lambda(lambda x: tf.one_hot(tf.cast(x, "int64"), vocab_size))

    model = tf.keras.Sequential(
        [
            onehot_layer,
            Conv1D(32, kernel_size=8, data_format="channels_last", activation="relu"),
            BatchNormalization(),
            MaxPooling1D(),
            Conv1D(16, kernel_size=8, data_format="channels_last", activation="relu"),
            BatchNormalization(),
            MaxPooling1D(),
            Conv1D(4, kernel_size=8, data_format="channels_last", activation="relu"),
            BatchNormalization(),
            MaxPooling1D(),
            Dropout(0.3),
            GlobalAveragePooling1D(),
            Dense(num_classes, activation="softmax"),
        ]
    )

    if num_classes == 1:
        loss = tf.keras.losses.BinaryCrossentropy()
        acc = tf.metrics.BinaryAccuracy()
        f1 = tfa.metrics.F1Score(num_classes=1, threshold=0.5, average="micro")
    else:
        loss = 'categorical_crossentropy'
        acc = tf.metrics.CategoricalAccuracy()
        f1 = tfa.metrics.F1Score(num_classes=num_classes, average="micro")

    model.compile(
        loss=loss,
        optimizer='adam',
        metrics=[acc, f1]
    )

    return model

