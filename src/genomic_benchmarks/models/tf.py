import tensorflow as tf
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
onehot_layer = tf.keras.layers.Lambda(lambda x: tf.one_hot(tf.cast(x, "int64"), 4))

basic_cnn_model_v0 = tf.keras.Sequential(
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
        Dense(1),
    ]
)
