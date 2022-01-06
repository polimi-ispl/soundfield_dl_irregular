import tensorflow as tf

def filter_compensation_model_wideband_skipped(filter_shape, nfft):

    input_layer = tf.keras.layers.Input(shape=(filter_shape, nfft, 1))
    x = tf.keras.layers.ZeroPadding2D((1, 0))(input_layer)
    x = tf.keras.layers.Conv2D(512, 3, 2, padding='valid')(x)
    x = tf.keras.layers.PReLU()(x)
    x1 = x  # Skipped connection 1
    x = tf.keras.layers.ZeroPadding2D((1, 0))(x)
    x = tf.keras.layers.Conv2D(256, 3, 2, padding='valid')(x)
    x = tf.keras.layers.PReLU()(x)
    x2 = x  # Skipped connection 1
    x = tf.keras.layers.ZeroPadding2D((1, 0))(x)
    x = tf.keras.layers.Conv2D(128, 3, 2, padding='valid')(x)
    x = tf.keras.layers.PReLU()(x)

    x = tf.keras.layers.Conv2DTranspose(256, 3, 2, padding='valid')(x)
    x = tf.keras.layers.Cropping2D(((1, 0), (0, 0)))(x)
    x = tf.keras.layers.PReLU()(x)

    x = tf.keras.layers.add([x, x2])

    x = tf.keras.layers.Conv2DTranspose(512, 3, 2, padding='valid')(x)
    x = tf.keras.layers.Cropping2D(((1, 0), (0, 0)))(x)
    x = tf.keras.layers.PReLU()(x)

    x1 = tf.keras.layers.Cropping2D(((0, 0), (0, 1)))(x1)
    x = tf.keras.layers.add([x, x1])

    x = tf.keras.layers.ZeroPadding2D(((0, 0), (0, 1)))(x)
    x = tf.keras.layers.Conv2DTranspose(512, 3, 2, padding='valid')(x)
    x = tf.keras.layers.Cropping2D(((1, 0), (0, 0)))(x)

    # Output
    x = tf.keras.layers.Conv2DTranspose(1, 3, 1, padding='same')(x)
    out = tf.keras.layers.Activation('tanh')(x)
    return tf.keras.models.Model(inputs=input_layer, outputs=out)

