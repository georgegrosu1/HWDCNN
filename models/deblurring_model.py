from typing import Optional, Union, Callable, List
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as B
from tf_layers_extended import WienerDeconv2D, TVDeconv2D


def peak_snr(y_true, y_pred):
    mse = tf.experimental.numpy.mean((y_true - y_pred) ** 2.0)
    if mse == 0.0:  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100.0

    max_pixel = 255.0
    psnr = 20.0 * tf.experimental.numpy.log10(max_pixel / tf.experimental.numpy.sqrt(mse))

    return psnr


def mse_ssim(y_true, y_pred):
    mse = tf.keras.losses.MeanSquaredError()

    return mse(y_true, y_pred) * (B.exp(B.exp(ssim(y_true, y_pred))) - tf.experimental.numpy.e) * (40 / peak_snr(y_true, y_pred))


def ssim(y_true, y_pred):
    loss_rec = tf.image.ssim(y_true, y_pred, max_val=255.0)

    return 1 - loss_rec


def ms_ssim(y_true, y_pred):
    loss_rec = tf.image.ssim_multiscale(y_true, y_pred, max_val=255.0)[..., None]

    return 1 - loss_rec


def step_decay(epoch):
    init_lr = 0.001
    drop = 0.1
    epo_drop = 20
    lr = init_lr * tf.math.pow(drop, tf.math.floor((1 + epoch)/epo_drop))
    return lr


def build_deblurring_model(input_shape,
                           loss=ssim,
                           optimizer=None,
                           pretrained_path=None):

    if optimizer is not None:
        optimizer = optimizer
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001,
                                             beta_1=0.9, beta_2=0.999,
                                             epsilon=1e-8,
                                             clipvalue=2)

    inputs = tf.keras.layers.Input(shape=input_shape)
    if pretrained_path is not None:
        base_model = tf.keras.models.load_model(pretrained_path, custom_objects={
            'WienerDeconvolution2D': WienerDeconv2D,
            'peak_snr': peak_snr,
            'ms_ssim': ms_ssim})
        base_model.trainable = False

        x = base_model(inputs, training=False)
    else:
        x = inputs

    # First branch
    deconv_1 = WienerDeconv2D(filters=2, kernel_size=input_shape, padding=((0, 0), (0, 0)))(x)
    # batch_11 = tf.keras.layers.BatchNormalization()(deconv_1)
    # rel11 = tf.keras.layers.ReLU()(batch_11)
    deconv_conv_1 = tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=3, strides=(1, 1))(deconv_1)
    norm_1_1 = tf.keras.layers.Normalization()(deconv_conv_1)
    rel_1_1 = tf.keras.layers.ReLU()(norm_1_1)
    deconv_conv_2 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=6, strides=(1, 1))(rel_1_1)
    norm_1_2 = tf.keras.layers.Normalization()(deconv_conv_2)
    rel_1_2 = tf.keras.layers.ReLU()(norm_1_2)
    deconv_conv_3 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=12, strides=(1, 1))(rel_1_2)
    norm_1_3 = tf.keras.layers.Normalization()(deconv_conv_3)
    rel_1_3 = tf.keras.layers.ReLU()(norm_1_3)

    # Second branch
    tvdeconv = TVDeconv2D(150, 'unit')(x)
    conv2d_1 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=8, strides=(1, 1))(tvdeconv)
    norm_2_1 = tf.keras.layers.Normalization()(conv2d_1)
    rel_2_1 = tf.keras.layers.ReLU()(norm_2_1)
    conv2d_2 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=12, strides=(1, 1))(rel_2_1)
    norm_2_2 = tf.keras.layers.Normalization()(conv2d_2)
    rel_2_2 = tf.keras.layers.ReLU()(norm_2_2)

    sums1 = tf.keras.layers.Add()([rel_1_2, rel_2_1])
    conv2d_sum1 = tf.keras.layers.Conv2D(filters=3, kernel_size=8, strides=(1, 1))(sums1)
    norm_sum1 = tf.keras.layers.Normalization()(conv2d_sum1)
    rel_sum1 = tf.keras.layers.ReLU()(norm_sum1)
    sums2 = tf.keras.layers.Add()([rel_1_3, rel_2_2])
    conv2d_sum2 = tf.keras.layers.Conv2D(filters=3, kernel_size=19, strides=(1, 1))(sums2)
    norm_sum2 = tf.keras.layers.Normalization()(conv2d_sum2)
    rel_sum2 = tf.keras.layers.ReLU()(norm_sum2)

    sum_fin = tf.keras.layers.Add()([rel_sum1, rel_sum2])

    if pretrained_path is not None:

        final_model = tf.keras.Model(inputs=inputs, outputs=sum_fin)
    else:
        final_model = tf.keras.Model(inputs=x, outputs=sum_fin)

    final_model.compile(loss=loss, metrics=['mse', peak_snr], optimizer=optimizer)
    final_model.summary()

    return final_model


def build_autoencoder_model(input_shape,
                            loss=ssim,
                            optimizer=None,
                            pretrained_path=None):
    if optimizer is not None:
        optimizer = optimizer
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001,
                                             beta_1=0.9, beta_2=0.999,
                                             epsilon=1e-8,
                                             clipvalue=2)

    inputs = tf.keras.layers.Input(shape=input_shape)
    if pretrained_path is not None:
        base_model = tf.keras.models.load_model(pretrained_path, custom_objects={
            'WienerDeconvolution2D': WienerDeconv2D,
            'peak_snr': peak_snr,
            'ms_ssim': ms_ssim,
            'ssim': ssim})
        base_model.trainable = False

        x_in = base_model(inputs, training=False)
    else:
        x_in = inputs

    # 1 downscale branch
    tvdeconv_1 = TVDeconv2D(150, 'unit')(x_in)
    x_in_1_down = tf.keras.layers.Conv2D(filters=32, kernel_size=2, strides=(1, 1))(tvdeconv_1)
    x_in_1_down = tf.keras.layers.Normalization()(x_in_1_down)
    x_in_1_down = tf.keras.layers.ReLU()(x_in_1_down)

    # 2 downscale branch
    x_in_2_down = WienerDeconv2D(filters=2, kernel_size=input_shape, padding=((0, 0), (0, 0)),
                                 kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=100., stddev=50.))(x_in)
    x_in_2_down = tf.keras.layers.Normalization()(x_in_2_down)
    x_in_2_down = tf.keras.layers.ReLU()(x_in_2_down)

    x_in_2_down = tf.keras.layers.Conv2D(filters=32, kernel_size=2, strides=(1, 1))(x_in_2_down)
    x_in_2_down = tf.keras.layers.Normalization()(x_in_2_down)
    x_in_2_down = tf.keras.layers.ReLU()(x_in_2_down)

    # 3 downscale branch
    tvdeconv_3 = TVDeconv2D(150, 'unit')(x_in)
    x_in_3_down = tf.keras.layers.Conv2D(filters=32, kernel_size=2, strides=(1, 1))(tvdeconv_3)
    x_in_3_down = tf.keras.layers.Normalization()(x_in_3_down)
    x_in_3_down = tf.keras.layers.ReLU()(x_in_3_down)

    # 4 downscale branch
    x_in_4_down = WienerDeconv2D(filters=2, kernel_size=input_shape, padding=((0, 0), (0, 0)),
                                 kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=100., stddev=50.))(x_in)
    x_in_4_down = tf.keras.layers.Normalization()(x_in_4_down)
    x_in_4_down = tf.keras.layers.ReLU()(x_in_4_down)

    x_in_4_down = tf.keras.layers.Conv2D(filters=32, kernel_size=2, strides=(1, 1))(x_in_4_down)
    x_in_4_down = tf.keras.layers.Normalization()(x_in_4_down)
    x_in_4_down = tf.keras.layers.ReLU()(x_in_4_down)

    # 1-2 downscale combine
    x_12_comb = tf.keras.layers.concatenate([x_in_1_down, x_in_2_down])
    x_12_comb_down = tf.keras.layers.Conv2D(filters=64, kernel_size=6, strides=(1, 1))(x_12_comb)
    x_12_comb_down = tf.keras.layers.Normalization()(x_12_comb_down)
    x_12_comb_down = tf.keras.layers.ReLU()(x_12_comb_down)

    # 3-4 downscale combine
    x_34_comb = tf.keras.layers.concatenate([x_in_3_down, x_in_4_down])
    x_34_comb_down = tf.keras.layers.Conv2D(filters=64, kernel_size=6, strides=(1, 1))(x_34_comb)
    x_34_comb_down = tf.keras.layers.Normalization()(x_34_comb_down)
    x_34_comb_down = tf.keras.layers.ReLU()(x_34_comb_down)

    # 1-2-3-4 downscale combine
    x_1234_comb = tf.keras.layers.concatenate([x_12_comb_down, x_34_comb_down])
    x_1234_comb_down = tf.keras.layers.Conv2D(filters=128, kernel_size=8, strides=(1, 1))(x_1234_comb)
    x_1234_comb_down = tf.keras.layers.Normalization()(x_1234_comb_down)
    x_1234_comb_down = tf.keras.layers.ReLU()(x_1234_comb_down)


    # 1-2-3-4 upscale combine
    x_in_1234_up = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=8, strides=(1, 1))(x_1234_comb_down)
    x_in_1234_up = tf.keras.layers.Normalization()(x_in_1234_up)
    x_in_1234_up = tf.keras.layers.ReLU()(x_in_1234_up)

    # 1-2 upscale combine
    x_in_12_up = tf.keras.layers.concatenate([x_in_1234_up, x_12_comb_down])
    x_in_12_up = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=6, strides=(1, 1))(x_in_12_up)
    x_in_12_up = tf.keras.layers.Normalization()(x_in_12_up)
    x_in_12_up = tf.keras.layers.ReLU()(x_in_12_up)

    # 3-4 upscale combine
    x_in_34_up = tf.keras.layers.concatenate([x_in_1234_up, x_34_comb_down])
    x_in_34_up = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=6, strides=(1, 1))(x_in_34_up)
    x_in_34_up = tf.keras.layers.Normalization()(x_in_34_up)
    x_in_34_up = tf.keras.layers.ReLU()(x_in_34_up)

    # 1 upscale combine
    x_in_1_up = tf.keras.layers.concatenate([x_in_1_down, x_in_12_up, x_in_2_down, x_in_3_down, x_in_4_down])
    x_in_1_up = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=2, strides=(1, 1))(x_in_1_up)
    x_in_1_up = tf.keras.layers.Normalization()(x_in_1_up)
    x_in_1_up = tf.keras.layers.ReLU()(x_in_1_up)

    # 2 upscale combine
    x_in_2_up = tf.keras.layers.concatenate([x_in_1_down, x_in_12_up, x_in_2_down, x_in_3_down, x_in_4_down])
    x_in_2_up = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=2, strides=(1, 1))(x_in_2_up)
    x_in_2_up = tf.keras.layers.Normalization()(x_in_2_up)
    x_in_2_up = tf.keras.layers.ReLU()(x_in_2_up)

    # 3 upscale combine
    x_in_3_up = tf.keras.layers.concatenate([x_in_1_down, x_in_34_up, x_in_2_down, x_in_3_down, x_in_4_down])
    x_in_3_up = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=2, strides=(1, 1))(x_in_3_up)
    x_in_3_up = tf.keras.layers.Normalization()(x_in_3_up)
    x_in_3_up = tf.keras.layers.ReLU()(x_in_3_up)

    # 4 upscale combine
    x_in_4_up = tf.keras.layers.concatenate([x_in_1_down, x_in_34_up, x_in_2_down, x_in_3_down, x_in_4_down])
    x_in_4_up = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=2, strides=(1, 1))(x_in_4_up)
    x_in_4_up = tf.keras.layers.Normalization()(x_in_4_up)
    x_in_4_up = tf.keras.layers.ReLU()(x_in_4_up)

    all_ups_concat = tf.keras.layers.concatenate([x_in_1_up, x_in_2_up, x_in_3_up, x_in_4_up])
    all_ups_concat = tf.keras.layers.Conv2D(filters=3, kernel_size=1, strides=(1, 1))(all_ups_concat)
    all_ups_concat = tf.keras.layers.Normalization()(all_ups_concat)
    all_ups_concat = tf.keras.layers.ReLU()(all_ups_concat)

    if pretrained_path is not None:

        final_model = tf.keras.Model(inputs=inputs, outputs=all_ups_concat)
    else:
        final_model = tf.keras.Model(inputs=x_in, outputs=all_ups_concat)

    final_model.compile(loss=loss, metrics=[ms_ssim, ssim, 'mse', peak_snr], optimizer=optimizer)
    final_model.summary()

    return final_model


def build_deblurring_second_model(input_shape,
                                  batch_size,
                                  loss=ssim,
                                  optimizer=None,
                                  pretrained_path=None):

    if optimizer is not None:
        optimizer = optimizer
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001,
                                             beta_1=0.9, beta_2=0.999,
                                             epsilon=1e-8,
                                             clipvalue=2)

    inputs = tf.keras.layers.Input(shape=input_shape, batch_size=batch_size)
    if pretrained_path is not None:
        base_model = tf.keras.models.load_model(pretrained_path, custom_objects={
            'WienerDeconvolution2D': WienerDeconv2D,
            'peak_snr': peak_snr,
            'ms_ssim': ms_ssim,
            'ssim': ssim})
        base_model.trainable = False

        x_in = base_model(inputs, training=False)
    else:
        x_in = inputs

    # First downscale branch
    tvdeconv_1 = TVDeconv2D(150, 'channel', activation='relu')(x_in)
    x_in_1_down = tf.keras.layers.Conv2D(filters=32, kernel_size=2, strides=(1, 1))(tvdeconv_1)
    x_in_1_down = tf.keras.layers.Normalization()(x_in_1_down)
    x_in_1_down = tf.keras.layers.ReLU()(x_in_1_down)
    x_in_1_down = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x_in_1_down)
    x_in_1_down_res_1 = x_in_1_down
    x_in_1_down = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=(1, 1))(x_in_1_down)
    x_in_1_down = tf.keras.layers.Normalization()(x_in_1_down)
    x_in_1_down = tf.keras.layers.ReLU()(x_in_1_down)
    x_in_1_down = tf.keras.layers.MaxPooling2D(pool_size=(4, 4), strides=(1, 1), padding='same')(x_in_1_down)
    x_in_1_down_res_2 = x_in_1_down
    x_in_1_down = tf.keras.layers.Conv2D(filters=128, kernel_size=5, strides=(1, 1))(x_in_1_down)
    x_in_1_down = tf.keras.layers.Normalization()(x_in_1_down)
    x_in_1_down = tf.keras.layers.ReLU()(x_in_1_down)
    x_in_1_down = tf.keras.layers.MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')(x_in_1_down)

    # Second downscale branch
    x_in_2_down = WienerDeconv2D(filters=1, kernel_size=input_shape, padding=((0, 0), (0, 0)),
                                 kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=150., stddev=90.))(x_in)
    x_in_2_down = tf.keras.layers.Normalization()(x_in_2_down)
    x_in_2_down = tf.keras.layers.ReLU()(x_in_2_down)
    x_in_2_down = tf.keras.layers.Conv2D(filters=32, kernel_size=4, strides=(1, 1))(x_in_2_down)
    x_in_2_down = tf.keras.layers.Normalization()(x_in_2_down)
    x_in_2_down = tf.keras.layers.ReLU()(x_in_2_down)
    x_in_2_down_res_1 = x_in_2_down
    x_in_2_down = tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=(1, 1))(x_in_2_down)
    x_in_2_down = tf.keras.layers.Normalization()(x_in_2_down)
    x_in_2_down = tf.keras.layers.ReLU()(x_in_2_down)
    x_in_2_down_res_2 = x_in_2_down
    # x_in_2_down_dec = Deconv2D(filters=2, kernel_size=input_shape, padding=((0, 0), (0, 0)))(x_in_2_down)
    x_in_2_down = tf.keras.layers.Conv2D(filters=64, kernel_size=7, strides=(1, 1))(x_in_2_down)
    x_in_2_down = tf.keras.layers.Normalization()(x_in_2_down)
    x_in_2_down = tf.keras.layers.ReLU()(x_in_2_down)

    # Third downscale branch
    x_in_3_down = tf.keras.layers.Conv2D(filters=32, kernel_size=6, strides=(1, 1))(x_in)
    x_in_3_down = tf.keras.layers.Normalization()(x_in_3_down)
    x_in_3_down = tf.keras.layers.ReLU()(x_in_3_down)
    # x_in_3_down = tf.keras.layers.MaxPooling2D(pool_size=(6, 6), strides=(1, 1), padding='same')(x_in_3_down)
    x_in_3_down_res_1 = x_in_3_down
    x_in_3_down = tf.keras.layers.Conv2D(filters=64, kernel_size=8, strides=(1, 1))(x_in_3_down)
    x_in_3_down = tf.keras.layers.Normalization()(x_in_3_down)
    x_in_3_down = tf.keras.layers.ReLU()(x_in_3_down)
    # x_in_3_down = tf.keras.layers.MaxPooling2D(pool_size=(8, 8), strides=(1, 1), padding='same')(x_in_3_down)
    x_in_3_down_res_2 = x_in_3_down
    x_in_3_down = tf.keras.layers.Conv2D(filters=128, kernel_size=9, strides=(1, 1))(x_in_3_down)
    x_in_3_down = tf.keras.layers.Normalization()(x_in_3_down)
    x_in_3_down = tf.keras.layers.ReLU()(x_in_3_down)
    # x_in_3_down = tf.keras.layers.MaxPooling2D(pool_size=(9, 9), strides=(1, 1), padding='same')(x_in_3_down)
    # x_in_3d_down = Deconv2D(filters=1, kernel_size=input_shape, padding=((0, 0), (0, 0)))(x_in_3d_down)

    # Fourth downscale branch
    # tvdeconv_4 = TVDeconv2D(150, 'unit')(x_in)
    x_in_4_down = tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=(1, 1))(x_in)
    x_in_4_down = tf.keras.layers.Normalization()(x_in_4_down)
    x_in_4_down = tf.keras.layers.ReLU()(x_in_4_down)
    # x_in_4_down = tf.keras.layers.MaxPooling2D(pool_size=(8, 8), strides=(1, 1), padding='same')(x_in_4_down)
    x_in_4_down_res_1 = x_in_4_down
    x_in_4_down = tf.keras.layers.Conv2D(filters=64, kernel_size=9, strides=(1, 1))(x_in_4_down)
    x_in_4_down = tf.keras.layers.Normalization()(x_in_4_down)
    x_in_4_down = tf.keras.layers.ReLU()(x_in_4_down)
    # x_in_4_down = tf.keras.layers.MaxPooling2D(pool_size=(9, 9), strides=(1, 1), padding='same')(x_in_4_down)
    x_in_4_down_res_2 = x_in_4_down
    x_in_4_down = tf.keras.layers.Conv2D(filters=64, kernel_size=11, strides=(1, 1))(x_in_4_down)
    x_in_4_down = tf.keras.layers.Normalization()(x_in_4_down)
    x_in_4_down = tf.keras.layers.ReLU()(x_in_4_down)
    # x_in_4_down = tf.keras.layers.MaxPooling2D(pool_size=(11, 11), strides=(1, 1), padding='same')(x_in_4_down)

    # Fifth downscale branch
    x_in_5_down = tf.keras.layers.Conv2D(filters=32, kernel_size=10, strides=(1, 1))(x_in)
    x_in_5_down = tf.keras.layers.Normalization()(x_in_5_down)
    x_in_5_down = tf.keras.layers.ReLU()(x_in_5_down)
    # x_in_5_down = tf.keras.layers.MaxPooling2D(pool_size=(10, 10), strides=(2, 2), padding='same')(x_in_5_down)
    x_in_5_down_res_1 = x_in_5_down
    x_in_5_down = tf.keras.layers.Conv2D(filters=64, kernel_size=12, strides=(1, 1))(x_in_5_down)
    x_in_5_down = tf.keras.layers.Normalization()(x_in_5_down)
    x_in_5_down = tf.keras.layers.ReLU()(x_in_5_down)
    # x_in_5_down = tf.keras.layers.MaxPooling2D(pool_size=(12, 12), strides=(2, 2), padding='same')(x_in_5_down)
    x_in_5_down_res_2 = x_in_5_down
    x_in_5_down = tf.keras.layers.Conv2D(filters=128, kernel_size=13, strides=(1, 1))(x_in_5_down)
    x_in_5_down = tf.keras.layers.Normalization()(x_in_5_down)
    x_in_5_down = tf.keras.layers.ReLU()(x_in_5_down)
    # x_in_5_down = tf.keras.layers.MaxPooling2D(pool_size=(13, 13), strides=(1, 1), padding='same')(x_in_5_down)

    # Sixth downscale branch
    x_in_6_down = tf.keras.layers.Conv2D(filters=32, kernel_size=12, strides=(1, 1))(x_in)
    x_in_6_down = tf.keras.layers.Normalization()(x_in_6_down)
    x_in_6_down = tf.keras.layers.ReLU()(x_in_6_down)
    # x_in_6_down = tf.keras.layers.MaxPooling2D(pool_size=(12, 12), strides=(1, 1), padding='same')(x_in_6_down)
    x_in_6_down_res_1 = x_in_6_down
    x_in_6_down = tf.keras.layers.Conv2D(filters=64, kernel_size=13, strides=(1, 1))(x_in_6_down)
    x_in_6_down = tf.keras.layers.Normalization()(x_in_6_down)
    x_in_6_down = tf.keras.layers.ReLU()(x_in_6_down)
    # x_in_6_down = tf.keras.layers.MaxPooling2D(pool_size=(13, 13), strides=(1, 1), padding='same')(x_in_6_down)
    x_in_6_down_res_2 = x_in_6_down
    x_in_6_down = tf.keras.layers.Conv2D(filters=64, kernel_size=15, strides=(1, 1))(x_in_6_down)
    x_in_6_down = tf.keras.layers.Normalization()(x_in_6_down)
    x_in_6_down = tf.keras.layers.ReLU()(x_in_6_down)
    # x_in_6_down = tf.keras.layers.MaxPooling2D(pool_size=(15, 15), strides=(1, 1), padding='same')(x_in_6_down)

    # Concat deconvs
    # concat_deconvs = tf.keras.layers.concatenate([x_in_1d_down, x_in_2d_down, x_in_3d_down])

    # First upscale branch
    x_in_1_up = tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=5, strides=(1, 1))(x_in_1_down)
    x_in_1_up = tf.keras.layers.Normalization()(x_in_1_up)
    x_in_1_up = tf.keras.layers.ReLU()(x_in_1_up)
    x_in_1_up = tf.keras.layers.concatenate([x_in_1_up, x_in_1_down_res_2])
    x_in_1_up = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=4, strides=(1, 1))(x_in_1_up)
    x_in_1_up = tf.keras.layers.Normalization()(x_in_1_up)
    x_in_1_up = tf.keras.layers.ReLU()(x_in_1_up)
    x_in_1_up = tf.keras.layers.concatenate([x_in_1_up, x_in_1_down_res_1])
    x_in_1_up = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=10, strides=(1, 1))(x_in_1_up)
    x_in_1_up = tf.keras.layers.Normalization()(x_in_1_up)
    x_in_1_up = tf.keras.layers.ReLU()(x_in_1_up)

    # Second upscale branch
    x_in_2_up = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=7, strides=(1, 1))(x_in_2_down)
    x_in_2_up = tf.keras.layers.Normalization()(x_in_2_up)
    x_in_2_up = tf.keras.layers.ReLU()(x_in_2_up)
    x_in_2_up = tf.keras.layers.concatenate([x_in_2_up, x_in_2_down_res_2])
    x_in_2_up = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=5, strides=(1, 1))(x_in_2_up)
    x_in_2_up = tf.keras.layers.Normalization()(x_in_2_up)
    x_in_2_up = tf.keras.layers.ReLU()(x_in_2_up)
    x_in_2_up = tf.keras.layers.concatenate([x_in_2_up, x_in_2_down_res_1])
    x_in_2_up = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=12, strides=(1, 1))(x_in_2_up)
    x_in_2_up = tf.keras.layers.Normalization()(x_in_2_up)
    x_in_2_up = tf.keras.layers.ReLU()(x_in_2_up)

    # Third upscale branch
    x_in_3_up = tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=9, strides=(1, 1))(x_in_3_down)
    x_in_3_up = tf.keras.layers.Normalization()(x_in_3_up)
    x_in_3_up = tf.keras.layers.ReLU()(x_in_3_up)
    x_in_3_up = tf.keras.layers.concatenate([x_in_3_up, x_in_3_down_res_2])
    x_in_3_up = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=8, strides=(1, 1))(x_in_3_up)
    x_in_3_up = tf.keras.layers.Normalization()(x_in_3_up)
    x_in_3_up = tf.keras.layers.ReLU()(x_in_3_up)
    x_in_3_up = tf.keras.layers.concatenate([x_in_3_up, x_in_3_down_res_1])
    x_in_3_up = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=14, strides=(1, 1))(x_in_3_up)
    x_in_3_up = tf.keras.layers.Normalization()(x_in_3_up)
    x_in_3_up = tf.keras.layers.ReLU()(x_in_3_up)

    # Fourth upscale branch
    x_in_4_up = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=11, strides=(1, 1))(x_in_4_down)
    x_in_4_up = tf.keras.layers.Normalization()(x_in_4_up)
    x_in_4_up = tf.keras.layers.ReLU()(x_in_4_up)
    x_in_4_up = tf.keras.layers.concatenate([x_in_4_up, x_in_4_down_res_2])
    x_in_4_up = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=9, strides=(1, 1))(x_in_4_up)
    x_in_4_up = tf.keras.layers.Normalization()(x_in_4_up)
    x_in_4_up = tf.keras.layers.ReLU()(x_in_4_up)
    x_in_4_up = tf.keras.layers.concatenate([x_in_4_up, x_in_4_down_res_1])
    x_in_4_up = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=16, strides=(1, 1))(x_in_4_up)
    x_in_4_up = tf.keras.layers.Normalization()(x_in_4_up)
    x_in_4_up = tf.keras.layers.ReLU()(x_in_4_up)

    # Fifth upscale branch
    x_in_5_up = tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=13, strides=(1, 1))(x_in_5_down)
    x_in_5_up = tf.keras.layers.Normalization()(x_in_5_up)
    x_in_5_up = tf.keras.layers.ReLU()(x_in_5_up)
    x_in_5_up = tf.keras.layers.concatenate([x_in_5_up, x_in_5_down_res_2])
    x_in_5_up = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=12, strides=(1, 1))(x_in_5_up)
    x_in_5_up = tf.keras.layers.Normalization()(x_in_5_up)
    x_in_5_up = tf.keras.layers.ReLU()(x_in_5_up)
    x_in_5_up = tf.keras.layers.concatenate([x_in_5_up, x_in_5_down_res_1])
    x_in_5_up = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=18, strides=(1, 1))(x_in_5_up)
    x_in_5_up = tf.keras.layers.Normalization()(x_in_5_up)
    x_in_5_up = tf.keras.layers.ReLU()(x_in_5_up)

    # Sixth upscale branch
    x_in_6_up = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=15, strides=(1, 1))(x_in_6_down)
    x_in_6_up = tf.keras.layers.Normalization()(x_in_6_up)
    x_in_6_up = tf.keras.layers.ReLU()(x_in_6_up)
    x_in_6_up = tf.keras.layers.concatenate([x_in_6_up, x_in_6_down_res_2])
    x_in_6_up = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=13, strides=(1, 1))(x_in_6_up)
    x_in_6_up = tf.keras.layers.Normalization()(x_in_6_up)
    x_in_6_up = tf.keras.layers.ReLU()(x_in_6_up)
    x_in_6_up = tf.keras.layers.concatenate([x_in_6_up, x_in_6_down_res_1])
    x_in_6_up = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=20, strides=(1, 1))(x_in_6_up)
    x_in_6_up = tf.keras.layers.Normalization()(x_in_6_up)
    x_in_6_up = tf.keras.layers.ReLU()(x_in_6_up)

    # Upscale deconv
    # x_d_up = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=521, strides=(1, 1))(concat_deconvs)
    # x_d_up = tf.keras.layers.Normalization()(x_d_up)
    # x_d_up = tf.keras.layers.ReLU()(x_d_up)

    span_xin = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=9, strides=(1, 1))(x_in)
    span_xin = tf.keras.layers.Normalization()(span_xin)
    span_xin = tf.keras.layers.ReLU()(span_xin)
    span_xin = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=1, strides=(1, 1))(span_xin)
    span_xin = tf.keras.layers.Normalization()(span_xin)
    span_xin = tf.keras.layers.ReLU()(span_xin)

    # adds135 = tf.keras.layers.Add()([x_in_1_up, x_in_3_up, x_in_5_up])
    # adds246 = tf.keras.layers.Add()([x_in_2_up, x_in_4_up, x_in_6_up])
    all_concat = tf.keras.layers.concatenate([x_in_1_up, x_in_2_up, x_in_3_up, x_in_4_up, x_in_5_up, x_in_6_up, span_xin])
    all_concat = tf.keras.layers.Conv2D(filters=3, kernel_size=9, strides=(1, 1))(all_concat)
    all_concat = tf.keras.layers.Normalization()(all_concat)
    all_concat = tf.keras.layers.Activation('relu', dtype=tf.float32)(all_concat)
    # all_concat = tf.keras.layers.AveragePooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')(all_concat)

    if pretrained_path is not None:

        final_model = tf.keras.Model(inputs=inputs, outputs=all_concat)
    else:
        final_model = tf.keras.Model(inputs=x_in, outputs=all_concat)

    final_model.compile(loss=loss, metrics=[ms_ssim, ssim, 'mse', peak_snr], optimizer=optimizer)
    final_model.summary()

    return final_model


class ConvBlock(tf.keras.layers.Layer):

    def __init__(self, layer_idx, filters_root, kernel_size, dropout_rate, padding, activation, **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        self.layer_idx=layer_idx
        self.filters_root=filters_root
        self.kernel_size=kernel_size
        self.dropout_rate=dropout_rate
        self.padding=padding
        self.activation=activation

        filters = _get_filter_count(layer_idx, filters_root)
        self.conv2d_1 = tf.keras.layers.Conv2D(filters=filters,
                                      kernel_size=(kernel_size, kernel_size),
                                      kernel_initializer='glorot_normal',
                                      strides=1,
                                      padding=padding)
        self.dropout_1 = tf.keras.layers.Dropout(rate=dropout_rate)
        self.activation_1 = tf.keras.layers.Activation(activation)

        self.conv2d_2 = tf.keras.layers.Conv2D(filters=filters,
                                      kernel_size=(kernel_size, kernel_size),
                                      kernel_initializer='glorot_normal',
                                      strides=1,
                                      padding=padding)
        self.dropout_2 = tf.keras.layers.Dropout(rate=dropout_rate)
        self.activation_2 = tf.keras.layers.Activation(activation)

    def call(self, inputs, training=None, **kwargs):
        x = inputs
        x = self.conv2d_1(x)

        if training:
            x = self.dropout_1(x)
        x = self.activation_1(x)
        x = self.conv2d_2(x)

        if training:
            x = self.dropout_2(x)

        x = self.activation_2(x)
        return x

    def get_config(self):
        return dict(layer_idx=self.layer_idx,
                    filters_root=self.filters_root,
                    kernel_size=self.kernel_size,
                    dropout_rate=self.dropout_rate,
                    padding=self.padding,
                    activation=self.activation,
                    **super(ConvBlock, self).get_config(),
                    )


class UpconvBlock(tf.keras.layers.Layer):

    def __init__(self, layer_idx, filters_root, kernel_size, pool_size, padding, activation, **kwargs):
        super(UpconvBlock, self).__init__(**kwargs)
        self.layer_idx=layer_idx
        self.filters_root=filters_root
        self.kernel_size=kernel_size
        self.pool_size=pool_size
        self.padding=padding
        self.activation=activation

        filters = _get_filter_count(layer_idx + 1, filters_root)
        self.upconv = tf.keras.layers.Conv2DTranspose(filters // 2,
                                             kernel_size=(pool_size, pool_size),
                                             kernel_initializer='glorot_normal',
                                             strides=pool_size, padding=padding)

        self.activation_1 = tf.keras.layers.Activation(activation)

    def call(self, inputs, **kwargs):
        x = inputs
        x = self.upconv(x)
        x = self.activation_1(x)
        return x

    def get_config(self):
        return dict(layer_idx=self.layer_idx,
                    filters_root=self.filters_root,
                    kernel_size=self.kernel_size,
                    pool_size=self.pool_size,
                    padding=self.padding,
                    activation=self.activation,
                    **super(UpconvBlock, self).get_config(),
                    )

class CropConcatBlock(tf.keras.layers.Layer):

    def call(self, x, down_layer, **kwargs):
        x1_shape = tf.shape(down_layer)
        x2_shape = tf.shape(x)

        height_diff = (x1_shape[1] - x2_shape[1]) // 2
        width_diff = (x1_shape[2] - x2_shape[2]) // 2

        down_layer_cropped = down_layer[:,
                                        height_diff: (x2_shape[1] + height_diff),
                                        width_diff: (x2_shape[2] + width_diff),
                                        :]

        x = tf.concat([down_layer_cropped, x], axis=-1)
        return x


def build_unet_model(nx: Optional[int] = None,
                ny: Optional[int] = None,
                channels: int = 3,
                num_classes: int = 3,
                layer_depth: int = 3,
                filters_root: int = 32,
                kernel_size: int = 3,
                pool_size: int = 2,
                dropout_rate: int = 0.5,
                padding:str="valid",
                activation:Union[str, Callable]="relu") -> tf.keras.Model:
    """
    Constructs a U-Net model
    :param nx: (Optional) image size on x-axis
    :param ny: (Optional) image size on y-axis
    :param channels: number of channels of the input tensors
    :param num_classes: number of classes
    :param layer_depth: total depth of unet
    :param filters_root: number of filters in top unet layer
    :param kernel_size: size of convolutional layers
    :param pool_size: size of maxplool layers
    :param dropout_rate: rate of dropout
    :param padding: padding to be used in convolutions
    :param activation: activation to be used
    :return: A TF Keras model
    """

    inputs = tf.keras.Input(shape=(nx, ny, channels), name="inputs")
    deconv_1 = WienerDeconv2D(filters=1, kernel_size=(nx, ny), padding=((0, 0), (0, 0)))(inputs)

    x = inputs
    contracting_layers = {}

    conv_params = dict(filters_root=filters_root,
                       kernel_size=kernel_size,
                       dropout_rate=dropout_rate,
                       padding=padding,
                       activation=activation)

    for layer_idx in range(0, layer_depth - 1):
        x = ConvBlock(layer_idx, **conv_params)(x)
        contracting_layers[layer_idx] = x
        x = tf.keras.layers.MaxPooling2D((pool_size, pool_size))(x)

    x = ConvBlock(layer_idx + 1, **conv_params)(x)

    for layer_idx in range(layer_idx, -1, -1):
        x = UpconvBlock(layer_idx,
                        filters_root,
                        kernel_size,
                        pool_size,
                        padding,
                        activation)(x)
        x = CropConcatBlock()(x, contracting_layers[layer_idx])
        x = ConvBlock(layer_idx, **conv_params)(x)

    x = tf.keras.layers.Conv2D(filters=num_classes,
                      kernel_size=(1, 1),
                      kernel_initializer="glorot_normal",
                      strides=1,
                      padding=padding)(x)

    x = tf.keras.layers.Activation(activation)(x)
    x = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=41, strides=(1, 1))(x)
    x = tf.keras.layers.Add()([x, deconv_1])
    x = tf.keras.layers.Conv2D(filters=3, kernel_size=1, strides=(1, 1))(x)
    outputs = tf.keras.layers.Activation("relu", name="outputs")(x)
    model = tf.keras.Model(inputs, outputs, name="unet")

    finalize_model(model)
    model.summary()

    return model


def _get_filter_count(layer_idx, filters_root):
    return 2 ** layer_idx * filters_root


def _get_kernel_initializer(filters, kernel_size):
    stddev = tf.cast(np.sqrt(2 / (kernel_size ** 2 * filters)), tf.float32)
    return tf.keras.initializers.TruncatedNormal(stddev=stddev)


def finalize_model(model: tf.keras.Model,
                   loss=ms_ssim,
                   optimizer: Optional = None,
                   metrics=('mse', peak_snr),
                   **opt_kwargs):
    """
    Configures the model for training by setting, loss, optimzer, and tracked metrics
    :param model: the model to compile
    :param loss: the loss to be optimized. Defaults to `categorical_crossentropy`
    :param optimizer: the optimizer to use. Defaults to `Adam`
    :param metrics: List of metrics to track. Is extended by `crossentropy` and `accuracy`
    :param dice_coefficient: Flag if the dice coefficient metric should be tracked
    :param auc: Flag if the area under the curve metric should be tracked
    :param mean_iou: Flag if the mean over intersection over union metric should be tracked
    :param opt_kwargs: key word arguments passed to default optimizer (Adam), e.g. learning rate
    """

    if optimizer is None:
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001,
                                             beta_1=0.9, beta_2=0.999,
                                             epsilon=1e-8,
                                             clipvalue=2)

    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics,
                  )

