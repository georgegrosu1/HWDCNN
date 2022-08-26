import tensorflow as tf
from signal_extended import power_spectrum_db


class PowSpectrumMSE(tf.keras.losses.Loss):
    def __int__(self, *args, **kwargs):
        super().__int__(*args, **kwargs)

    def call(self, y_true, y_pred):
        # Instantiate MSE object
        err_f = tf.keras.losses.MeanSquaredError()

        # Eliminate last axis
        y_true = tf.squeeze(y_true, -1)

        # Get the power of y_true and y_pred and return their MSE
        y_true_psd = power_spectrum_db(y_true)
        y_pred_psd = power_spectrum_db(y_pred)
        return err_f(y_true_psd, y_pred_psd)
