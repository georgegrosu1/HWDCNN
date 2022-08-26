import tensorflow as tf


def power_spectrum_db(series, sides='two'):
    return 10 * tf.experimental.numpy.log10(power_spectrum(series, sides))


def power_spectrum(series, sides='two'):
    # TODO: Verify if it is required to detrend the series by mean
    # series -= tf.reshape(tf.math.reduce_mean(series, axis=-1), (series.shape[0], 1))
    # Set default sampling frequency to 1KHz times number of samples
    # tf.signal.fft requires complex values
    signal = tf.cast(series, tf.complex64)
    # Take FFT
    signal = tf.signal.fft(signal)
    # Consider sides
    if sides == 'one':
        # Need just positive or negative freq
        signal = signal[:, :(tf.shape(signal)[-1] // 2 + 1)]
    # Compute the power
    signal = (1.0 / float(tf.shape(series)[-1])) * tf.math.abs(signal)**2
    signal = tf.concat([tf.expand_dims(signal[:, 0], axis=-1),
                        signal[:, 1:-1],
                        tf.expand_dims(signal[:, -1], axis=-1)], axis=-1)

    return signal


# def acf_fft_method(series):
#
#
#     # Variance
#     var = tf.expand_dims(tf.math.reduce_variance(series, axis=-1), axis=-1)

    # Normalized data
    # ndata = data - numpy.mean(data)
    #
    # # Compute the FFT
    # fft = numpy.fft.fft(ndata, size)
    #
    # # Get the power spectrum
    # pwr = np.abs(fft) ** 2
    #
    # # Calculate the autocorrelation from inverse FFT of the power spectrum
    # acorr = numpy.fft.ifft(pwr).real / var / len(data)

# def auto_correlation(series):
#     n = series.shape[1]
#     data = tf.transpose(tf.constant(series), perm=(0, 2, 1))
#     mean = tf.math.reduce_mean(data, axis=-1)
#     c0 = tf.math.reduce_sum.sum((data - mean) ** 2) / tf.cast(n, tf.float32)
#
#     def r(h):
#         acf_lag = ((data[:n - h] - mean) * (data[h:] - mean)).sum() / float(n) / c0
#         return round(acf_lag, 3)
#
#     # Avoiding lag 0 calculation
#     x = tf.range(n)
#     acf_coeffs = map(r, x)
#     return acf_coeffs
