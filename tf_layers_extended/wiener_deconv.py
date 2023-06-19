import tensorflow as tf
from tensorflow.keras import backend as bend
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras import activations
from nn_ops_extent import deconv1d, deconv2d


class WienerAbstractDeconvolution(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 padding,
                 activation,
                 use_bias,
                 kernel_initializer,
                 lambd_initializer,
                 bias_initializer,
                 kernel_regularizer,
                 lambd_regularizer,
                 bias_regularizer,
                 activity_regularizer,
                 trainable,
                 name,
                 **kwargs):
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.lambd_initializer = initializers.get(lambd_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.lambd_regularizer = regularizers.get(lambd_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.use_bias = use_bias

        super(WienerAbstractDeconvolution, self).__init__(
            trainable=trainable,
            name=name,
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)

    def build(self, input_shape):
        raise NotImplementedError

    def call(self, inputs, *args, **kwargs):
        raise NotImplementedError

    def _serialize_to_tensors(self):
        pass

    def _restore_from_tensors(self, restored_tensors):
        pass

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'padding': self.padding,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'lambd_initializer':  initializers.serialize(self.lambd_initializer),
            'bias_initializer':  initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'lambd_regularizer': regularizers.serialize(self.lambd_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer)
        })
        return config


class WienerDeconvolution1D(WienerAbstractDeconvolution):
    def __init__(self,
                 filters: int,
                 kernel_size: tuple,
                 padding=(None,),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='lecun_normal',
                 lambd_initializer='lecun_normal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 lambd_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(WienerDeconvolution1D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            padding=padding,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            lambd_initializer=lambd_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            lambd_regularizer=lambd_regularizer,
            bias_regularizer=bias_regularizer,
            trainable=trainable,
            name=name,
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)
        self.w_real = None
        self.w_imag = None
        self.s = None
        self.b = None

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        self.w_real = self.add_weight(shape=(self.filters, self.kernel_size[0]),
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      trainable=True,
                                      name='w_real',
                                      dtype='float32')
        self.w_imag = self.add_weight(shape=self.w_real.shape,
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      trainable=True,
                                      name='w_imag',
                                      dtype='float32')
        self._pad_filters2match_input(input_shape)
        self.s = self.add_weight(shape=(self.filters, 1),
                                 initializer=self.lambd_initializer,
                                 regularizer=self.lambd_regularizer,
                                 trainable=True,
                                 name='snr',
                                 dtype='float32')
        if self.use_bias:
            self.b = self.add_weight(shape=(1, self.filters * input_shape[-1]),
                                     initializer=self.bias_initializer,
                                     regularizer=self.bias_regularizer,
                                     trainable=True,
                                     name='bias',
                                     dtype='float32')

    @tf.function
    def call(self, inputs, *args, **kwargs):
        x = tf.cast(inputs, dtype=tf.float32)
        # Make sure input shape corresponds to convention of (batch size, timestamps, features)
        assert bend.ndim(x) == 3, f'Inputs shape must be of form (batch size, #timestamps, #features, ' \
                                  f'yours is of form {bend.ndim(x)}'
        # Apply padding to input if specified
        if self.padding[0] is not None:
            if len(self.padding) == 2:
                x = tf.pad(x, ((0, 0), (0, self.padding[0]), (0, 0)), mode=self.padding[-1])
            else:
                x = tf.pad(x, ((0, 0), (0, self.padding[0]), (0, 0)))
        assert x.shape[1] == self.w_real.shape[-1], 'Input and kernels must have equal shapes. Reduce filters ' \
                                                    'length, use input padding or increase input length.'
        # Perform Wiener deconvolution with trainable complex weights and SNRs
        x = deconv1d(input_vect=x, filters=(self.w_real, self.w_imag), lambds=self.s)
        # Apply bias accordingly
        if self.use_bias:
            x = x + self.b
        # And activation
        if self.activation is not None:
            x = self.activation(x)
        return x

    def _pad_filters2match_input(self, input_shape):
        # Determine pad length
        if self.padding[0] is not None:
            len_pad = input_shape[1] + self.padding[0] - self.kernel_size[0]
        else:
            len_pad = input_shape[1] - self.kernel_size[0]
        # Pad both real and imaginary weights
        if len_pad > 0:
            self.w_real = tf.pad(self.w_real, ((0, 0), (0, len_pad)), 'constant')
            self.w_imag = tf.pad(self.w_imag, ((0, 0), (0, len_pad)), 'constant')


class WienerDeconvolution2D(WienerAbstractDeconvolution):

    def __init__(self,
                 filters: int,
                 kernel_size: tuple,
                 padding=((0, 0), (0, 0)),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_normal',
                 lambd_initializer='glorot_normal',
                 bias_initializer='glorot_normal',
                 kernel_regularizer=None,
                 lambd_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(WienerDeconvolution2D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            padding=padding,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            lambd_initializer=lambd_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            lambd_regularizer=lambd_regularizer,
            bias_regularizer=bias_regularizer,
            trainable=trainable,
            name=name,
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)
        unzipped_pads = [val for vals in self.padding for val in vals]
        if any(pad is not None for pad in unzipped_pads):
            assert all(pad is not None for pad in unzipped_pads), \
                'Incomplete padding values. Please provide values for all padding positions of form ' \
                '((height_top, height_bottom), (width_left, width_right))'
        self.w = None
        self.s = None
        self.b = None

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        self.w = self.add_weight(shape=(self.filters,
                                        input_shape[1] + self.padding[0][0] + self.padding[0][1],
                                        input_shape[2] + self.padding[1][0] + self.padding[1][1]),
                                 initializer=self.kernel_initializer,
                                 regularizer=self.kernel_regularizer,
                                 trainable=True,
                                 name='w',
                                 dtype='float32')

        self.s = self.add_weight(shape=(1, ),
                                 initializer=self.lambd_initializer,
                                 regularizer=self.lambd_regularizer,
                                 trainable=True,
                                 name='snr',
                                 dtype='float32')
        if self.use_bias:
            self.b = self.add_weight(shape=(1, 1, self.filters * input_shape[-1]),
                                     initializer=self.bias_initializer,
                                     regularizer=self.bias_regularizer,
                                     trainable=True,
                                     name='bias',
                                     dtype='float32')

    @tf.function
    def call(self, inputs, *args, **kwargs):
        x = tf.cast(inputs, dtype=tf.float32)
        # Make sure input shape corresponds to convention of (batch size, height, width, channels)
        assert bend.ndim(x) == 4, f'Inputs shape must be of form (batch size, width, height, channels)' \
                                  f'yours is of form {bend.ndim(x)}'

        # Apply padding to input if specified
        x = tf.pad(x,
                   ((0, 0), (self.padding[0][0], self.padding[0][1]), (self.padding[1][0], self.padding[1][1]), (0, 0)),
                   'constant')

        assert (x.shape[1] == self.w.shape[1]) and (x.shape[2] == self.w.shape[2]), \
            'Input and kernels must have equal shapes. Reduce filters ' \
            'width/height, use input padding or increase input width/height.'

        # Perform Wiener deconvolution with trainable weights and SNRs
        x = deconv2d(input_mat=x, filters=self.w, lambds=self.s)

        # Apply bias accordingly to resulted shape
        if self.use_bias:
            x = x + tf.broadcast_to(self.b, (tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[-1]))

        # And activation
        if self.activation is not None:
            x = self.activation(x)

        return x

    def _serialize_to_tensors(self):
        pass

    def _restore_from_tensors(self, restored_tensors):
        pass


WienerDeconv1D = WienerDeconvolution1D
WienerDeconv2D = WienerDeconvolution2D
