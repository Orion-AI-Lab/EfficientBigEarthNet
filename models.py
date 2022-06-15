# -*- coding: utf-8 -*-
#
# The models
#

import math

import tensorflow as tf
from tensorflow import einsum
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16, VGG19, ResNet50, ResNet101, ResNet152
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Conv2D,
    Conv3D,
    Dense,
    Flatten,
    GlobalAveragePooling1D,
    Input,
    Layer,
    LayerNormalization,
    Permute,
)
from tensorflow.keras.models import Model

# from einops import rearrange
from modules import cbam_module, coord_module, eca_module, ghost_module, se_module

SEED = 42

MODELS_CLASS = {
    "dense": "BigEarthModel",
    "ResNet18": "ResNet18BigEarthModel",
    "ResNet50": "ResNet50BigEarthModel",
    "ResNet_50": "ResNet_50",
    "ResNet101": "ResNet101BigEarthModel",
    "ResNet152": "ResNet152BigEarthModel",
    "VGG16": "VGG16BigEarthModel",
    "VGG19": "VGG19BigEarthModel",
    "KBranch": "DNN_model",
    "DenseNet121": "DenseNet121BigEarthModel",
    "DenseNet161": "DenseNet169BigEarthModel",
    "DenseNet201": "DenseNet201BigEarthModel",
    "WideResNet": "WideResNet",
    "EfficientNet": "EfficientNet",
    "EfficientNetB0": "EfficientNetB0",
    "EfficientNetB1": "EfficientNetB1",
    "EfficientNetB2": "EfficientNetB2",
    "EfficientNetB3": "EfficientNetB3",
    "EfficientNetB4": "EfficientNetB4",
    "EfficientNetB5": "EfficientNetB5",
    "EfficientNetB6": "EfficientNetB6",
    "EfficientNetB7": "EfficientNetB7",
    "ViT": "ViT",
    "Lambda": "LambdaResNet",
    "MLPMixer": "Mixer",
    "MLPMixer_Tiny": "Mixer_Tiny",
}


class BigEarthModel:
    def __init__(self, nb_class, resolution=(120, 120), modality="MM", dtype="float64"):
        self._nb_class = nb_class
        self._dtype = dtype
        self._resolution = resolution
        self._modality = modality

        def resize(input):
            if input.shape[1:3] != self._resolution:
                return tf.image.resize(
                    input, self._resolution, method=tf.image.ResizeMethod.BICUBIC
                )
            return input

        self._inputB02 = Input(
            shape=(
                120,
                120,
            ),
            dtype=tf.float32,
        )
        self._inputB03 = Input(
            shape=(
                120,
                120,
            ),
            dtype=tf.float32,
        )
        self._inputB04 = Input(
            shape=(
                120,
                120,
            ),
            dtype=tf.float32,
        )
        self._inputB08 = Input(
            shape=(
                120,
                120,
            ),
            dtype=tf.float32,
        )
        self._inputVV = Input(
            shape=(
                120,
                120,
            ),
            dtype=tf.float32,
        )
        self._inputVH = Input(
            shape=(
                120,
                120,
            ),
            dtype=tf.float32,
        )
        bands_10m = tf.stack(
            [
                self._inputB02,
                self._inputB03,
                self._inputB04,
                self._inputB08,
            ],
            axis=3,
        )
        bands_10m = resize(bands_10m)
        print("10m shape: {}".format(bands_10m.shape))

        self._inputB05 = Input(
            shape=(
                60,
                60,
            ),
            dtype=tf.float32,
        )
        self._inputB06 = Input(
            shape=(
                60,
                60,
            ),
            dtype=tf.float32,
        )
        self._inputB07 = Input(
            shape=(
                60,
                60,
            ),
            dtype=tf.float32,
        )
        self._inputB8A = Input(
            shape=(
                60,
                60,
            ),
            dtype=tf.float32,
        )
        self._inputB11 = Input(
            shape=(
                60,
                60,
            ),
            dtype=tf.float32,
        )
        self._inputB12 = Input(
            shape=(
                60,
                60,
            ),
            dtype=tf.float32,
        )
        bands_20m = tf.stack(
            [
                self._inputB05,
                self._inputB06,
                self._inputB07,
                self._inputB8A,
                self._inputB11,
                self._inputB12,
            ],
            axis=3,
        )
        bands_20m = resize(bands_20m)
        print("20m shape: {}".format(bands_20m.shape))

        s1_bands = resize(tf.stack([self._inputVV, self._inputVH], axis=3))
        s2_bands = tf.concat([bands_10m, bands_20m], axis=3)

        if self._modality == "S1":
            allbands = s1_bands
            inputs = [
                self._inputVV,
                self._inputVH,
            ]
        elif self._modality == "S2":
            allbands = s2_bands
            inputs = [
                self._inputB02,
                self._inputB03,
                self._inputB04,
                self._inputB05,
                self._inputB06,
                self._inputB07,
                self._inputB08,
                self._inputB8A,
                self._inputB11,
                self._inputB12,
            ]
        elif self._modality == "MM":
            allbands = tf.concat([s2_bands, s1_bands], axis=3)
            inputs = [
                self._inputB02,
                self._inputB03,
                self._inputB04,
                self._inputB05,
                self._inputB06,
                self._inputB07,
                self._inputB08,
                self._inputB8A,
                self._inputB11,
                self._inputB12,
                self._inputVV,
                self._inputVH,
            ]
        else:
            raise ValueError("Incompatible modality.")

        self._num_bands = allbands.shape[3]
        print("allbands shape: {}".format(allbands.shape))

        # Legacy attributes for BigEarthNet-S2 related models
        self.band_dict = {"band_10": bands_10m, "band_20": bands_20m}
        self.feature_size = 128
        self.nb_bands_10m = 4
        self.nb_bands_20m = 6
        self.bands_10m = self.band_dict["band_10"]
        self.bands_20m = self.band_dict["band_20"]

        # Create internal model
        self._logits = self._create_model_logits(allbands)

        # Add one last dense layer with biases and sigmoid activation
        # This is like having nb_class separate binary classifiers
        self._output = Dense(
            units=self._nb_class, dtype=self._dtype, activation="sigmoid", use_bias=True
        )(self._logits)

        self._model = Model(inputs=inputs, outputs=self._output)
        self._logits_model = Model(inputs=inputs, outputs=self._logits)

    @property
    def model(self):
        return self._model

    @property
    def logits_model(self):
        return self._logits_model

    def _create_model_logits(self, allbands):
        x = Flatten(dtype=self._dtype)(allbands)
        x = Dense(128, dtype=self._dtype)(x)
        x = Dense(64, dtype=self._dtype)(x)
        return x


class DenseNet121BigEarthModel(BigEarthModel):
    def __init__(self, nb_class, modality):
        super().__init__(nb_class, modality=modality)

    def _create_model_logits(self, allbands):
        num_bands = self._num_bands
        x = tf.keras.applications.DenseNet121(
            include_top=False,
            weights=None,
            input_shape=(120, 120, num_bands),
            pooling="avg",
        )(allbands)

        return x


class DenseNet169BigEarthModel(BigEarthModel):
    def __init__(self, nb_class, modality):
        super().__init__(nb_class, modality=modality)

    def _create_model_logits(self, allbands):
        num_bands = self._num_bands
        x = tf.keras.applications.DenseNet169(
            include_top=False,
            weights=None,
            input_shape=(120, 120, num_bands),
            pooling="avg",
        )(allbands)

        return x


class DenseNet201BigEarthModel(BigEarthModel):
    def __init__(self, nb_class, modality):
        super().__init__(nb_class, modality=modality)

    def _create_model_logits(self, allbands):
        num_bands = self._num_bands
        x = tf.keras.applications.DenseNet201(
            include_top=False,
            weights=None,
            input_shape=(120, 120, num_bands),
            pooling="avg",
        )(allbands)

        return x


class ResNet50BigEarthModel(BigEarthModel):
    def __init__(self, nb_class, modality):
        super().__init__(nb_class, modality=modality)

    def _create_model_logits(self, allbands):
        num_bands = self._num_bands
        x = ResNet50(
            include_top=False,
            weights=None,
            input_shape=(120, 120, num_bands),
            pooling="avg",
        )(allbands)

        return x


class ResNet101BigEarthModel(BigEarthModel):
    def __init__(self, nb_class, modality):
        super().__init__(nb_class, modality=modality)

    def _create_model_logits(self, allbands):
        num_bands = self._num_bands
        x = ResNet101(
            include_top=False,
            weights=None,
            input_shape=(120, 120, num_bands),
            pooling="avg",
        )(allbands)

        return x


class ResNet152BigEarthModel(BigEarthModel):
    def __init__(self, nb_class, modality):
        super().__init__(nb_class, modality=modality)

    def _create_model_logits(self, allbands):
        num_bands = self._num_bands
        x = ResNet152(
            include_top=False,
            weights=None,
            input_shape=(120, 120, num_bands),
            pooling="avg",
        )(allbands)

        return x


class VGG16BigEarthModel(BigEarthModel):
    def __init__(self, nb_class, modality):
        super().__init__(nb_class, modality=modality)

    def _create_model_logits(self, allbands):
        num_bands = self._num_bands
        x = VGG16(
            include_top=False,
            weights=None,
            input_shape=(120, 120, num_bands),
            pooling="avg",
        )(allbands)

        return x


class VGG19BigEarthModel(BigEarthModel):
    def __init__(self, nb_class, modality):
        super().__init__(nb_class, modality=modality)

    def _create_model_logits(self, allbands):
        num_bands = self._num_bands
        # Add VGG19
        x = VGG19(
            include_top=False,
            weights=None,
            input_shape=(120, 120, num_bands),
            pooling="avg",
        )(allbands)

        return x


class ResNet_50(BigEarthModel):
    def __init__(self, nb_class, modality, resolution=(120, 120), dtype="float64"):
        super().__init__(
            nb_class, modality=modality, resolution=resolution, dtype=dtype
        )

    def _create_model_logits(self, allbands):
        def identity_block(input_tensor, kernel_size, filters, stage, block):

            filters1, filters2, filters3 = filters
            conv_name_base = "res" + str(stage) + block + "_branch"
            bn_name_base = "bn" + str(stage) + block + "_branch"

            x = Conv2D(filters1, (1, 1), name=conv_name_base + "2a")(input_tensor)
            x = BatchNormalization(name=bn_name_base + "2a")(x)
            x = Activation("relu")(x)

            x = Conv2D(
                filters2, kernel_size, padding="same", name=conv_name_base + "2b"
            )(x)
            x = BatchNormalization(name=bn_name_base + "2b")(x)
            x = Activation("relu")(x)

            x = Conv2D(filters3, (1, 1), name=conv_name_base + "2c")(x)
            x = BatchNormalization(name=bn_name_base + "2c")(x)

            x = layers.add([x, input_tensor])
            x = Activation("relu")(x)
            return x

        def conv_block(
            input_tensor, kernel_size, filters, stage, block, strides=(2, 2)
        ):
            filters1, filters2, filters3 = filters
            conv_name_base = "res" + str(stage) + block + "_branch"
            bn_name_base = "bn" + str(stage) + block + "_branch"

            x = Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + "2a")(
                input_tensor
            )
            x = BatchNormalization(name=bn_name_base + "2a")(x)
            x = Activation("relu")(x)

            x = Conv2D(
                filters2, kernel_size, padding="same", name=conv_name_base + "2b"
            )(x)
            x = BatchNormalization(name=bn_name_base + "2b")(x)
            x = Activation("relu")(x)

            x = Conv2D(filters3, (1, 1), name=conv_name_base + "2c")(x)
            x = BatchNormalization(name=bn_name_base + "2c")(x)

            shortcut = Conv2D(
                filters3, (1, 1), strides=strides, name=conv_name_base + "1"
            )(input_tensor)
            shortcut = BatchNormalization(name=bn_name_base + "1")(shortcut)

            x = layers.add([x, shortcut])
            x = Activation("relu")(x)
            return x

        x = tf.keras.layers.ZeroPadding2D((3, 3))(allbands)
        x = Conv2D(64, (7, 7), strides=(2, 2), name="conv1")(x)
        x = BatchNormalization(name="bn_conv1")(x)
        x = Activation("relu")(x)
        x = tf.keras.layers.ZeroPadding2D(padding=(1, 1), name="pool1_pad")(x)
        x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = conv_block(x, 3, [64, 64, 256], stage=2, block="a", strides=(1, 1))
        x = identity_block(x, 3, [64, 64, 256], stage=2, block="b")
        x = identity_block(x, 3, [64, 64, 256], stage=2, block="c")

        x = conv_block(x, 3, [128, 128, 512], stage=3, block="a")
        x = identity_block(x, 3, [128, 128, 512], stage=3, block="b")
        x = identity_block(x, 3, [128, 128, 512], stage=3, block="c")
        x = identity_block(x, 3, [128, 128, 512], stage=3, block="d")

        x = conv_block(x, 3, [256, 256, 1024], stage=4, block="a")
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block="b")
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block="c")
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block="d")
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block="e")
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block="f")

        x = conv_block(x, 3, [512, 512, 2048], stage=5, block="a")
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block="b")
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block="c")

        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        return x


class EfficientNet(BigEarthModel):
    def __init__(self, nb_class, coefficients, modality):
        self.phi = coefficients["phi"]
        self.alpha = coefficients["alpha"] ** self.phi
        self.beta = coefficients["beta"] ** self.phi
        self.gamma = coefficients["gamma"] ** self.phi
        self.dropout_rate = coefficients["dropout"]
        self.attention_module = (
            eca_module  # se_module, cbam_module, coord_module, eca_module
        )
        self.ghost_conv = True
        super().__init__(
            nb_class,
            tuple(
                int(math.ceil(dimension * self.gamma / 10.0)) * 10
                for dimension in (60, 60)
            ),
            modality,
        )

    def _create_model_logits(self, allbands):
        def swish(x):
            return x * tf.nn.sigmoid(x)

        def round_filters(filters, multiplier):
            depth_divisor = 8
            min_depth = None
            min_depth = min_depth or depth_divisor
            filters = filters * multiplier
            new_filters = max(
                min_depth,
                int(filters + depth_divisor / 2) // depth_divisor * depth_divisor,
            )
            if new_filters < 0.9 * filters:
                new_filters += depth_divisor
            return int(new_filters)

        def round_repeats(repeats, multiplier):
            if not multiplier:
                return repeats
            return int(math.ceil(multiplier * repeats))

        def conv_bn(x, filters, kernel_size, strides, activation=True):
            if self.ghost_conv:
                x = ghost_module(x, filters, kernel_size, strides)
            else:
                x = tf.keras.layers.Conv2D(
                    filters=filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding="same",
                    kernel_initializer="he_normal",
                    kernel_regularizer=tf.keras.regularizers.l2(5e-4),
                )(x)
            x = tf.keras.layers.BatchNormalization(
                momentum=0.1, epsilon=1e-5, gamma_initializer="uniform"
            )(x)
            if activation:
                x = swish(x)
            return x

        def conv_bn_0(x, filters, kernel_size, strides, activation=True):
            x = tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding="same",
                kernel_initializer="he_normal",
                kernel_regularizer=tf.keras.regularizers.l2(5e-4),
            )(x)
            x = tf.keras.layers.BatchNormalization(
                momentum=0.1, epsilon=1e-5, gamma_initializer="uniform"
            )(x)
            if activation:
                x = swish(x)
            return x

        def depthwiseConv_bn(x, kernel_size, strides):
            x = tf.keras.layers.DepthwiseConv2D(
                kernel_size,
                padding="same",
                strides=strides,
                kernel_initializer="he_normal",
                kernel_regularizer=tf.keras.regularizers.l2(5e-4),
            )(x)
            x = tf.keras.layers.BatchNormalization(
                momentum=0.1, epsilon=1e-5, gamma_initializer="uniform"
            )(x)
            x = tf.keras.layers.Activation(tf.nn.relu6)(x)
            return x

        def MBConv_idskip(x, filters, drop_connect_rate, kernel_size, strides, t):
            x_input = x
            x = conv_bn(x, filters=x.shape[-1] * t, kernel_size=1, strides=1)
            x = depthwiseConv_bn(x, kernel_size=kernel_size, strides=strides)
            x = self.attention_module(x)
            x = swish(x)
            x = conv_bn(x, filters=filters, kernel_size=1, strides=1, activation=False)

            if strides == 1 and x.shape[-1] == x_input.shape[-1]:
                if drop_connect_rate:
                    x = tf.keras.layers.Dropout(rate=drop_connect_rate)(x)
                x = tf.keras.layers.add([x_input, x])
            return x

        def MBConv(x, filters, drop_connect_rate, kernel_size, strides, t, n):
            x = MBConv_idskip(x, filters, drop_connect_rate, kernel_size, strides, t)
            for _ in range(1, n):
                x = MBConv_idskip(x, filters, drop_connect_rate, kernel_size, 1, t)
            return x

        print(self.alpha, self.beta, self.gamma, self.dropout_rate)

        x = conv_bn_0(
            allbands,
            round_filters(32, self.beta),
            kernel_size=3,
            strides=2,
            activation=True,
        )
        x = MBConv(
            x,
            filters=round_filters(16, self.beta),
            kernel_size=3,
            drop_connect_rate=self.dropout_rate,
            strides=1,
            t=1,
            n=round_repeats(1, self.alpha),
        )
        x = MBConv(
            x,
            filters=round_filters(24, self.beta),
            kernel_size=3,
            drop_connect_rate=self.dropout_rate,
            strides=2,
            t=6,
            n=round_repeats(2, self.alpha),
        )
        x = MBConv(
            x,
            filters=round_filters(40, self.beta),
            kernel_size=5,
            drop_connect_rate=self.dropout_rate,
            strides=2,
            t=6,
            n=round_repeats(2, self.alpha),
        )
        x = MBConv(
            x,
            filters=round_filters(80, self.beta),
            kernel_size=3,
            drop_connect_rate=self.dropout_rate,
            strides=2,
            t=6,
            n=round_repeats(3, self.alpha),
        )
        x = MBConv(
            x,
            filters=round_filters(112, self.beta),
            kernel_size=5,
            drop_connect_rate=self.dropout_rate,
            strides=1,
            t=6,
            n=round_repeats(3, self.alpha),
        )
        x = MBConv(
            x,
            filters=round_filters(192, self.beta),
            kernel_size=5,
            drop_connect_rate=self.dropout_rate,
            strides=2,
            t=6,
            n=round_repeats(4, self.alpha),
        )
        x = MBConv(
            x,
            filters=round_filters(320, self.beta),
            kernel_size=3,
            drop_connect_rate=self.dropout_rate,
            strides=1,
            t=6,
            n=round_repeats(1, self.alpha),
        )

        x = conv_bn_0(
            x, filters=round_filters(1280, self.beta), kernel_size=1, strides=1
        )
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(rate=self.dropout_rate)(x)

        return x


class WideResNet(BigEarthModel):
    def __init__(self, nb_class, coefficients, modality):
        self.phi = coefficients["phi"]
        self.alpha = coefficients["alpha"] ** self.phi
        self.beta = coefficients["beta"] ** self.phi
        self.gamma = coefficients["gamma"] ** self.phi
        self.dropout_rate = coefficients["dropout"]
        self.attention_addition = True
        self.attention_module = (
            eca_module  # se_module, cbam_module, coord_module, eca_module
        )
        self.ghost_conv = False
        super().__init__(
            nb_class,
            tuple(
                int(math.ceil(dimension * self.gamma / 10.0)) * 10
                for dimension in (60, 60)
            ),
            modality,
        )

    def _create_model_logits(self, allbands):
        def _make_divisible(value, divisor=4, min_value=None):
            if min_value is None:
                min_value = divisor
            new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
            # Make sure that round-down does not exceed 10%.
            if new_value < 0.9 * value:
                new_value += divisor
            return new_value

        def initial_conv(input):
            x = tf.keras.layers.Conv2D(
                16,
                (3, 3),
                padding="same",
                kernel_initializer="he_normal",
                kernel_regularizer=tf.keras.regularizers.l2(5e-4),
                use_bias=False,
            )(input)

            x = tf.keras.layers.BatchNormalization(
                momentum=0.1, epsilon=1e-5, gamma_initializer="uniform"
            )(x)
            x = tf.keras.layers.Activation("relu")(x)
            return x

        def expand_conv(init, base, k, strides=(1, 1)):
            if self.ghost_conv:
                x = ghost_module(init, _make_divisible(base * k), (3, 3), strides)
            else:
                x = tf.keras.layers.Conv2D(
                    _make_divisible(base * k),
                    (3, 3),
                    padding="same",
                    strides=strides,
                    kernel_initializer="he_normal",
                    kernel_regularizer=tf.keras.regularizers.l2(5e-4),
                    use_bias=False,
                )(init)

            x = tf.keras.layers.BatchNormalization(
                momentum=0.1, epsilon=1e-5, gamma_initializer="uniform"
            )(x)
            x = tf.keras.layers.Activation("relu")(x)

            if self.ghost_conv:
                x = ghost_module(x, _make_divisible(base * k), (3, 3), 1)
            else:
                x = tf.keras.layers.Conv2D(
                    _make_divisible(base * k),
                    (3, 3),
                    padding="same",
                    kernel_initializer="he_normal",
                    kernel_regularizer=tf.keras.regularizers.l2(5e-4),
                    use_bias=False,
                )(x)
            if self.attention_addition:
                x = self.attention_module(x)

            if self.ghost_conv:
                skip = ghost_module(init, _make_divisible(base * k), (1, 1), strides)
            else:
                skip = tf.keras.layers.Conv2D(
                    _make_divisible(base * k),
                    (1, 1),
                    padding="same",
                    strides=strides,
                    kernel_initializer="he_normal",
                    kernel_regularizer=tf.keras.regularizers.l2(5e-4),
                    use_bias=False,
                )(init)
            m = Add()([x, skip])
            return m

        def conv1_block(input, k=1, dropout=0.0):
            init = input

            x = tf.keras.layers.BatchNormalization(
                momentum=0.1, epsilon=1e-5, gamma_initializer="uniform"
            )(input)
            x = tf.keras.layers.Activation("relu")(x)
            if self.ghost_conv:
                x = ghost_module(x, _make_divisible(16 * k), (3, 3), 1)
            else:
                x = tf.keras.layers.Conv2D(
                    _make_divisible(16 * k),
                    (3, 3),
                    padding="same",
                    kernel_initializer="he_normal",
                    kernel_regularizer=tf.keras.regularizers.l2(5e-4),
                    use_bias=False,
                )(x)

            if dropout > 0.0:
                x = tf.keras.layers.Dropout(dropout)(x)

            x = tf.keras.layers.BatchNormalization(
                momentum=0.1, epsilon=1e-5, gamma_initializer="uniform"
            )(x)
            x = tf.keras.layers.Activation("relu")(x)
            if self.ghost_conv:
                x = ghost_module(x, _make_divisible(16 * k), (3, 3), 1)
            else:
                x = tf.keras.layers.Conv2D(
                    _make_divisible(16 * k),
                    (3, 3),
                    padding="same",
                    kernel_initializer="he_normal",
                    kernel_regularizer=tf.keras.regularizers.l2(5e-4),
                    use_bias=False,
                )(x)
            if self.attention_addition:
                x = self.attention_module(x)

            m = Add()([init, x])
            return m

        def conv2_block(input, k=1, dropout=0.0):
            init = input

            x = tf.keras.layers.BatchNormalization(
                momentum=0.1, epsilon=1e-5, gamma_initializer="uniform"
            )(init)
            x = tf.keras.layers.Activation("relu")(x)
            if self.ghost_conv:
                x = ghost_module(x, _make_divisible(32 * k), (3, 3), 1)
            else:
                x = tf.keras.layers.Conv2D(
                    _make_divisible(32 * k),
                    (3, 3),
                    padding="same",
                    kernel_initializer="he_normal",
                    kernel_regularizer=tf.keras.regularizers.l2(5e-4),
                    use_bias=False,
                )(x)

            if dropout > 0.0:
                x = tf.keras.layers.Dropout(dropout)(x)

            x = tf.keras.layers.BatchNormalization(
                momentum=0.1, epsilon=1e-5, gamma_initializer="uniform"
            )(x)
            x = tf.keras.layers.Activation("relu")(x)
            if self.ghost_conv:
                x = ghost_module(x, _make_divisible(32 * k), (3, 3), 1)
            else:
                x = tf.keras.layers.Conv2D(
                    _make_divisible(32 * k),
                    (3, 3),
                    padding="same",
                    kernel_initializer="he_normal",
                    kernel_regularizer=tf.keras.regularizers.l2(5e-4),
                    use_bias=False,
                )(x)
            if self.attention_addition:
                x = self.attention_module(x)

            m = Add()([init, x])
            return m

        def conv3_block(input, k=1, dropout=0.0):
            init = input

            x = tf.keras.layers.BatchNormalization(
                momentum=0.1, epsilon=1e-5, gamma_initializer="uniform"
            )(input)
            x = tf.keras.layers.Activation("relu")(x)
            if self.ghost_conv:
                x = ghost_module(x, _make_divisible(64 * k), (3, 3), 1)
            else:
                x = tf.keras.layers.Conv2D(
                    _make_divisible(64 * k),
                    (3, 3),
                    padding="same",
                    kernel_initializer="he_normal",
                    kernel_regularizer=tf.keras.regularizers.l2(5e-4),
                    use_bias=False,
                )(x)

            if dropout > 0.0:
                x = tf.keras.layers.Dropout(dropout)(x)

            x = tf.keras.layers.BatchNormalization(
                momentum=0.1, epsilon=1e-5, gamma_initializer="uniform"
            )(x)
            x = tf.keras.layers.Activation("relu")(x)
            if self.ghost_conv:
                x = ghost_module(x, _make_divisible(64 * k), (3, 3), 1)
            else:
                x = tf.keras.layers.Conv2D(
                    _make_divisible(64 * k),
                    (3, 3),
                    padding="same",
                    kernel_initializer="he_normal",
                    kernel_regularizer=tf.keras.regularizers.l2(5e-4),
                    use_bias=False,
                )(x)
            if self.attention_addition:
                x = self.attention_module(x)

            m = Add()([init, x])
            return m

        def WResNet(depth=16, k=2, dropout=self.dropout_rate):
            # assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
            N = int((depth - 4) // 6 * self.alpha)

            x = initial_conv(allbands)
            nb_conv = 4

            x = expand_conv(x, 16, k)
            nb_conv += 2

            for _ in range(N - 1):
                x = conv1_block(x, k, dropout)
                nb_conv += 2

            x = tf.keras.layers.BatchNormalization(
                momentum=0.1, epsilon=1e-5, gamma_initializer="uniform"
            )(x)
            x = tf.keras.layers.Activation("relu")(x)

            x = expand_conv(x, 32, k, strides=(2, 2))
            nb_conv += 2

            for _ in range(N - 1):
                x = conv2_block(x, k, dropout)
                nb_conv += 2

            x = tf.keras.layers.BatchNormalization(
                momentum=0.1, epsilon=1e-5, gamma_initializer="uniform"
            )(x)
            x = tf.keras.layers.Activation("relu")(x)

            x = expand_conv(x, 64, k, strides=(2, 2))
            nb_conv += 2

            for _ in range(N - 1):
                x = conv3_block(x, k, dropout)
                nb_conv += 2

            x = tf.keras.layers.BatchNormalization(
                momentum=0.1, epsilon=1e-5, gamma_initializer="uniform"
            )(x)
            x = tf.keras.layers.Activation("relu")(x)
            x = tf.keras.layers.AveragePooling2D((8, 8))(x)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)

            print("Wide Residual Network-{}-{} created.".format(nb_conv, int(k)))
            return x

        return WResNet(10 * self.alpha, 2 * self.beta, self.dropout_rate)


class EfficientNetB0(BigEarthModel):
    def __init__(self, nb_class, modality):
        super().__init__(nb_class, modality=modality)

    def _create_model_logits(self, allbands):
        num_bands = self._num_bands
        # Add VGG19
        x = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights=None,
            input_shape=(120, 120, num_bands),
            pooling="avg",
        )(allbands)

        return x


class EfficientNetB1(BigEarthModel):
    def __init__(self, nb_class, modality):
        super().__init__(nb_class, modality=modality)

    def _create_model_logits(self, allbands):
        num_bands = self._num_bands
        # Add VGG19
        x = tf.keras.applications.EfficientNetB1(
            include_top=False,
            weights=None,
            input_shape=(120, 120, num_bands),
            pooling="avg",
        )(allbands)

        return x


class EfficientNetB2(BigEarthModel):
    def __init__(self, nb_class, modality):
        super().__init__(nb_class, modality=modality)

    def _create_model_logits(self, allbands):
        num_bands = self._num_bands
        # Add VGG19
        x = tf.keras.applications.EfficientNetB2(
            include_top=False,
            weights=None,
            input_shape=(120, 120, num_bands),
            pooling="avg",
        )(allbands)

        return x


class EfficientNetB3(BigEarthModel):
    def __init__(self, nb_class, modality):
        super().__init__(nb_class, modality=modality)

    def _create_model_logits(self, allbands):
        num_bands = self._num_bands
        # Add VGG19
        x = tf.keras.applications.EfficientNetB3(
            include_top=False,
            weights=None,
            input_shape=(120, 120, num_bands),
            pooling="avg",
        )(allbands)

        return x


class EfficientNetB4(BigEarthModel):
    def __init__(self, nb_class, modality):
        super().__init__(nb_class, modality=modality)

    def _create_model_logits(self, allbands):
        num_bands = self._num_bands
        # Add VGG19
        x = tf.keras.applications.EfficientNetB4(
            include_top=False,
            weights=None,
            input_shape=(120, 120, num_bands),
            pooling="avg",
        )(allbands)

        return x


class EfficientNetB5(BigEarthModel):
    def __init__(self, nb_class, modality):
        super().__init__(nb_class, modality=modality)

    def _create_model_logits(self, allbands):
        num_bands = self._num_bands
        # Add VGG19
        x = tf.keras.applications.EfficientNetB5(
            include_top=False,
            weights=None,
            input_shape=(120, 120, num_bands),
            pooling="avg",
        )(allbands)

        return x


class EfficientNetB6(BigEarthModel):
    def __init__(self, nb_class, modality):
        super().__init__(nb_class, modality=modality)

    def _create_model_logits(self, allbands):
        num_bands = self._num_bands
        # Add VGG19
        x = tf.keras.applications.EfficientNetB6(
            include_top=False,
            weights=None,
            input_shape=(120, 120, num_bands),
            pooling="avg",
        )(allbands)

        return x


class EfficientNetB7(BigEarthModel):
    def __init__(self, nb_class, modality):
        super().__init__(nb_class, modality=modality)

    def _create_model_logits(self, allbands):
        num_bands = self._num_bands
        # Add VGG19
        x = tf.keras.applications.EfficientNetB7(
            include_top=False,
            weights=None,
            input_shape=(120, 120, num_bands),
            pooling="avg",
        )(allbands)

        return x


class DNN_model(BigEarthModel):
    def __init__(self, nb_class, modality):
        super().__init__(nb_class, modality=modality)

    def fully_connected_block(self, inputs, nb_neurons, name):
        fully_connected_res = tf.keras.layers.Dense(
            units=nb_neurons,
            activation=None,
            use_bias=True,
            kernel_initializer=tf.keras.initializers.GlorotNormal(seed=SEED),
            bias_initializer=tf.keras.initializers.Zeros(),
            kernel_regularizer=tf.keras.regularizers.l2(l=2e-5),
            bias_regularizer=None,
            activity_regularizer=None,
            trainable=True,
            name=name,
        )(inputs)
        batch_res = tf.keras.layers.BatchNormalization()(fully_connected_res)
        return tf.nn.relu(features=batch_res, name="relu")

    def conv_block(self, inputs, nb_filter, filter_size, name):
        conv_res = tf.keras.layers.Conv2D(
            filters=nb_filter,
            kernel_size=filter_size,
            strides=(1, 1),
            padding="same",
            data_format="channels_last",
            dilation_rate=(1, 1),
            activation=None,
            use_bias=True,
            kernel_initializer=tf.keras.initializers.GlorotNormal(seed=SEED),
            bias_initializer=tf.keras.initializers.Zeros(),
            kernel_regularizer=tf.keras.regularizers.l2(l=2e-5),
            bias_regularizer=None,
            activity_regularizer=None,
            trainable=True,
            name=name,
        )(inputs)
        batch_res = tf.keras.layers.BatchNormalization()(conv_res)

        return tf.nn.relu(features=batch_res, name="relu")

    def pooling(self, inputs, name):
        return tf.nn.max_pool(
            input=inputs,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding="VALID",
            data_format="NHWC",
            name=name,
        )

    def dropout(self, inputs, drop_rate, name):
        return tf.nn.dropout(
            inputs, rate=drop_rate, noise_shape=None, seed=SEED, name=name
        )

    def convert_image_to_uint8(self, img_batch):
        return tf.map_fn(
            lambda x: tf.image.convert_image_dtype(
                (tf.image.per_image_standardization(x) + 1.0) / 2.0,
                dtype=tf.uint8,
                saturate=True,
            ),
            img_batch,
            dtype=tf.uint8,
        )

    def branch_model_10m(self, inputs):
        out = self.conv_block(inputs, 32, [5, 5], "conv_block_10_0")
        out = self.pooling(out, "max_pooling_10")
        out = self.dropout(out, 0.25, "dropout_10_0")

        out = self.conv_block(out, 32, [5, 5], "conv_block_10_1")
        out = self.pooling(out, "max_pooling_10_1")
        out = self.dropout(out, 0.25, "dropout_10_1")
        out = self.conv_block(out, 64, [3, 3], "conv_block_10_2")
        out = self.dropout(out, 0.25, "dropout_10_2")
        out = tf.keras.layers.Flatten()(out)
        out = self.fully_connected_block(out, self.feature_size, "fc_block_10_0")
        feature = self.dropout(out, 0.5, "dropout_10_3")
        return feature

    def branch_model_20m(self, inputs):
        out = self.conv_block(inputs, 32, [3, 3], "conv_block_20_0")
        out = self.pooling(out, "max_pooling_20_0")
        out = self.dropout(out, 0.25, "dropout_20_0")
        out = self.conv_block(out, 32, [3, 3], "conv_block_20_1")
        out = self.dropout(out, 0.25, "dropout_20_1")
        out = self.conv_block(out, 64, [3, 3], "conv_block_20_2")
        out = self.dropout(out, 0.25, "dropout_20_2")
        out = tf.keras.layers.Flatten()(out)
        out = self.fully_connected_block(out, self.feature_size, "fc_block_20_0")
        feature = self.dropout(out, 0.5, "dropout_20_3")
        return feature

    def branch_model_60m(self, inputs):
        out = self.conv_block(inputs, 32, [2, 2], "conv_block_60_0")
        out = self.dropout(out, 0.25, "dropout_60_0")
        out = self.conv_block(out, 32, [2, 2], "conv_block_60_1")
        out = self.dropout(out, 0.25, "dropout_60_1")
        out = self.conv_block(out, 32, [2, 2], "conv_block_60_2")
        out = self.dropout(out, 0.25, "dropout_60_2")
        out = tf.keras.layers.Flatten()(out)
        out = self.fully_connected_block(out, self.feature_size, "fc_block_60_0")
        feature = self.dropout(out, 0.5, "dropout_60_3")
        return feature

    def _create_model_logits(self, allbands):
        branch_features = []
        for img_bands, nb_bands, branch_model, resolution in zip(
            [self.bands_10m, self.bands_20m, self.bands_60m],
            [self.nb_bands_10m, self.nb_bands_20m, self.nb_bands_60m],
            [self.branch_model_10m, self.branch_model_20m, self.branch_model_60m],
            ["_10", "_20", "_60"],
        ):
            print("Shape : ", img_bands.shape)
            branch_features.append(
                tf.reshape(branch_model(img_bands), [-1, self.feature_size])
            )

            patches_concat_embed_ = tf.concat(branch_features, -1)
            patches_concat_embed_ = self.fully_connected_block(
                patches_concat_embed_, self.feature_size, "fc_block_0" + resolution
            )
            patches_concat_embed_ = self.dropout(
                patches_concat_embed_, 0.25, "dropout_0" + resolution
            )

        return self.dropout(patches_concat_embed_, 0.5, "dropout_0" + resolution)


class Patches(tf.keras.layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = tf.keras.layers.Dense(units=projection_dim)
        self.position_embedding = tf.keras.layers.Dense(
            units=projection_dim, use_bias=False
        )  # self.add_weight(
        #    "pos_emb", shape=(1, num_patches + 1, projection_dim)
        # ) #tf.keras.layers.Embedding(
        #    input_dim=num_patches, output_dim=projection_dim
        # )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        positions = tf.expand_dims(positions, axis=1)

        emb = self.position_embedding(positions)
        # print(emb.shape,flush = True)
        proj = self.projection(patch)
        # print('projection shape : ', proj.shape, flush = True)
        encoded = proj + emb
        return encoded


class ViT(BigEarthModel):
    def __init__(self, nb_class, modality):
        super().__init__(nb_class, modality=modality)

    def _create_model_logits(self, allbands):
        # num_bands = self._num_bands

        # self.learning_rate = 0.001
        # weight_decay = 0.0001
        # batch_size = 256
        # num_epochs = 100
        image_size = 120  # We'll resize input images to this size
        self.patch_size = 6  # Size of the patches to be extract from the input images
        self.num_patches = (image_size // self.patch_size) ** 2
        self.projection_dim = 64
        self.num_heads = 4
        self.transformer_units = [
            self.projection_dim * 2,
            self.projection_dim,
        ]  # Size of the transformer layers
        self.transformer_layers = 8
        self.mlp_head_units = [2048, 1024]
        x = self.create_vit_classifier(allbands)

        return x

    def mlp(self, x, hidden_units, dropout_rate):
        for units in hidden_units:
            x = layers.Dense(units, activation=tf.nn.gelu)(x)
            x = layers.Dropout(dropout_rate)(x)
        return x

    def create_vit_classifier(self, allbands):
        num_bands = self._num_bands
        # Augment data.
        # augmented = data_augmentation(inputs)
        # Create patches.
        patches = Patches(self.patch_size)(allbands)
        # Encode patches.
        encoded_patches = PatchEncoder(self.num_patches, self.projection_dim)(patches)

        # Create multiple layers of the Transformer block.
        for _ in range(self.transformer_layers):
            # Layer normalization 1.
            x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            # Create a multi-head attention layer.
            attention_output = layers.MultiHeadAttention(
                num_heads=self.num_heads, key_dim=self.projection_dim, dropout=0.1
            )(x1, x1)
            # Skip connection 1.
            x2 = layers.Add()([attention_output, encoded_patches])
            # Layer normalization 2.
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            # MLP.
            x3 = self.mlp(x3, hidden_units=self.transformer_units, dropout_rate=0.1)
            # Skip connection 2.
            encoded_patches = layers.Add()([x3, x2])

        # Create a [batch_size, projection_dim] tensor.
        representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        representation = layers.Flatten()(representation)
        representation = layers.Dropout(0.5)(representation)
        # Add MLP.
        features = self.mlp(
            representation, hidden_units=self.mlp_head_units, dropout_rate=0.5
        )
        # Classify outputs.
        # logits = layers.Dense(num_classes)(features)
        # Create the Keras model.
        # model = tf.keras.Model(inputs=allbands, outputs=features)
        # model.summary()
        return features


# lambda layer


class LambdaLayer(Layer):
    def __init__(self, *, dim_k, n=None, r=None, heads=4, dim_out=None, dim_u=1):
        super(LambdaLayer, self).__init__()

        self.u = dim_u
        self.heads = heads

        assert (dim_out % heads) == 0
        dim_v = dim_out // heads

        self.to_q = Conv2D(filters=dim_k * heads, kernel_size=(1, 1), use_bias=False)
        self.to_k = Conv2D(filters=dim_k * dim_u, kernel_size=(1, 1), use_bias=False)
        self.to_v = Conv2D(filters=dim_v * dim_u, kernel_size=(1, 1), use_bias=False)

        self.norm_q = BatchNormalization()
        self.norm_v = BatchNormalization()

        self.local_contexts = r is not None

        if self.local_contexts:
            assert (r % 2) == 1, "Receptive kernel size should be odd."
            self.pad_fn = lambda x: tf.pad(
                x, tf.constant([[0, 0], [r // 2, r // 2], [r // 2, r // 2], [0, 0]])
            )
            self.pos_conv = Conv3D(filters=dim_k, kernel_size=(1, r, r))
            self.flatten = tf.keras.layers.Flatten()
        else:
            assert n is not None, "You must specify the total sequence length (h x w)"
            self.pos_emb = self.add_weight(
                name="position_embed", shape=(n, n, dim_k, dim_u)
            )

    def call(self, x):
        # For verbosity and understandings sake
        batch_size, height, width, channels, u, heads = *x.shape, self.u, self.heads
        b, hh, ww, c, u, h = batch_size, height, width, channels, u, heads

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = self.norm_q(q)
        v = self.norm_v(v)

        q = rearrange(q, "b hh ww (h k) -> b h (hh ww) k", h=h)
        k = rearrange(k, "b hh ww (k u) -> b u (hh ww) k", u=u)
        v = rearrange(v, "b hh ww (v u) -> b u (hh ww) v", u=u)

        k = tf.nn.softmax(k, axis=-1)

        lambda_c = einsum("b u m k, b u m v -> b k v", k, v)
        Y_c = einsum("b h n k, b k v -> b n h v", q, lambda_c)

        if self.local_contexts:
            v = rearrange(v, "b u (hh ww) v -> b v hh ww u", hh=hh, ww=ww)
            # We need to add explicit padding across the batch dimension
            lambda_p = tf.map_fn(self.pad_fn, v)
            lambda_p = self.pos_conv(lambda_p)
            lambda_p = tf.reshape(
                lambda_p,
                (
                    lambda_p.shape[0],
                    lambda_p.shape[1],
                    lambda_p.shape[2] * lambda_p.shape[3],
                    lambda_p.shape[4],
                ),
            )
            Y_p = einsum("b h n k, b v n k -> b n h v", q, lambda_p)
        else:
            lambda_p = einsum("n m k u, b u m v -> b n k v", self.pos_emb, v)
            Y_p = einsum("b h n k, b n k v -> b n h v", q, lambda_p)

        Y = Y_c + Y_p
        out = rearrange(Y, "b (hh ww) h v -> b hh ww (h v)", hh=hh, ww=ww)

        return out


class LambdaConv(Layer):
    def __init__(
        self,
        channels_out,
        *,
        receptive_field=None,
        key_dim=16,
        intra_depth_dim=1,
        heads=4
    ):
        super(LambdaConv, self).__init__()
        self.channels_out = channels_out
        self.receptive_field = receptive_field
        self.key_dim = key_dim
        self.intra_depth_dim = intra_depth_dim
        self.heads = heads

    def build(self, input_shape):
        self.layer = LambdaLayer(
            dim_out=self.channels_out,
            dim_k=self.key_dim,
            n=input_shape[1] * input_shape[2],
        )

    def call(self, x):
        return self.layer(x)


class LambdaResNet(BigEarthModel):
    def __init__(self, nb_class, modality):
        super().__init__(nb_class, modality=modality)

    def _create_model_logits(self, allbands):
        def initial_conv(input):
            x = tf.keras.layers.Conv2D(
                16,
                (3, 3),
                padding="same",
                kernel_initializer="he_normal",
                kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                use_bias=False,
            )(input)

            x = tf.keras.layers.BatchNormalization(
                momentum=0.1, epsilon=1e-5, gamma_initializer="uniform"
            )(x)
            x = tf.keras.layers.Activation("relu")(x)
            return x

        def expand_conv(init, base, k, strides=(1, 1)):
            x = tf.keras.layers.Conv2D(
                base * k,
                (3, 3),
                padding="same",
                strides=strides,
                kernel_initializer="he_normal",
                kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                use_bias=False,
            )(init)

            x = tf.keras.layers.BatchNormalization(
                momentum=0.1, epsilon=1e-5, gamma_initializer="uniform"
            )(x)
            x = tf.keras.layers.Activation("relu")(x)

            x = tf.keras.layers.Conv2D(
                base * k,
                (3, 3),
                padding="same",
                kernel_initializer="he_normal",
                kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                use_bias=False,
            )(x)

            skip = tf.keras.layers.Conv2D(
                base * k,
                (1, 1),
                padding="same",
                strides=strides,
                kernel_initializer="he_normal",
                kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                use_bias=False,
            )(init)

            m = Add()([x, skip])
            return m

        def conv1_block(input, k=1, dropout=0.0):
            init = input

            x = tf.keras.layers.BatchNormalization(
                momentum=0.1, epsilon=1e-5, gamma_initializer="uniform"
            )(input)
            x = tf.keras.layers.Activation("relu")(x)
            x = tf.keras.layers.Conv2D(
                8 * k,
                (3, 3),
                padding="same",
                kernel_initializer="he_normal",
                kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                use_bias=False,
            )(x)

            if dropout > 0.0:
                x = Dropout(dropout)(x)

            x = tf.keras.layers.BatchNormalization(
                momentum=0.1, epsilon=1e-5, gamma_initializer="uniform"
            )(x)
            x = tf.keras.layers.Activation("relu")(x)
            # x = tf.keras.layers.Conv2D(16 * k, (3, 3), padding='same', kernel_initializer='he_normal',
            #                           kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
            #                           use_bias=False)(x)
            print("Conv1")
            print(x.shape, flush=True)
            layer = LambdaConv(8 * k)
            x = layer(x)
            m = Add()([init, x])
            return m

        def conv2_block(input, k=1, dropout=0.0):
            init = input

            x = tf.keras.layers.BatchNormalization(
                momentum=0.1, epsilon=1e-5, gamma_initializer="uniform"
            )(init)
            x = tf.keras.layers.Activation("relu")(x)
            x = tf.keras.layers.Conv2D(
                16 * k,
                (3, 3),
                padding="same",
                kernel_initializer="he_normal",
                kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                use_bias=False,
            )(x)

            if dropout > 0.0:
                x = Dropout(dropout)(x)

            x = tf.keras.layers.BatchNormalization(
                momentum=0.1, epsilon=1e-5, gamma_initializer="uniform"
            )(x)
            x = tf.keras.layers.Activation("relu")(x)
            # x = tf.keras.layers.Conv2D(32 * k, (3, 3), padding='same', kernel_initializer='he_normal',
            #                           kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
            #                           use_bias=False)(x)
            layer = LambdaConv(16 * k)
            print("Conv2")
            print(x.shape, flush=True)
            x = layer(x)
            m = Add()([init, x])
            return m

        def conv3_block(input, k=1, dropout=0.0):
            init = input

            x = tf.keras.layers.BatchNormalization(
                momentum=0.1, epsilon=1e-5, gamma_initializer="uniform"
            )(input)
            x = tf.keras.layers.Activation("relu")(x)
            x = tf.keras.layers.Conv2D(
                32 * k,
                (3, 3),
                padding="same",
                kernel_initializer="he_normal",
                kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                use_bias=False,
            )(x)

            if dropout > 0.0:
                x = Dropout(dropout)(x)

            x = tf.keras.layers.BatchNormalization(
                momentum=0.1, epsilon=1e-5, gamma_initializer="uniform"
            )(x)
            x = tf.keras.layers.Activation("relu")(x)
            # x = tf.keras.layers.Conv2D(64 * k, (3, 3), padding='same', kernel_initializer='he_normal',
            #                           kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
            #                           use_bias=False)(x)
            layer = LambdaConv(32 * k)
            print("Conv3")
            print(x.shape, flush=True)
            x = layer(x)
            m = Add()([init, x])
            return m

        depth = 16
        k = 2
        dropout = 0.0
        weight_decay = 0.0005

        N = (depth - 4) // 6

        x = initial_conv(allbands)
        nb_conv = 4

        x = expand_conv(x, 16, k)
        nb_conv += 2

        for i in range(N - 1):
            x = conv1_block(x, k, dropout)
            nb_conv += 2

        x = tf.keras.layers.BatchNormalization(
            momentum=0.1, epsilon=1e-5, gamma_initializer="uniform"
        )(x)
        x = tf.keras.layers.Activation("relu")(x)

        x = expand_conv(x, 32, k, strides=(2, 2))
        nb_conv += 2

        for i in range(N - 1):
            x = conv2_block(x, k, dropout)
            nb_conv += 2

        x = tf.keras.layers.BatchNormalization(
            momentum=0.1, epsilon=1e-5, gamma_initializer="uniform"
        )(x)
        x = tf.keras.layers.Activation("relu")(x)

        x = expand_conv(x, 64, k, strides=(2, 2))
        nb_conv += 2

        for i in range(N - 1):
            x = conv3_block(x, k, dropout)
            nb_conv += 2

        x = tf.keras.layers.BatchNormalization(
            momentum=0.1, epsilon=1e-5, gamma_initializer="uniform"
        )(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.AveragePooling2D((8, 8))(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)

        print("Wide Residual Network-{}-{} created.".format(nb_conv, k))
        return x


class MlpBlock(Layer):
    def __init__(self, dim: int, hidden_dim: int, activation=None, **kwargs):
        super(MlpBlock, self).__init__(**kwargs)

        if activation is None:
            activation = tf.keras.activations.gelu

        self.dim = dim
        self.dense1 = Dense(hidden_dim)
        self.activation = Activation(activation)
        self.dense2 = Dense(dim)

    def call(self, inputs):
        x = inputs
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dense2(x)
        return x

    def compute_output_shape(self, input_signature):
        return (input_signature[0], self.dim)


class MixerBlock(Layer):
    def __init__(
        self,
        num_patches: int,
        channel_dim: int,
        token_mixer_hidden_dim: int,
        channel_mixer_hidden_dim: int = None,
        activation=None,
        **kwargs
    ):
        super(MixerBlock, self).__init__(**kwargs)

        if activation is None:
            activation = tf.keras.activations.gelu

        if channel_mixer_hidden_dim is None:
            channel_mixer_hidden_dim = token_mixer_hidden_dim

        self.norm1 = LayerNormalization(axis=1)
        self.permute1 = Permute((2, 1))
        self.token_mixer = MlpBlock(
            num_patches, token_mixer_hidden_dim, name="token_mixer"
        )

        self.permute2 = Permute((2, 1))
        self.norm2 = LayerNormalization(axis=1)
        self.channel_mixer = MlpBlock(
            channel_dim, channel_mixer_hidden_dim, name="channel_mixer"
        )

        self.skip_connection1 = Add()
        self.skip_connection2 = Add()

    def call(self, inputs):
        x = inputs
        skip_x = x
        x = self.norm1(x)
        x = self.permute1(x)
        x = self.token_mixer(x)

        x = self.permute2(x)

        x = self.skip_connection1([x, skip_x])
        skip_x = x

        x = self.norm2(x)
        x = self.channel_mixer(x)

        x = self.skip_connection2([x, skip_x])  # TODO need 2?

        return x

    def compute_output_shape(self, input_shape):
        return input_shape


# Implementation based on https://github.com/Benjamin-Etheredge/mlp-mixer-keras/blob/main/mlp_mixer_keras/mlp_mixer.py
def MlpMixerModel(
    input_shape: int,
    num_blocks: int,
    patch_size: int,
    hidden_dim: int,
    tokens_mlp_dim: int,
    channels_mlp_dim: int = None,
):
    height, width, _ = input_shape

    if channels_mlp_dim is None:
        channels_mlp_dim = tokens_mlp_dim

    num_patches = (height * width) // (
        patch_size**2
    )  # TODO verify how this behaves with same padding

    inputs = tf.keras.Input(input_shape)
    x = inputs

    x = Conv2D(
        hidden_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding="same",
        name="projector",
    )(x)

    x = tf.keras.layers.Reshape([-1, hidden_dim])(x)

    for _ in range(num_blocks):
        x = MixerBlock(
            num_patches=num_patches,
            channel_dim=hidden_dim,
            token_mixer_hidden_dim=tokens_mlp_dim,
            channel_mixer_hidden_dim=channels_mlp_dim,
        )(x)

    x = GlobalAveragePooling1D()(
        x
    )  # TODO verify this global average pool is correct choice here

    x = LayerNormalization(name="pre_head_layer_norm")(x)
    # x = Dense(num_classes, name='head')(x)

    # if use_softmax:
    #    x = Softmax()(x)
    return tf.keras.Model(inputs, x)


class Mixer(BigEarthModel):
    def __init__(self, nb_class, modality):
        super().__init__(nb_class, modality=modality)

    def _create_model_logits(self, allbands):
        num_bands = self._num_bands
        x = MlpMixerModel(
            input_shape=(120, 120, num_bands),
            num_blocks=4,  # 4,
            patch_size=12,
            hidden_dim=128,
            tokens_mlp_dim=64,
            channels_mlp_dim=200,
        )(allbands)
        return x


class Mixer_Tiny(BigEarthModel):
    def __init__(self, nb_class, modality):
        super().__init__(nb_class, modality=modality)

    def _create_model_logits(self, allbands):
        num_bands = self._num_bands
        x = MlpMixerModel(
            input_shape=(120, 120, num_bands),
            num_blocks=2,  # 4,
            patch_size=12,
            hidden_dim=64,
            tokens_mlp_dim=64,
            channels_mlp_dim=64,
        )(allbands)
        return x
