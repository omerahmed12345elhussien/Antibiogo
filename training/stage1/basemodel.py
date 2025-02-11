"""
* basemodel.py contains The baseline model architecture.
* we use MobileNetv2 as the baseline. We are following these two implementations:
    - https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py
    - https://www.tensorflow.org/tutorials/images/segmentation#define_the_model

Reference:
    - [MobileNetV2: Inverted Residuals and Linear Bottlenecks](
        https://arxiv.org/abs/1801.04381) (CVPR 2018)

"""
from tensorflow_examples.models.pix2pix import pix2pix
from ssl import _create_default_https_context, _create_unverified_context
from utils import IMG_SIZE,optimAdam,OUTPUT_CLASSES
import tensorflow as tf

def unet_model(output_channels:int,model_version:int=2):
    _create_default_https_context = _create_unverified_context
    if model_version==2:
        base_model = tf.keras.applications.MobileNetV2(input_shape=[IMG_SIZE, IMG_SIZE, 3],
                                                    alpha=0.35,
                                                    include_top=False)
        # Use the activations of these layers
        layer_names = [
            'block_1_expand_relu',   # (Batch_size x Height x Width x No. channels): (None x 128 x 128 x 96)
            'block_3_expand_relu',   # (None x 64 x 64 x 144) 
            'block_6_expand_relu',   # (None x 32 x 32 x 192) 
            'block_13_expand_relu',  # (None x 16 x 16 x 576) 
            'block_16_project',      # (None x 8 x 8 x 960) -> (None x 8 x 8 x 320)
        ]
    elif model_version==3:
        base_model= tf.keras.applications.MobileNetV3Small(
            input_shape=[IMG_SIZE, IMG_SIZE, 3],
            alpha=0.75,
            include_top=False,
            input_tensor=None,
            dropout_rate=0,
            classifier_activation=None,
            include_preprocessing=False
            )
        # Use the activations of these layers
        layer_names = [
            'activation',       # (Batch_size x Height x Width x No. channels): (None x 128 x 128 x 16)
            're_lu_2',          # (None x 64 x 64 x 72) 
            'activation_1',     # (None x 32 x 32 x 96) 
            'activation_11',    # (None x 16 x 16 x 240) 
            'activation_15',    # (None x 8 x 8 x 432) 
        ]
    elif model_version==4:
        base_model=tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_tensor=None,
            input_shape=[IMG_SIZE, IMG_SIZE, 3],
            pooling=None,
            classifier_activation=None
            )
        # Use the activations of these layers
        layer_names = [
            'block1a_activation',       # (Batch_size x Height x Width x No. channels): (None x 128 x 128 x 32)
            'block2b_activation',          # (None x 64 x 64 x 144) 
            'block3b_activation',     # (None x 32 x 32 x 240) 
            'block5c_activation',    # (None x 16 x 16 x 672) 
            'block7a_activation',    # (None x 8 x 8 x 1152) 
        ]
    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]
    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)
    down_stack.trainable = False
    up_stack = [
        pix2pix.upsample(512, 3),  # 8x8 -> 16x16
        pix2pix.upsample(256, 3),  # 16x16 -> 32x32
        pix2pix.upsample(128, 3),  # 32x32 -> 64x64
        pix2pix.upsample(64, 3),   # 64x64 -> 128x128
    ]
    inputs = tf.keras.layers.Input(shape=[IMG_SIZE, IMG_SIZE, 3])
    # Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])
    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])
    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
        filters=output_channels, 
        kernel_size=3,
        strides=2,
        padding='same')  #128x128 -> 256x256
    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)
