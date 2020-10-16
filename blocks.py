import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, UpSampling2D, Add, Input
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import mean_squared_error
import tensorflow.keras.backend as K

import numpy as np

def create_front_module(input, num_channels, bottleneck):
    x = Conv2D(64, kernel_size = (7, 7), strides = (2, 2), padding = 'same', activation = 'relu', name = 'front_conv_1x1_x1')(input)
    x = BatchNormalization()(x)
    x = bottleneck(x, num_channels//2, 'front_residual_x1' )
    x = MaxPool2D(pool_size = (2, 2), strides = (2, 2))(x)
    x = bottleneck(x, num_channels//2, 'front_residual_x2')
    x = bottleneck(x, num_channels, 'front_residual_x3')

    return x

#front_features = create_front_module(input, num_channels, bottleneck)

def create_left_half_blocks(bottom, bottleneck, hglayer, num_channels):
    """
    Create the half blocks for the hourglass module with 1/2, 1/4, 1/8 resolutions
    """
    hgname = 'hg' + str(hglayer)

    f1 = bottleneck(bottom, num_channels, hgname + '_l1')
    x = MaxPool2D(pool_size = (2, 2), strides = (2, 2))(f1)
    f2 = bottleneck(x, num_channels, hgname + '_l2')
    x = MaxPool2D(pool_size = (2, 2), strides = (2, 2))(f2)
    f4 = bottleneck(x, num_channels, hgname + '_l4')
    x = MaxPool2D(pool_size = (2, 2), strides = (2, 2))(f4)
    f8 = bottleneck(x, num_channels, hgname + '_l8')

    return f1, f2, f4, f8
    
# Create three filter blocks per resolution.

"""outputs = []
for i in range(2):
    head_next_stage, head_to_loss = create_left_half_blocks(front_features, num_classes, num_channels, bottleneck, i)
    outputs.append(head_to_loss)"""

def connect_left_to_right(left, right, bottleneck, name, num_channels):
    x_left = bottleneck(left, num_channels, name + '_connect')
    x_right = UpSampling2D()(right)
    add = Add()([x_left, x_right])
    out = bottleneck(add, num_channels, name + '_connect_conv')

    return out

def create_right_half_blocks(leftfeatures, bottleneck, hglayer, num_channels):
    """
    Apply the left to right bottleneck to each of the left features to get the right features.
    """
    lf1, lf2, lf4, lf8 = leftfeatures
    rf8 = bottom_layer(lf8, bottleneck, hglayer, num_channels)
    rf4 = connect_left_to_right(lf4, rf8, bottleneck, 'hg' + str(hglayer) + '_rf4', num_channels)
    rf2 = connect_left_to_right(lf2, rf4, bottleneck, 'hg' + str(hglayer) + '_rf2', num_channels)
    rf1 = connect_left_to_right(lf1, rf2, bottleneck, 'hg' + str(hglayer) + '_rf1', num_channels)
    
    return rf1

"""for i in range(2):
    head_next_stage, head_to_loss = create_right_half_blocks(front_features, num_classes, num_channels, bottleneck, i)
    outputs.append(head_to_loss)"""

def create_heads(prelayerfeatures, rf1, num_classes, hgid, num_channels):
    """
    Head to next stage + Head to intermediate features.
    """
    head = Conv2D(num_channels, kernel_size = (1, 1), activation = 'relu', padding = 'same', name = str(hgid) + '_conv1x1_x1')(rf1)
    head = BatchNormalization()(head)
    head_parts = Conv2D(num_classes, kernel_size = (1, 1), activation = 'linear', padding = 'same', name = str(hgid) + 'conv_1x1_parts')(head)
    head = Conv2D(num_channels, kernel_size = (1, 1), activation = 'linear', padding = 'same', name = str(hgid) + '_conv_1x1_x2')(head)
    head_m = Conv2D(num_channels, kernel_size = (1, 1), activation = 'linear', padding = 'same', name = str(hgid) + '_conv_1x1_x3')(head_parts)
    head_next_stage = Add()([head, head_m, prelayerfeatures])

    return head_next_stage, head_parts

"""for i in range(2):
    head_next_stage, head_to_loss = create_head_blocks(front_features, num_classes, num_channels, bottleneck, i)
    outputs.append(head_to_loss)"""

def hourglass_module(bottom, num_classes, num_channels, bottleneck, hgid):
    left_features = create_left_half_blocks(bottom, bottleneck, hgid, num_channels)

    rf1 = create_right_half_blocks(left_features, bottleneck, hgid, num_channels)

    head_next_stage, head_parts = create_heads(bottom, rf1, num_classes, hgid, num_channels)

    return head_next_stage, head_parts

def bottleneck_block(bottom, num_out_channels, block_name):

    # Skip layer
    if K.int_shape(bottom)[-1] == num_out_channels:
        skip = bottom
    else:
        skip = Conv2D(num_out_channels, kernel_size = (1, 1), activation = 'relu', padding = 'same', name = block_name + 'skip')(bottom)

    #Residual layer : 3 convolutional blocks, [n_channels_out/2, n_channels_out/2 n_channels_out]
    x = Conv2D(num_out_channels//2, kernel_size = (1, 1), activation = 'relu', padding = 'same', name = block_name + 'conv_1x1_x1')(bottom)
    x = BatchNormalization()(x)
    x = Conv2D(num_out_channels//2, kernel_size = (3, 3), activation = 'relu', padding = 'same', name = block_name + 'conv_3x3_x2')(x)
    x = BatchNormalization()(x)
    x = Conv2D(num_out_channels, kernel_size = (1, 1), activation = 'relu', padding = 'same', name = block_name + '_conv_1x1_x3')(x)
    x = BatchNormalization()(x)
    x = Add(name = block_name + '_residual')([skip, x])

    return x

def bottom_layer(lf8, bottleneck, hgid, num_channels):
    lf8_connect = bottleneck(lf8, num_channels, str(hgid) + '_lf8')
    x = bottleneck(lf8, num_channels, str(hgid) + '_lf8_x1')
    x = bottleneck(x, num_channels, str(hgid) + '_lf8_x2')          # Fixed issue #1
    x = bottleneck(x, num_channels, str(hgid) + '_lf8_x3')

    rf8 = Add()([x, lf8_connect])

    return rf8

def bottleneck_mobile(bottom, num_out_channels, block_name):

    # Skip layer
    if K.int_shape(bottom)[-1] == num_out_channels:
        skip = bottom
    else:
        skip = SeparableConv2D(num_out_channels, kernel_size = (1, 1), activation = 'relu', padding = 'same', name = block_name + 'skip')(bottom)

    # Residual layer : 3 convolutional blocks, [n_channels_out/2, n_channels_out/2 n_channels_out]
    x = SeparableConv2D(num_out_channls/2, kernel_size = (1, 1), activation = 'relu', padding = 'same', name = block_name + 'conv_1x1_x1')(bottom)
    x = BatchNormalization()(x)
    x = SeparableConv2D(num_out_channels/2, kernel_size = (3, 3), activation = 'relu', padding = 'same', name = block_name + 'conv_3x3_x2')(x)
    x = BatchNormalization()(x)
    x = SeparableConv2D(num_out_channels, kernel_size = (1, 1), activation = 'relu', padding = 'same', name = block_name + '_conv_1x1_x3')(x)
    x = BatchNormalization()(x)
    x = Add(name = block_name + '_residual')([skip, x])

    return x

def create_front_module(input, num_channels, bottleneck):
    # front module, input to 1/4 of the resolution.
    # 1 (7, 7) conv2D + maxpool + 3 residual layers.

    x = Conv2D(64, kernel_size = (7, 7), strides = (2, 2), padding = 'same', activation = 'relu', name = 'front_conv_1x1_x1')(input)
    x = BatchNormalization()(x)

    x = bottleneck(x, num_channels//2, 'front_residual_x1')
    x = MaxPool2D(pool_size = (2, 2), strides = (2, 2))(x)

    x = bottleneck(x, num_channels//2, 'front_residual_x2')
    x = bottleneck(x, num_channels, 'front_residual_x3')

    return x

def euclidean_loss(x, y):
    return K.sqrt(K.sum(K.square(x - y)))

def create_hourglass_network(num_classes, num_stacks, num_channels, inres, outres, bottleneck):
    input = Input(shape = (*inres[:2], 3))

    front_features = create_front_module(input, num_channels, bottleneck)

    head_next_stage = front_features

    outputs = []
    for i in range(num_stacks):
        head_next_stage, head_to_loss = hourglass_module(head_next_stage, num_classes, num_channels, bottleneck, i)
        outputs.append(head_to_loss)

    model = tf.keras.Model(inputs = input, outputs = outputs)
    rms = RMSprop(learning_rate = 5e-4)
    model.compile(optimizer = rms, loss = mean_squared_error, metrics = ['accuracy'])

    return model
