"""
Benchmark radar nowcast model.

U-Net model that predicts future frames of VIL from previous frames of VIL.

"""

import tensorflow as tf
from tensorflow.keras import layers,losses,models,regularizers
from tensorflow.keras.losses import mean_squared_error, mean_absolute_error

from src.nn.unet import conv_block,encoder_block,decoder_block

# Approximate mean, std of training data
MEAN=33.44
SCALE=47.54

def create_model(input_shape=(384,384,13),
                 start_neurons=32, 
	             num_outputs=12, 
	             activation='relu'):
    """

    Inputs are assumed to be in range [0-255].
    
    Shape [L,L,T], where T is number of time frames

    """
    inputs = layers.Input(shape=input_shape)
    
    # normalize inputs
    ir069 = tf.keras.layers.Input(shape=(192,192,1)) 
    ir107 = tf.keras.layers.Input(shape=(192,192,1)) 
    lght  = tf.keras.layers.Input(shape=(48,48,1))
    inputs = [ir069,ir107,lght]

    # Normalize inputs
    ir069_norm = tf.keras.layers.Lambda(lambda x,mu,scale: (x-mu)/scale,
                                        arguments={'mu':MEAN,'scale':SCALE})(inputs)

    encoder0_pool, encoder0 = encoder_block(inputs, start_neurons, activation=activation)
    encoder1_pool, encoder1 = encoder_block(encoder0_pool, start_neurons*2, activation=activation)
    encoder2_pool, encoder2 = encoder_block(encoder1_pool, start_neurons*4, activation=activation)
    encoder3_pool, encoder3 = encoder_block(encoder2_pool, start_neurons*8, activation=activation)
    center = conv_block(encoder3_pool, start_neurons*32)
    decoder3 = decoder_block(center, encoder3, start_neurons*8)
    decoder2 = decoder_block(decoder3, encoder2, start_neurons*4, activation=activation)
    decoder1 = decoder_block(decoder2, encoder1, start_neurons*2, activation=activation)
    decoder0 = decoder_block(decoder1, encoder0, start_neurons, activation=activation)
    outputs = layers.Conv2D(num_outputs, (1, 1), padding='same',
                                activation='linear', name='output_layer')(decoder0)
    
    return inputs, outputs


def nowcast_mse(y_true,y_pred):
	""" 
	MSE loss normalized by SCALE*SCALE
	"""
	return mean_squared_error(y_true,y_pred)/(SCALE*SCALE)


def nowcast_mae(y_true,y_pred):
	"""
	MAE normalized by SCALE
	"""
    retrun mean_absolute_error(y_true,y_pred)/(SCALE)



