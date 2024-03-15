from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Cropping2D

def conv_block(input_tensor, num_filters):
    encoder = Conv2D(num_filters, (3, 3), padding='same', activation='relu')(input_tensor)
    encoder = Conv2D(num_filters, (3, 3), padding='same', activation='relu')(encoder)
    return encoder

def encoder_block(input_tensor, num_filters):
    encoder = conv_block(input_tensor, num_filters)
    pool = MaxPooling2D((2, 2), strides=(2, 2))(encoder)
    return encoder, pool

def decoder_block(input_tensor, concat_tensor, num_filters):
    decoder = Conv2D(num_filters, (3, 3), padding='same', activation='relu')(input_tensor)
    decoder = Conv2D(num_filters, (3, 3), padding='same', activation='relu')(decoder)
    decoder = UpSampling2D((2, 2))(decoder)
    decoder = concatenate([decoder, concat_tensor], axis=-1)
    return decoder

def build_unet(input_shape):
    inputs = Input(input_shape)
    
    encoder0, pool0 = encoder_block(inputs, 64)
    encoder1, pool1 = encoder_block(pool0, 128)
    encoder2, pool2 = encoder_block(pool1, 256)
    encoder3, pool3 = encoder_block(pool2, 512)
    encoder4, pool4 = encoder_block(pool3, 1024)
    
    center = conv_block(pool4, 1024)
    
    decoder4 = decoder_block(center, encoder4, 1024)
    decoder3 = decoder_block(decoder4, encoder3, 512)
    decoder2 = decoder_block(decoder3, encoder2, 256)
    decoder1 = decoder_block(decoder2, encoder1, 128)
    decoder0 = decoder_block(decoder1, encoder0, 64)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(decoder0) 
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

