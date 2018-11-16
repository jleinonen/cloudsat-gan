from keras import regularizers
from keras.models import Model
from keras.layers.core import Flatten, Dense, Dropout, Activation, Reshape, Lambda
from keras.layers.convolutional import Conv2D, UpSampling2D, Conv3D, UpSampling3D
from keras.layers import Input, Concatenate, average
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras_contrib.layers.normalization import InstanceNormalization
from normgan import BatchRenormalization
import keras.backend as K

"""
def cs_modis_predictor(scene_size=64, modis_var_dim=4):
    p_input = Input(shape=(scene_size,scene_size,1))

    x = Conv2D(64, (3, 5), strides=(2, 1), padding="same")(p_input)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(64, (3, 3), strides=(2, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(64, (5, 1), strides=(4, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(64, (5, 1), strides=(4, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x_f = Conv2D(128, (1, 1), padding="same")(x)
    x_f = BatchNormalization()(x_f)
    x_f = LeakyReLU(0.2)(x_f)

    y_m = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(x_f)
    y_m = Reshape((scene_size, 1), name="modis_mask")(y_m)

    y_c = Conv2D(modis_var_dim, (1, 1), padding="same")(x_f)
    y_c = Reshape((scene_size, modis_var_dim), name="modis_vars")(y_c)

    model = Model(inputs=p_input, outputs=[y_c, y_m])

    return model
"""

def cs_generator(scene_size, modis_var_dim, noise_dim): #, cont_dim):
    f = 256
    start_dim = 8
    reshape_shape = (start_dim, start_dim, f)

    modis_var_input = Input(shape=(scene_size,modis_var_dim), name="modis_var_in")
    modis_mask_input = Input(shape=(scene_size,1), name="modis_mask_in")
    noise_input = Input(shape=(noise_dim,), name="noise_in")
    #cont_input = Input(shape=(cont_dim,), name="cont_input")

    inputs = [noise_input, modis_var_input, modis_mask_input]
    inputs_flat = [inputs[0], Flatten()(inputs[1]), Flatten()(inputs[2])]
    gen_input = Concatenate()(inputs_flat)

    x = Dense(f * start_dim * start_dim)(gen_input)
    x = Activation("relu")(x)
    x = BatchRenormalization(momentum=0.8)(x)

    x = Reshape(reshape_shape)(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(256, (3, 3), padding="same")(x)    
    x = Activation("relu")(x)
    x = BatchRenormalization(momentum=0.8)(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(128, (3, 3), padding="same")(x)    
    x = Activation("relu")(x)
    x = BatchRenormalization(momentum=0.8)(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(64, (3, 3), padding="same")(x)    
    x = Activation("relu")(x)
    x = BatchRenormalization(momentum=0.8)(x)

    x = Conv2D(1, (3, 3), padding="same", 
        activation='tanh')(x)

    gen = Model(inputs=inputs, outputs=x, name="gen")

    return gen


def modis_upsampler(modis_var_input, modis_mask_input, 
    modis_var_dim, scene_size, upsampled_channels=None):

    if upsampled_channels is None:
        upsampled_channels = modis_var_dim+1

    modis_input = Concatenate()([modis_var_input, modis_mask_input])
    x = Reshape((1,scene_size,modis_var_dim+1))(modis_input)    

    x = UpSampling2D(size=(4,1))(x)
    x = Conv2D(256, (5, 3), padding="same")(x)
    x = LeakyReLU(0.2)(x)
    x = BatchRenormalization(momentum=0.8)(x)

    x = UpSampling2D(size=(4,1))(x)
    x = Conv2D(128, (5, 3), padding="same")(x)
    x = LeakyReLU(0.2)(x)
    x = BatchRenormalization(momentum=0.8)(x)

    x = UpSampling2D(size=(2,1))(x)
    x = Conv2D(64, (3, 3), padding="same")(x)
    x = LeakyReLU(0.2)(x)
    x = BatchRenormalization(momentum=0.8)(x)

    x = UpSampling2D(size=(2,1))(x)
    x = Conv2D(upsampled_channels, (3, 3), padding="same")(x)
    x = LeakyReLU(0.2)(x)
    x = BatchRenormalization(momentum=0.8)(x)

    return x


#def discriminator(scene_size, cont_dim, mode="disc"):
def discriminator(scene_size, modis_var_dim): #, cont_dim):
    disc_input = Input(shape=(scene_size,scene_size,1), name="disc_in")
    modis_var_input = Input(shape=(scene_size,modis_var_dim), name="modis_var_in")
    modis_mask_input = Input(shape=(scene_size,1), name="modis_mask_in")

    modis_upsampled = modis_upsampler(modis_var_input, modis_mask_input,
        modis_var_dim, scene_size)

    full_input = Concatenate()([disc_input, modis_upsampled])

    x = Conv2D(64, (3, 3), strides=(2, 2), padding="same")(full_input)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.25)(x)

    x = Conv2D(128, (3, 3), strides=(2, 2), padding="same")(x)    
    x = LeakyReLU(0.2)(x)
    #x = BatchNormalization(momentum=0.8)(x)
    x = Dropout(0.25)(x)

    x = Conv2D(256, (3, 3), strides=(2, 2), padding="same")(x)    
    x = LeakyReLU(0.2)(x)
    #x = BatchNormalization(momentum=0.8)(x)
    x = Dropout(0.25)(x)

    x = Conv2D(256, (3, 3), strides=(2, 2), padding="same")(x)
    x = LeakyReLU(0.2)(x) 
    #x = BatchNormalization(momentum=0.8)(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)

    #x = Dense(1024)(x)    
    #x = LeakyReLU(0.2)(x)
    #x = BatchNormalization(momentum=0.8)(x)

    #if mode=="disc":
        # Create discriminator model
    x_disc = Dense(1, activation='sigmoid', name="disc_out")(x)
    model = Model(inputs=[disc_input, modis_var_input, modis_mask_input], 
        outputs=x_disc,
        name="disc")
    """
    elif mode=="aux":
        x_Q = Dense(128)(x)        
        x_Q = LeakyReLU(0.2)(x_Q)
        #x_Q = BatchNormalization(momentum=0.8)(x_Q)
        #x_Q_Y = Dense(cat_dim[0], activation='softmax', name="Q_cat_out")(x_Q)
        x_Q_C = Dense(cont_dim, activation='linear', name="aux_out")(x_Q)
        model = Model(inputs=disc_input, outputs=x_Q_C,
            name="aux")
    """

    return model


def cs_modis_cgan(gen, disc, 
    #aux, modis_pred, 
    scene_size, modis_var_dim, noise_dim): #, cont_dim):

    modis_var_input = Input(shape=(scene_size,modis_var_dim), 
        name="modis_var_in")
    modis_mask_input = Input(shape=(scene_size,1), name="modis_mask_in")
    noise_input = Input(shape=(noise_dim,), name="noise_in")
    #cont_input = Input(shape=(cont_dim,), name="cont_input")
    inputs = [noise_input, modis_var_input, modis_mask_input]#, cont_input]

    generated_image = gen(inputs)
    disc_inputs = [generated_image, modis_var_input, modis_mask_input]
    x_disc = disc(disc_inputs)
    #x_aux = aux(generated_image)
    #(modis_var_pred, modis_mask_pred) = modis_pred(generated_image)

    gan = Model(inputs=inputs,
        outputs=x_disc,
        #outputs=[x_disc, modis_var_pred, modis_mask_pred],
        name="cs_modis_cgan")

    return gan



