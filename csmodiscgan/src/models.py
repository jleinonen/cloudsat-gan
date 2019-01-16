from keras.models import Model
from keras.layers import Activation, Dense, Flatten, Input, LeakyReLU, Reshape
from keras.layers import Conv2D, UpSampling2D
from keras.layers import BatchNormalization


def cs_generator(scene_size, modis_var_dim, noise_dim):
    f = 256
    start_dim = 8
    reshape_shape = (start_dim, start_dim, f)

    modis_var_input = Input(shape=(scene_size,modis_var_dim), name="modis_var_in")
    modis_mask_input = Input(shape=(scene_size,1), name="modis_mask_in")
    noise_input = Input(shape=(noise_dim,), name="noise_in")

    inputs = [noise_input, modis_var_input, modis_mask_input]
    inputs_flat = [inputs[0], Flatten()(inputs[1]), Flatten()(inputs[2])]
    gen_input = Concatenate()(inputs_flat)

    x = Dense(f * start_dim * start_dim)(gen_input)
    x = Activation("relu")(x)
    x = BatchNormalization(momentum=0.8)(x)

    x = Reshape(reshape_shape)(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(256, (3, 3), padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization(momentum=0.8)(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(128, (3, 3), padding="same")(x)    
    x = Activation("relu")(x)
    x = BatchNormalization(momentum=0.8)(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(64, (3, 3), padding="same")(x)    
    x = Activation("relu")(x)
    x = BatchNormalization(momentum=0.8)(x)

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
    x = BatchNormalization(momentum=0.8)(x)

    x = UpSampling2D(size=(4,1))(x)
    x = Conv2D(128, (5, 3), padding="same")(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)

    x = UpSampling2D(size=(2,1))(x)
    x = Conv2D(64, (3, 3), padding="same")(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)

    x = UpSampling2D(size=(2,1))(x)
    x = Conv2D(upsampled_channels, (3, 3), padding="same")(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)

    return x


def discriminator(scene_size, modis_var_dim):
    disc_input = Input(shape=(scene_size,scene_size,1), name="disc_in")
    modis_var_input = Input(shape=(scene_size,modis_var_dim), name="modis_var_in")
    modis_mask_input = Input(shape=(scene_size,1), name="modis_mask_in")

    modis_upsampled = modis_upsampler(modis_var_input, modis_mask_input,
        modis_var_dim, scene_size)

    full_input = Concatenate()([disc_input, modis_upsampled])

    x = Conv2D(64, (3, 3), strides=(2, 2), padding="same")(full_input)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(128, (3, 3), strides=(2, 2), padding="same")(x)    
    x = LeakyReLU(0.2)(x)

    x = Conv2D(256, (3, 3), strides=(2, 2), padding="same")(x)    
    x = LeakyReLU(0.2)(x)

    x = Conv2D(256, (3, 3), strides=(2, 2), padding="same")(x)
    x = LeakyReLU(0.2)(x) 

    x = Flatten()(x)

    x_disc = Dense(1, activation='sigmoid', name="disc_out")(x)
    model = Model(inputs=[disc_input, modis_var_input, modis_mask_input], 
        outputs=x_disc,
        name="disc")

    return model


def cs_modis_cgan(gen, disc, scene_size, modis_var_dim, noise_dim): 
    modis_var_input = Input(shape=(scene_size,modis_var_dim), 
        name="modis_var_in")
    modis_mask_input = Input(shape=(scene_size,1), name="modis_mask_in")
    noise_input = Input(shape=(noise_dim,), name="noise_in")
    inputs = [noise_input, modis_var_input, modis_mask_input]

    generated_image = gen(inputs)
    disc_inputs = [generated_image, modis_var_input, modis_mask_input]
    x_disc = disc(disc_inputs)

    gan = Model(inputs=inputs, outputs=x_disc, name="cs_modis_cgan")

    return gan



