import gc
import os
import time

from keras.utils import generic_utils
from keras.optimizers import Adam, SGD
import numpy as np
import models
import data_utils

dir_path = os.path.dirname(os.path.realpath(__file__))

"""
def train_cs_modis_predictor(**kwargs):
    scenes_fn = kwargs["scenes_fn"]
    weights_fn = kwargs["weights_fn"]

    print("Loading data...")
    (cs_scenes, modis_vars, modis_mask) = \
        data_utils.load_cloudsat_scenes(scenes_fn)

    scene_size = cs_scenes.shape[1]
    modis_var_dim = modis_vars.shape[-1]

    print("Creating model...")
    model = models.cs_modis_predictor()
    model.compile(loss=["mean_squared_error", "binary_crossentropy"], 
        optimizer="adam")

    print("Starting training...")
    for batch_size in [32, 64, 128, 256]:
        model.fit(cs_scenes, [modis_vars, modis_mask], epochs=10, 
            validation_split=0.1, batch_size=batch_size)

    print("Saving weights...")
    model.save_weights(weights_fn, overwrite=True)

    gc.collect()

    return model
"""

def model_state_paths(model_name, epoch, model_dir=None):
    if model_dir is None:
        model_dir = dir_path + "/../models/%s/" % model_name

    paths = {
        "gen_weights_path": os.path.join(
            model_dir + "/gen_weights_epoch%s.h5" % epoch),
        "disc_weights_path": os.path.join(
            model_dir + "/disc_weights_epoch%s.h5" % epoch),
        "opt_disc_weights_path": os.path.join(
            model_dir + "/opt_disc_weights_epoch%s.h5" % epoch),
        "opt_gan_weights_path": os.path.join(
            model_dir + "/opt_gan_weights_epoch%s.h5" % epoch)
    }
    return paths


def load_model_state(gen, disc, gan, model_name, epoch):
    paths = model_state_paths(model_name, epoch)
    gen.load_weights(paths["gen_weights_path"])
    disc.load_weights(paths["disc_weights_path"])
    disc.trainable = False
    gan._make_train_function()
    data_utils.load_opt_weights(gan, paths["opt_gan_weights_path"])    
    disc.trainable = True
    disc._make_train_function()
    data_utils.load_opt_weights(disc, paths["opt_disc_weights_path"])  


def save_model_state(gen, disc, gan, model_name, epoch):
    paths = model_state_paths(model_name, epoch)
    gen.save_weights(paths["gen_weights_path"], overwrite=True)
    disc.save_weights(paths["disc_weights_path"], overwrite=True)
    data_utils.save_opt_weights(disc, paths["opt_disc_weights_path"])
    data_utils.save_opt_weights(gan, paths["opt_gan_weights_path"])


def train_cs_modis_cgan(
        scenes_fn=None,
        noise_dim=64,
        #cont_dim=4,
        #modis_pred_weights=None,
        epoch=1,
        model_name="cs_modis_cgan",
        num_epochs=1,
        batch_size=32,
        noise_scale=1.0,
        cs_scenes=None,
        modis_vars=None,
        modis_mask=None,
        save_every=5,
        lr_disc=0.0001,
        lr_gan=0.0002,
    ):    

    # Load and rescale data
    if cs_scenes is None:
        print("Loading data...")
        (cs_scenes, modis_vars, modis_mask) = \
            data_utils.load_cloudsat_scenes(scenes_fn)
    num_scenes = cs_scenes.shape[0]
    batches_per_epoch = num_scenes // batch_size
    scene_size = cs_scenes.shape[1]
    modis_var_dim = modis_vars.shape[-1]

    print("Creating models...")
    # Create optimizers
    opt_disc = Adam(lr_disc, 0.5)
    opt_gan = Adam(lr_gan, 0.5)

    # Create models
    gen = models.cs_generator(scene_size, modis_var_dim, noise_dim)#, cont_dim)
    disc = models.discriminator(scene_size, modis_var_dim)#, cont_dim, mode='disc')
    #aux = models.discriminator(scene_size, cont_dim, mode='aux')
    #modis_pred = models.cs_modis_predictor(scene_size, modis_var_dim)
    #modis_pred.load_weights(modis_pred_weights)
    #modis_pred.trainable = False

    disc.trainable = False
    #aux.trainable = False

    gan = models.cs_modis_cgan(gen, disc, 
        #aux, modis_pred, 
        scene_size, modis_var_dim, noise_dim)#, cont_dim)

    gan_losses = ['binary_crossentropy']
    gan.compile(loss=gan_losses, optimizer=opt_gan)

    disc.trainable = True
    #aux.trainable = True
    disc.compile(loss='binary_crossentropy', optimizer=opt_disc)
    #aux.compile(loss='mse', optimizer=opt)

    if epoch > 1:
        print("Loading weights...")
        load_model_state(gen, disc, gan, model_name, epoch-1)

    # Start training
    print("Starting training...")
    for e in range(epoch,epoch+num_epochs):
        # Initialize progbar and batch counter
        progbar = generic_utils.Progbar(num_scenes)
        batch_counter = 1
        start = time.time()

        batch_gen = data_utils.gen_batch(cs_scenes, modis_vars, modis_mask, 
            batch_size)
        for (cs_scenes_b, modis_vars_b, modis_mask_b) in batch_gen:

            disc_loss = 0
            for fake in [False, True]:
                # Create a batch to feed the discriminator model
                (X_disc, y_disc) = data_utils.get_disc_batch(
                    cs_scenes_b, modis_vars_b, modis_mask_b, gen, fake, 
                    batch_size, noise_dim, 
                    #cont_dim, 
                    noise_scale=noise_scale)

                # Train the discriminator
                disc_loss += disc.train_on_batch(X_disc, y_disc)
            disc_loss /= 2

            # Create a batch to feed the generator model
            (X_gan, y_gan_disc) = data_utils.get_gan_batch(batch_size, 
                noise_dim, 
                #cont_dim, 
                noise_scale=noise_scale)
            noise = X_gan
            #(noise, cont) = X_gan

            # Freeze the discriminator while training the generator
            disc.trainable = False
            #aux.trainable = False
            gen_loss = gan.train_on_batch([noise, modis_vars_b, modis_mask_b],
                [y_gan_disc])
            disc.trainable = True
            #aux.trainable = True

            batch_counter += 1
            progbar.add(batch_size, values=[("D loss", disc_loss),
                ("G loss", gen_loss)])
                #("D aux", aux_loss),
                #("G tot", gen_loss[0]), ("G disc", gen_loss[1]),
                #("G aux", gen_loss[2]),
                #("G vars", gen_loss[3]), ("G mask", gen_loss[4])])

            if batch_counter % 50 == 0:
                #scene_gen = gen.predict([modis_vars_3d_b, modis_mask_3d_b,
                #    noise, cont])
                scene_gen = gen.predict([noise, modis_vars_b, modis_mask_b])
                modis_vars_bs = np.zeros_like(modis_vars_b)
                modis_mask_bs = np.zeros_like(modis_mask_b)
                for i in range(1,batch_size):
                    modis_vars_bs[i,...] = modis_vars_bs[0,...]
                    modis_mask_bs[i,...] = modis_mask_bs[0,...]
                scene_gen_s = gen.predict([noise, modis_vars_bs, modis_mask_bs])

                np.savez_compressed(dir_path+"/../data/gen_scene.npz",
                    scene=scene_gen, real_scene=cs_scenes_b,
                    scene_single=scene_gen_s,
                    modis_vars=modis_vars_b, modis_mask=modis_mask_b)

        # Save images for visualization
        # data_utils.plot_generated_batch(X_real_batch, gen,
        #    batch_size, cat_dim, cont_dim, noise_dim, image_data_format)

        gc.collect()

        print("")
        print('Epoch %s/%s, Time: %s' % (e, epoch+num_epochs-1, time.time() - start))

        if (e % save_every == 0):
            print("Saving weights...")
            save_model_state(gen, disc, gan, model_name, e)

    return (gan, gen, disc)


def train_cs_modis_cgan_full(scenes_fn):
    print("Loading data...")
    (cs_scenes, modis_vars, modis_mask) = \
        data_utils.load_cloudsat_scenes(scenes_fn)
    train_kwargs = {
        "cs_scenes": cs_scenes,
        "modis_vars": modis_vars,
        "modis_mask": modis_mask,
        "noise_dim": 64,
        "cont_dim": 4,
        "modis_pred_weights": dir_path + \
            "/../../models/cs_modis_pred/cs_modis_pred_weights.h5",
        "save_every": 1
    }

    train_cs_modis_gan(num_epochs=5, epoch=1, batch_size=32,
        **train_kwargs)
    train_cs_modis_gan(num_epochs=5, epoch=6, batch_size=64,
        **train_kwargs)
    train_cs_modis_gan(num_epochs=10, epoch=11, batch_size=128,
        **train_kwargs)
    train_cs_modis_gan(num_epochs=20, epoch=21, batch_size=256,
        **train_kwargs)


