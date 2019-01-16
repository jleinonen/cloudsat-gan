import gc
import os
import time

from keras.utils import generic_utils
from keras.optimizers import Adam
import numpy as np
import models
import data_utils


dir_path = os.path.dirname(os.path.realpath(__file__))


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


def create_models(scene_size, modis_var_dim, noise_dim, lr_disc, lr_gan):
    # Create optimizers
    opt_disc = Adam(lr_disc, 0.5)
    opt_gan = Adam(lr_gan, 0.5)

    # Create models
    gen = models.cs_generator(scene_size, modis_var_dim, noise_dim)
    disc = models.discriminator(scene_size, modis_var_dim)

    # Compile models
    disc.trainable = False
    gan = models.cs_modis_cgan(gen, disc, scene_size, modis_var_dim, 
        noise_dim)
    gan.compile(loss='binary_crossentropy', optimizer=opt_gan)
    disc.trainable = True    
    disc.compile(loss='binary_crossentropy', optimizer=opt_disc)

    return (gen, disc, gan, opt_disc, opt_gan)


def train_cs_modis_cgan(
        scenes_fn=None,
        noise_dim=64,
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
    (gen, disc, gan, opt_disc, opt_gan) = create_models(
        scene_size, modis_var_dim, noise_dim, lr_disc, lr_gan)

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
            gen_loss = gan.train_on_batch([noise, modis_vars_b, modis_mask_b],
                [y_gan_disc])
            disc.trainable = True

            batch_counter += 1
            progbar.add(batch_size, values=[("D loss", disc_loss),
                ("G loss", gen_loss)])

            if batch_counter % 50 == 0:
                scene_gen = gen.predict([noise, modis_vars_b, modis_mask_b])
                modis_vars_bs = np.zeros_like(modis_vars_b)
                modis_mask_bs = np.zeros_like(modis_mask_b)
                for i in range(1,batch_size):
                    modis_vars_bs[i,...] = modis_vars_b[0,...]
                    modis_mask_bs[i,...] = modis_mask_b[0,...]
                scene_gen_s = gen.predict([noise, modis_vars_bs, modis_mask_bs])

                np.savez_compressed(dir_path+"/../data/gen_scene.npz",
                    scene=scene_gen, real_scene=cs_scenes_b,
                    scene_single=scene_gen_s,
                    modis_vars=modis_vars_b, modis_mask=modis_mask_b)

        gc.collect()

        print("")
        print('Epoch %s/%s, Time: %s' % (e, epoch+num_epochs-1, time.time() - start))

        if (e % save_every == 0):
            print("Saving weights...")
            save_model_state(gen, disc, gan, model_name, e)

    return (gan, gen, disc)


def train_cs_modis_cgan_full(scenes_fn, run_name=None):
    print("Loading data...")
    scenes = data_utils.load_cloudsat_scenes(scenes_fn,
        shuffle_seed=214101)
    (cs_scenes, modis_vars, modis_mask) = scenes["train"]
    
    model_name = "cs_modis_cgan"
    if run_name:
        model_name += "-"+run_name

    paths = model_state_paths(model_name, 1)
    model_dir = os.path.dirname(paths["gen_weights_path"])
    try:
        os.mkdir(model_dir)
    except OSError:
        pass

    train_kwargs = {
        "model_name": model_name,
        "cs_scenes": cs_scenes,
        "modis_vars": modis_vars,
        "modis_mask": modis_mask,
        "noise_dim": 64,
        "save_every": 1
    }

    train_cs_modis_cgan(num_epochs=5, epoch=1, batch_size=32,
        **train_kwargs)
    train_cs_modis_cgan(num_epochs=10, epoch=6, batch_size=64,
        **train_kwargs)
    train_cs_modis_cgan(num_epochs=10, epoch=16, batch_size=128,
        **train_kwargs)
    train_cs_modis_cgan(num_epochs=20, epoch=26, batch_size=256,
        **train_kwargs)
