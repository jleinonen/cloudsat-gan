import gc
import numpy as np
import h5py
import keras.backend as K
import netCDF4


def load_cloudsat_scenes(fn, n=None, right_handed=False, frac_validate=0.1,
    shuffle=True, shuffle_seed=None):

    with netCDF4.Dataset(fn, 'r') as ds:
        if n is None:
            n = ds["scenes"].shape[0]
        cs_scenes = ds["scenes"][:n,:,:].data
        cs_scenes = cs_scenes.reshape(cs_scenes.shape+(1,))
        if right_handed:
            cs_scenes = np.rot90(cs_scenes, axes=(2,1))
        # rescale from (0,1) to (-1,1)
        cs_scenes *= 2
        cs_scenes -= 1
        modis_vars = np.zeros((n,)+ds["tau_c"].shape[1:]+(4,), 
            dtype=np.float32)
        modis_vars[:,:,0] = ds["tau_c"][:n,:]
        modis_vars[:,:,1] = ds["p_top"][:n,:]
        modis_vars[:,:,2] = ds["r_e"][:n,:]
        modis_vars[:,:,3] = ds["twp"][:n,:]
        modis_mask = np.zeros((n,)+ds["tau_c"].shape[1:]+(1,), 
            dtype=np.float32)
        modis_mask[:,:,0] = ds["modis_mask"][:n,:]

    num_scenes = cs_scenes.shape[0]
    if shuffle:
        prng = np.random.RandomState(shuffle_seed)
        ind = np.arange(num_scenes)
        prng.shuffle(ind)
        cs_scenes = cs_scenes[ind,...]
        modis_vars = modis_vars[ind,...]
        modis_mask = modis_mask[ind,...]

    gc.collect()

    num_train = int(round(num_scenes*(1.0-frac_validate)))

    scenes = {
        "train": (
            cs_scenes[:num_train,...], 
            modis_vars[:num_train,...], 
            modis_mask[:num_train,...]
        ),
        "validate": (
            cs_scenes[num_train:,...], 
            modis_vars[num_train:,...], 
            modis_mask[num_train:,...]
        )
    }

    return scenes


def decode_modis_vars(modis_vars, modis_mask):
    tau_c_scaled = modis_vars[:,:,0]
    p_top_scaled = modis_vars[:,:,1]
    r_e_scaled = modis_vars[:,:,2]
    twp_scaled = modis_vars[:,:,3]

    decoded_vars = {}
    decoded_vars["tau_c"] = np.exp((1.13*tau_c_scaled+2.20))    
    decoded_vars["p_top"] = 265.0*p_top_scaled+532.0    
    decoded_vars["r_e"] = np.exp((0.542*r_e_scaled+3.06))
    decoded_vars["twp"] = np.exp((1.11*twp_scaled+0.184))
    for var in decoded_vars:
        decoded_vars[var][~modis_mask[:,:,0].astype(bool)] = np.nan

    return decoded_vars


def rescale_scene(scene, Z_range=(-35,20), missing_max=-30):
    sc = Z_range[0] + (scene+1)/2.0 * (Z_range[1]-Z_range[0])
    sc[sc <= missing_max] = np.nan
    return sc


def gen_batch(cs_scenes, modis_vars, modis_mask, batch_size):
    ind = np.arange(cs_scenes.shape[0], dtype=int)
    np.random.shuffle(ind)
    while len(ind) >= batch_size:      
        idx = ind[:batch_size]
        ind = ind[batch_size:]
        yield (cs_scenes[idx,...], modis_vars[idx,...], modis_mask[idx,...])


def gen_modis_batch_2d(modis_vars_2d, modis_mask_2d, batch_size):
    ind = np.arange(modis_vars_2d.shape[0], dtype=int)
    np.random.shuffle(ind)
    while len(ind) >= batch_size:      
        idx = ind[:batch_size]
        ind = ind[batch_size:]
        (modis_vars_2d_b, modis_mask_2d_b) = (
            modis_vars_2d[idx,...], modis_mask_2d[idx,...])        
        yield (modis_vars_2d_b, modis_mask_2d_b)


def expand_modis_batch(modis_vars_2d_b, modis_mask_2d_b, scene_size):
    vars_shape = modis_vars_2d_b.shape[:3]+(scene_size,modis_vars_2d_b.shape[-1])
    mask_shape = modis_mask_2d_b.shape[:3]+(scene_size,modis_mask_2d_b.shape[-1])
    modis_vars_3d_b = np.empty(vars_shape, dtype=np.float32)
    modis_mask_3d_b = np.empty(mask_shape, dtype=np.float32)
    for i in range(scene_size):
        modis_vars_3d_b[:,:,:,i,:] = modis_vars_2d_b
        modis_mask_3d_b[:,:,:,i,:] = modis_mask_2d_b
    return (modis_vars_3d_b, modis_mask_3d_b)


def sample_noise(noise_scale, batch_size, noise_dim):
    return np.random.normal(scale=noise_scale, size=(batch_size, noise_dim))


def get_disc_batch(cs_scenes_b, modis_vars_b, modis_mask_b, gen, fake, 
    batch_size, noise_dim, 
    #cont_dim, 
    noise_scale=1.0, max_smoothing=0.1):

    # Create X_disc: alternatively only generated or real images
    #if fake: # generate fake samples
    
    if fake:
        # Pass noise to the generator
        noise = sample_noise(noise_scale, batch_size, noise_dim)
        #cont = sample_noise(noise_scale, batch_size, cont_dim)
        X_disc = [
            gen.predict([noise, modis_vars_b, modis_mask_b]),
            modis_vars_b,
            modis_mask_b
        ]
        # label smoothing
        y_disc = 1-max_smoothing*np.random.rand(batch_size, 1)
    else:
        X_disc = [cs_scenes_b, modis_vars_b, modis_mask_b]
        y_disc = max_smoothing*np.random.rand(batch_size, 1)

    return (X_disc, y_disc)


def get_gan_batch(batch_size, noise_dim, 
    #cont_dim, 
    noise_scale=1.0, max_smoothing=0.1, num_disc=1):

    noise = sample_noise(noise_scale, batch_size, noise_dim)
    X_gan = noise
    y_gan_disc = max_smoothing*np.random.rand(batch_size, num_disc)

    return (X_gan, y_gan_disc)


def save_opt_weights(model, filepath):
    with h5py.File(filepath, 'w') as f:
        # Save optimizer weights.
        symbolic_weights = getattr(model.optimizer, 'weights')
        if symbolic_weights:
            optimizer_weights_group = f.create_group('optimizer_weights')
            weight_values = K.batch_get_value(symbolic_weights)
            weight_names = []
            for i, (w, val) in enumerate(zip(symbolic_weights, weight_values)):
                # Default values of symbolic_weights is /variable for theano
                if K.backend() == 'theano':
                    if hasattr(w, 'name') and w.name != "/variable":
                        name = str(w.name)
                    else:
                        name = 'param_' + str(i)
                else:
                    if hasattr(w, 'name') and w.name:
                        name = str(w.name)
                    else:
                        name = 'param_' + str(i)
                weight_names.append(name.encode('utf8'))
            optimizer_weights_group.attrs['weight_names'] = weight_names
            for name, val in zip(weight_names, weight_values):
                param_dset = optimizer_weights_group.create_dataset(
                    name,
                    val.shape,
                    dtype=val.dtype)
                if not val.shape:
                    # scalar
                    param_dset[()] = val
                else:
                    param_dset[:] = val


def load_opt_weights(model, filepath):
    with h5py.File(filepath, mode='r') as f:        
        optimizer_weights_group = f['optimizer_weights']
        optimizer_weight_names = [n.decode('utf8') for n in
                                  optimizer_weights_group.attrs['weight_names']]
        optimizer_weight_values = [optimizer_weights_group[n] for n in
                                   optimizer_weight_names]
        model.optimizer.set_weights(optimizer_weight_values)

