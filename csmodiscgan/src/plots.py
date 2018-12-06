import os
import matplotlib
matplotlib.use("Agg")
from matplotlib import cm, colors, gridspec, pyplot as plt
import numpy as np
import data_utils
import train

dir_path = os.path.dirname(os.path.realpath(__file__))

vir_white = colors.ListedColormap([[1.,1.,1.]]+cm.viridis.colors)


def plot_scene(ax, img, pix_extent, scene_size):
    (pix_height, pix_width) = pix_extent
    norm = colors.Normalize(vmin=-35, vmax=20)
    ax.imshow(img, interpolation='nearest', aspect='auto',
        extent=[0, pix_width*scene_size, 0, pix_height*scene_size],
        cmap=vir_white, norm=norm)


def plot_column(ax, y1, y2, pix_extent, scene_size, c1='r', c2='b',
    range1=None, range2=None, inv1=False, inv2=False, 
    log1=False, log2=False):

    pix_width = pix_extent[1]
    x = (np.arange(scene_size)+0.5)*pix_width
    ax.plot(x, y1, color=c1)
    ax.set_xlim((0,scene_size*pix_width))
    ax.set_ylim(*range1)
    ax.tick_params(colors=c1)
    if inv1:
        ax.invert_yaxis()
    if log1:
        ax.set_yscale('log')
    ax2 = ax.twinx()
    ax2.plot(x, y2, color=c2)
    ax2.set_xlim((0,scene_size*pix_width))
    ax2.set_ylim(*range2)
    ax2.tick_params(colors=c2)
    if inv2:
        ax2.invert_yaxis()
    if log2:
        ax2.set_yscale('log')
    return ax2


def generate_scenes(gen, modis_vars, modis_mask, noise_dim=64, rng_seed=None):
    batch_size = modis_vars.shape[0]
    prng = np.random.RandomState(rng_seed)
    noise = prng.normal(scale=2.0, size=(batch_size, noise_dim))
    scene_gen = gen.predict([noise, modis_vars, modis_mask])
    return scene_gen


def plot_samples_cmp(gen, scene_real, modis_vars, modis_mask, noise_dim=64,
    num_gen=4, rng_seeds=[762149,20183,417662,783924], 
    pix_extent=(0.24,1.09)):

    scene_gen = [None]*num_gen
    for k in range(num_gen):
        scene_gen[k] = generate_scenes(gen, modis_vars, modis_mask, 
            noise_dim=noise_dim, rng_seed=rng_seeds[k])
        scene_gen[k] = data_utils.rescale_scene(scene_gen[k])
    scene_real = data_utils.rescale_scene(scene_real)

    num_samples = scene_real.shape[0]
    modis_vars_real = data_utils.decode_modis_vars(modis_vars, modis_mask)
    scene_size = scene_real.shape[1]
    tau_c = modis_vars_real["twp"][:,:]
    tau_c_range = (0, np.nanmax(tau_c)*1.05)
    p_top = modis_vars_real["p_top"][:,:]
    p_top_range = (100, 1024)

    gs = gridspec.GridSpec(num_gen+2,num_samples,
        height_ratios=(1,)+(2,)*(num_gen+1),hspace=0.1,wspace=0.1)
    fig = plt.figure(figsize=(num_samples*1.5, 1.5+num_gen*1.6))

    for i in range(num_samples):
        ax_tau = plt.subplot(gs[0,i])
        ax_p = plot_column(ax_tau, tau_c[i,:], p_top[i,:], 
            pix_extent, scene_size, range1=tau_c_range, 
            range2=p_top_range, inv2=True, log2=True)
        ax_tau.tick_params(labelbottom=False, labelleft=(i==0))
        if i==0:
            ax_tau.set_ylabel("$\\tau_c$", color='r')
        ax_p.tick_params(labelbottom=False, labelright=(i==num_samples-1))
        if i == num_samples-1:
            ax_p.set_ylabel("$P_\\mathrm{top}$ [hPa]", color='b')

        for k in range(num_gen):
            ax_gen = plt.subplot(gs[1+k,i])
            plot_scene(ax_gen, scene_gen[k][i,:,:,0], pix_extent, scene_size)
            ax_gen.tick_params(labelbottom=False, labelleft=(i==0))
            if i==0:
                ax_gen.set_ylabel("Generated\nAltitude [km]")

        ax_real = plt.subplot(gs[-1,i])
        plot_scene(ax_real, scene_real[i,:,:,0], pix_extent, scene_size)
        ax_real.tick_params(labelleft=(i==0))
        if i==0:
            ax_real.set_ylabel("Real\nAltitude [km]")
        ax_real.set_xlabel("Distance [km]")

    return fig


def load_data_and_models(scenes_fn, model_name="cs_modis_cgan-reference", 
    epoch=40, scene_size=64, modis_var_dim=4, noise_dim=64, 
    lr_disc=0.0001, lr_gan=0.0002):

    scenes = data_utils.load_cloudsat_scenes(scenes_fn, shuffle_seed=214101)

    (gen, disc, gan, opt_disc, opt_gan) = train.create_models(
        scene_size, modis_var_dim, noise_dim, lr_disc, lr_gan)

    train.load_model_state(gen, disc, gan, model_name, epoch)

    return (scenes, gen, disc, gan)
