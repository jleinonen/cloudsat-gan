import os
import matplotlib
matplotlib.use("Agg")
from matplotlib import cm, colors, colorbar
from matplotlib import gridspec, pyplot as plt
import numpy as np
import data_utils
import train

dir_path = os.path.dirname(os.path.realpath(__file__))

vir_white = colors.ListedColormap([[1.,1.,1.]]+cm.viridis.colors)

modis_colors = {
    "tau_c": "#e41a1c",
    "p_top": "#377eb8",
    "twp": "#4daf4a",
    "r_e": "#984ea3"
}

dBZ_norm = colors.Normalize(vmin=-35, vmax=20)

def plot_scene(ax, img, pix_extent=(0.24,1.09), scene_size=64):
    (pix_height, pix_width) = pix_extent
    ax.imshow(img, interpolation='nearest', aspect='auto',
        extent=[0, pix_width*scene_size, 0, pix_height*scene_size],
        cmap=vir_white, norm=dBZ_norm)
    ax.set_xticks([0,20,40,60])
    ax.set_yticks([0,4,8,12])


def plot_column(ax, y1, y2, pix_extent=(0.24,1.09), 
    scene_size=64, c1='r', c2='b',
    range1=None, range2=None, inv1=False, inv2=False, 
    ticks1=None, ticks2=None,
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
    if ticks1 is not None:
        ax.set_yticks(ticks1[0])
        ax.set_yticklabels(ticks1[1])
    ax.set_xticks([0,20,40,60])

    ax2 = ax.twinx()
    ax2.plot(x, y2, color=c2)
    ax2.set_xlim((0,scene_size*pix_width))
    ax2.set_ylim(*range2)
    ax2.tick_params(colors=c2)
    if inv2:
        ax2.invert_yaxis()
    if log2:
        ax2.set_yscale('log')
    if ticks2 is not None:
        ax2.set_yticks(ticks2[0])
        ax2.set_yticklabels(ticks2[1])
    return ax2


def add_dBZ_colorbar(fig, pos):
    cax = fig.add_axes(pos)
    colorbar.ColorbarBase(cax, norm=dBZ_norm,
        cmap=vir_white)
    cax.set_ylabel("Reflectivity [dBZ]")
    cax.yaxis.set_label_position("right")
    return cax


def generate_scenes(gen, modis_vars, modis_mask, noise_dim=64, rng_seed=None,
    zero_noise=False, noise_scale=1.0):

    batch_size = modis_vars.shape[0]
    if zero_noise:
        noise = np.zeros((batch_size, noise_dim), dtype=np.float32)
    else:
        prng = np.random.RandomState(rng_seed)
        noise = prng.normal(scale=noise_scale, size=(batch_size, noise_dim))
    scene_gen = gen.predict([noise, modis_vars, modis_mask])
    return scene_gen


def plot_samples_cmp(gen, scene_real, modis_vars, modis_mask, noise_dim=64,
    num_gen=4, rng_seeds=[20183,417662,783924], 
    pix_extent=(0.24,1.09), first_column_num=1):

    scene_gen = [None]*num_gen
    for k in range(num_gen):
        scene_gen[k] = generate_scenes(gen, modis_vars, modis_mask, 
            noise_dim=noise_dim, 
            rng_seed=(rng_seeds[k-1] if k>0 else 0),
            zero_noise=(k==0))
        scene_gen[k] = data_utils.rescale_scene(scene_gen[k])
    scene_real = data_utils.rescale_scene(scene_real)

    num_samples = scene_real.shape[0]
    modis_vars_real = data_utils.decode_modis_vars(modis_vars, modis_mask)
    scene_size = scene_real.shape[1]
    tau_c = modis_vars_real["tau_c"][:,:]
    tau_c_range = (1, 150)
    p_top = modis_vars_real["p_top"][:,:]
    p_top_range = (100, 1024)
    r_e = modis_vars_real["r_e"][:,:]
    r_e_range = (0, 70)
    twp = modis_vars_real["twp"][:,:]
    twp_range = (0.25, 25)

    gs = gridspec.GridSpec(num_gen+3,num_samples,
        height_ratios=(1,1)+(2,)*(num_gen+1),hspace=0.1,wspace=0.1)
    fig = plt.figure(figsize=(num_samples*1.5, 2.0+num_gen*1.6))

    for i in range(num_samples):
        ax_tau = plt.subplot(gs[0,i])
        ax_tau.set_title(str(i+first_column_num))
        ax_p = plot_column(ax_tau, tau_c[i,:], p_top[i,:], 
            pix_extent, scene_size, range1=tau_c_range, 
            range2=p_top_range, inv2=True, log1=True, log2=True,
            ticks1=([1, 10, 100], ["1", "10", "100"]),
            ticks2=([1000, 300, 100], ["1000", "300", "100"]),
            c1=modis_colors["tau_c"], c2=modis_colors["p_top"])
        ax_tau.tick_params(labelbottom=False, labelleft=(i==0))
        if i==0:
            ax_tau.set_ylabel("$\\tau_c$", color=modis_colors["tau_c"])
        ax_p.tick_params(labelbottom=False, labelright=(i==num_samples-1))
        if i == num_samples-1:
            ax_p.set_ylabel("$P_\\mathrm{top}$\n$\\mathrm{[hPa]}$", 
                color=modis_colors["p_top"])

        ax_twp = plt.subplot(gs[1,i])
        ax_re = plot_column(ax_twp, twp[i,:], r_e[i,:], 
            pix_extent, scene_size, range1=twp_range, 
            range2=r_e_range, log1=True, log2=False,
            ticks1=([0.25, 1, 4, 16], ["0.25", "1", "4", "16"]),
            ticks2=([0, 20, 40, 60], ["0", "20", "40", "60"]),
            c1=modis_colors["twp"], c2=modis_colors["r_e"])
        ax_twp.tick_params(labelbottom=False, labelleft=(i==0))
        if i==0:
            ax_twp.set_ylabel("$\\mathrm{CWP}$\n$\\mathrm{[g\\,m^{-2}]}$", 
                color=modis_colors["twp"])
        ax_re.tick_params(labelbottom=False, labelright=(i==num_samples-1))
        if i == num_samples-1:
            ax_re.set_ylabel("$r_e$\n$\\mathrm{[\\mu m]}$", 
                color=modis_colors["r_e"])

        for k in range(num_gen):
            ax_gen = plt.subplot(gs[2+k,i])
            plot_scene(ax_gen, scene_gen[k][i,:,:,0], pix_extent, scene_size)
            ax_gen.tick_params(labelbottom=False, labelleft=(i==0))
            if i==0:
                if k==0:
                    label = "Gen. ($\\mathbf{z}=\\mathbf{0}$)\nAltitude [km]"
                else:
                    label = "Generated\nAltitude [km]"
                ax_gen.set_ylabel(label)

        ax_real = plt.subplot(gs[-1,i])
        plot_scene(ax_real, scene_real[i,:,:,0], pix_extent, scene_size)
        ax_real.tick_params(labelleft=(i==0))
        if i==0:
            ax_real.set_ylabel("Real\nAltitude [km]")
        ax_real.set_xlabel("Distance [km]")

    add_dBZ_colorbar(fig, [0.91, 0.11, 0.018, 0.625])

    return fig


def plot_samples_cmp_all(gen, scene_real, modis_vars, modis_mask):
    # these have been hand-picked to illustrate specific cases
    samples_sel_1 = [13013, 14048, 6781, 4624, 4046, 13186, 12232, 16610]
    samples_sel_2 = [1572, 9281, 19336, 7966, 1720, 14624, 12110, 4831]

    # these were selected randomly from the validation dataset
    samples_rnd = [
        [ 5834,  3965, 10272, 18781,  7287, 16371,  6954,  9073],
        [15363, 14201, 19502,  2323,  2584,  5085,  4095,  8863],
        [ 7709,  7128,  8838, 11384,  9287, 14853,  7351,  4117],
        [ 9386, 14239,  3404, 18171,  7986,  8909,  3667, 10532],
        [ 2476, 14630, 11998, 19703, 14103,  3712, 13003, 18242],
        [ 5510, 10169, 10056,  6431,  7940, 15881,  8121, 14294],
        [14182, 17578, 14450,   942, 14607,  5357, 10741,  4936],
        [12566, 10774,   705,  6876,  1099,  9760,  7851,  4748]
    ]

    samples = [samples_sel_1, samples_sel_2] + samples_rnd
    plot_names = ["real_gen_cmp_sel-1", "real_gen_cmp_sel-2"] + \
        ["real_gen_cmp_rnd-{}".format(i) for i in range(len(samples_rnd))]
    first_column_num = [1]*len(plot_names)
    first_column_num[1] = 9

    for (s, fn, fcn) in zip(samples, plot_names, first_column_num):
        plot_samples_cmp(gen, scene_real[s,...], 
            modis_vars[s,...], modis_mask[s,...], first_column_num=fcn)
        plt.savefig("../figures/{}.pdf".format(fn), bbox_inches='tight')
        plt.close()

    dist_plot_names = \
        ["gen_dist-{}.pdf".format(i) for i in range(len(samples_sel_2))]
    for (s,fn) in zip(samples_sel_2, dist_plot_names):
        plot_distribution(gen, modis_vars, modis_mask, s)
        plt.savefig("../figures/{}".format(fn), bbox_inches='tight')
        plt.close()


def plot_distribution(gen, modis_vars, modis_mask, sample_num,
    noise_dim=64, grid_size=(8,8)):

    gs = gridspec.GridSpec(grid_size[0], grid_size[1],
        hspace=0.1,wspace=0.1)
    fig = plt.figure(figsize=(grid_size[1]*1.5, grid_size[0]*1.5))

    num_samples = grid_size[0]*grid_size[1]
    ind_array = [sample_num]*num_samples
    modis_vars_s = modis_vars[ind_array,...]
    modis_mask_s = modis_mask[ind_array,...]

    scene_gen = generate_scenes(gen, modis_vars_s, modis_mask_s, 
        noise_dim=noise_dim, noise_scale=2.0,
        rng_seed=335573)
    scene_gen = data_utils.rescale_scene(scene_gen)

    k = 0
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            ax_gen = plt.subplot(gs[i,j])
            plot_scene(ax_gen, scene_gen[k,:,:,0])
            k += 1
            if j==0:
                ax_gen.set_ylabel("Altitude [km]")
            if i==grid_size[0]-1:
                ax_gen.set_xlabel("Distance [km]")
            ax_gen.tick_params(labelbottom=(i==grid_size[0]-1), 
                labelleft=(j==0))

    add_dBZ_colorbar(fig, [0.91, 0.11, 0.018, 0.77])


def plot_gen_vary(gen, modis_vars, modis_mask,
    scene_size=64, vary_space=(-2,2,9), noise_dim=64):
    # s_vary = 10939

    modis_var_dim = modis_vars.shape[-1]
    mask_bool = modis_mask[...,0].astype(bool)
    means = [modis_vars[...,i][mask_bool].mean() for 
        i in range(modis_var_dim)]
    stds = [modis_vars[...,i][mask_bool].std() for 
        i in range(modis_var_dim)]

    grid_size = (modis_var_dim, vary_space[2])
    N = grid_size[0]*grid_size[1]
    gs = gridspec.GridSpec(grid_size[0], grid_size[1],
        hspace=0.1,wspace=0.1)
    fig = plt.figure(figsize=(grid_size[1]*1.5, grid_size[0]*1.5))

    vary_samples = np.linspace(*vary_space)
    num_samples = len(vary_samples)

    modis_vars = np.zeros((N,scene_size,modis_var_dim),
        dtype=np.float32)
    modis_mask = np.ones((N,scene_size,1), dtype=np.float32)

    k = 0
    for i in range(modis_var_dim):
        modis_vars[...,i] = means[i]
        for s in vary_samples:
            modis_vars[k,:,i] = means[i]+stds[i]*s
            k += 1
    #modis_vars[0,:,0] = -1

    scene_gen = np.zeros((N,64,64,1), dtype=np.float32)
    #for i in range(N):
    #    scene_gen[i:i+1,...] = generate_scenes(gen,
    #        modis_vars[i:i+1,...], modis_mask[i:i+1,...],
    #        noise_dim=noise_dim, zero_noise=True)
    scene_gen = generate_scenes(gen, modis_vars, modis_mask, 
        noise_dim=noise_dim, zero_noise=True,
        rng_seed=743708)
    scene_gen = data_utils.rescale_scene(scene_gen)

    #return scene_gen

    var_labels = {
        0: "$\\tau_c'$",
        1: "$P_\\mathrm{top}'$",
        2: "$r_e'$",
        3: "$\\mathrm{CWP}'$"
    } 

    k=0
    for i in range(modis_var_dim):
        for j in range(num_samples):
            ax_gen = plt.subplot(gs[i,j])
            plot_scene(ax_gen, scene_gen[k,:,:,0])
            k += 1
            if j==0:
                ax_gen.set_ylabel(var_labels[i]+"\nAltitude [km]")
            ax_gen.tick_params(labelbottom=(i==modis_var_dim-1), 
                labelleft=(j==0))
            if i==grid_size[0]-1:
                sig_label = "${:+.1f}\\sigma$".format(vary_samples[j])
                if vary_samples[j]==0:
                    sig_label = sig_label.replace('+','')
                ax_gen.set_xlabel("Distance [km]\n"+sig_label)

    add_dBZ_colorbar(fig, [0.91, 0.11, 0.018, 0.77])


def make_dBZ_hist(batch, dBZ_range=(-30,20.1,1)):
    dBZ_bins = np.arange(*dBZ_range)
    hist_shape = (
        batch.shape[1],
        len(dBZ_bins)-1
    )
    hist = np.zeros(hist_shape)
    for i in range(hist_shape[0]):
        data = batch[:,i,...].ravel()
        data = data[np.isfinite(data)]
        hist_height = np.histogram(data, dBZ_bins)[0]
        hist[i,:] = hist_height

    return hist


def generated_hist(gen, modis_vars, modis_mask, batch_size=1024,
    noise_dim=64):

    hist = None
    N = modis_vars.shape[0]
    for i in range(0,N,batch_size):
        vars_batch = modis_vars[i:i+batch_size,...]
        mask_batch = modis_mask[i:i+batch_size,...]
        gen_batch = generate_scenes(gen, vars_batch, mask_batch,
            noise_dim=noise_dim, rng_seed=493411+i)
        gen_batch = data_utils.rescale_scene(gen_batch)
        hist_batch = make_dBZ_hist(gen_batch)
        if hist is None:
            hist = hist_batch
        else:
            hist += hist_batch

    return hist


def real_hist(scenes_real):
    scenes_real = data_utils.rescale_scene(scenes_real)
    return make_dBZ_hist(scenes_real)


def plot_hist(hist_real, hist_gen, pix_extent=(0.24,1.09)):
    gs = gridspec.GridSpec(1, 3,
        hspace=0.1,wspace=0.15)
    fig = plt.figure(figsize=(9,4))

    scene_size = hist_real.shape[0]
    hist_real_norm = hist_real / hist_real.sum()
    hist_gen_norm = hist_gen / hist_gen.sum()
    hist_diff = hist_gen_norm - hist_real_norm
    norm = colors.Normalize(0,
        max(hist_real_norm.max(),hist_gen_norm.max()))
    max_diff = abs(hist_diff).max()
    norm_diff = colors.Normalize(-max_diff,max_diff)

    ax = plt.subplot(gs[0,0])
    plt.imshow(hist_real_norm, aspect='auto', norm=norm,
        extent=[-30,20,0,scene_size*pix_extent[0]],
        cmap=vir_white)
    plt.xlabel("Reflectivity [dBZ]")
    plt.ylabel("Altitude [km]")
    ax.set_xticks([-20,-10,0,10,20])
    ax.set_yticks([0,4,8,12])
    cb = plt.colorbar(orientation='horizontal')
    cb.ax.set_xlabel("Norm. occurrence (real)")
    cb.ax.tick_params(labelsize=8)

    ax = plt.subplot(gs[0,1])
    plt.imshow(hist_gen_norm, aspect='auto', norm=norm,
        extent=[-30,20,0,scene_size*pix_extent[0]],
        cmap=vir_white)
    plt.xlabel("Reflectivity [dBZ]")
    ax.set_xticks([-20,-10,0,10,20])
    ax.set_yticks([0,4,8,12])
    cb = plt.colorbar(orientation='horizontal')
    cb.ax.set_xlabel("Norm. occurrence (generated)")
    cb.ax.tick_params(labelsize=8)

    ax = plt.subplot(gs[0,2])
    plt.imshow(hist_diff, aspect='auto', norm=norm_diff,
        extent=[-30,20,0,scene_size*pix_extent[0]],
        cmap="RdBu_r")
    plt.xlabel("Reflectivity [dBZ]")
    ax.set_xticks([-20,-10,0,10,20])
    ax.set_yticks([0,4,8,12])
    cb = plt.colorbar(orientation='horizontal')
    cb.ax.set_xlabel("Occurrence bias (gen. - real)")
    cb.ax.tick_params(labelsize=8)


def load_data_and_models(scenes_fn, model_name="cs_modis_cgan-release", 
    epoch=45, scene_size=64, modis_var_dim=4, noise_dim=64, 
    lr_disc=0.0001, lr_gan=0.0002):

    scenes = data_utils.load_cloudsat_scenes(scenes_fn, shuffle_seed=214101)

    (gen, disc, gan, opt_disc, opt_gan) = train.create_models(
        scene_size, modis_var_dim, noise_dim, lr_disc, lr_gan)

    train.load_model_state(gen, disc, gan, model_name, epoch)

    return (scenes, gen, disc, gan)


def plot_all(scenes_fn, model_name="cs_modis_cgan-release"):
    (scenes, gen, disc, gan) = load_data_and_models(scenes_fn,
        model_name=model_name)
    
    (scene_real, modis_vars, modis_mask) = scenes["validate"]

    plot_samples_cmp_all(gen, scene_real, modis_vars, modis_mask)

    plot_gen_vary(gen, modis_vars, modis_mask)
    plt.savefig("../figures/gen_vary.pdf", bbox_inches='tight')
    plt.close()

    hist_gen = generated_hist(gen, modis_vars, modis_mask)
    hist_real = real_hist(scene_real)
    plot_hist(hist_real, hist_gen)
    plt.savefig("../figures/real_gen_hist.pdf", bbox_inches='tight')
    plt.close()

