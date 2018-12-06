import os
import argparse


def launch_training(**kwargs):

    # Launch training
    train.train(**kwargs)


def launch_eval(**kwargs):

    # Launch training
    eval.eval(**kwargs)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('mode', type=str, help="train or eval")
    parser.add_argument('--generator', type=str, default="upsampling", help="upsampling or deconv")
    parser.add_argument('--dset', type=str, default="mnist", help="mnist or celebA")
    parser.add_argument('--scenes_file', type=str, default="../data/patches_comp.nc", help="CloudSat scenes file")
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    parser.add_argument('--n_batch_per_epoch', default=2000, type=int, help="Number of batches per epoch")
    parser.add_argument('--nb_epoch', default=400, type=int, help="Number of training epochs")
    parser.add_argument('--epoch', default=-1, type=int, help="Epoch at which weights were saved for evaluation")
    parser.add_argument('--nb_classes', default=2, type=int, help="Number of classes")
    parser.add_argument('--do_plot', default=False, type=bool, help="Debugging plot")
    parser.add_argument('--bn_mode', default=2, type=int, help="Batch norm mode")
    parser.add_argument('--img_dim', default=64, type=int, help="Image width == height (only specify for CelebA)")
    parser.add_argument('--noise_dim', default=16, type=int, help="noise dimension")
    parser.add_argument('--cont_dim', default=16, type=int, help="Latent continuous dimensions")
    parser.add_argument('--cat_dim', default=1, type=int, help="Latent categorical dimension")
    parser.add_argument('--noise_scale', default=0.5, type=float,
                        help="variance of the normal from which we sample the noise")
    parser.add_argument('--lr_factor', default=1.0, type=float,
                        help="Learning rate multiplier")


    args = parser.parse_args()

    scenes_file = args.scenes_file
    
