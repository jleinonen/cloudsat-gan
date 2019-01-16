import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, help="train or plot")
    parser.add_argument('--scenes_file', type=str, 
        help="CloudSat scenes file")
    parser.add_argument('--run_name', type=str, default="",
        help="Suffix to use for this training run")
    
    args = parser.parse_args()
    scenes_file = args.scenes_file

    if mode == "train":
        import train
        train.train_cs_modis_cgan_full(scenes_fn)
    elif mode == "plot":
        import plots
        