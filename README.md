# CS-MODIS-CGAN

Keras implementation of CGAN for generation of CloudSat scenes from MODIS data. 

This code can be used to reproduce results from:  
Jussi Leinonen, Alexandre Guillaume and Tianle Yuan, Reconstruction of Cloud Vertical Structure with a Generative Adversarial Network, Submitted to _Geophysical Research Letters_, preprint available at https://doi.org/10.31223/osf.io/w26ja.

To run the code, you'll need NumPy, matplotlib, NetCDF4 and Keras with Tensorflow (and the dependencies thereof). A GPU is highly recommended (we trained the model with an Nvidia Tesla K80).

You'll also want to download the training dataset at https://doi.org/10.7910/DVN/BZEZC2.

To reproduce the results with the pre-trained model:
```bash
python main.py plot --scenes_file=<file>
```
where `<file>` is a path to the file containing the dataset. This will output plots into the `figures` directory.

To run the training:
```bash
python main.py train --scenes_file=<file>
```
This will output saved model weights into the `models` directory. Note that, since the initial weights are randomizes, the results will not be exactly equal to those reported in the paper.

If you find any problems with the code, please submit an [issue](/../../issues/). For other questions, email jussi.leinonen@epfl.ch.
