# CS-MODIS-CGAN

Keras implementation of CGAN for generation of CloudSat scenes from MODIS data. 

This code can be used to reproduce results from:  
Jussi Leinonen, Alexandre Guillaume and Tianle Yuan, Reconstruction of Cloud Vertical Structure with a Generative Adversarial Network, Submitted to _Geophysical Research Letters_, preprint available at https://XXX.

To run the code, you'll need NumPy, matplotlib, NetCDF4 and Keras with Tensorflow (and the dependencies thereof). A GPU is highly recommended (we trained the model with an Nvidia Tesla K80).

To reproduce the results with the pre-trained model:
```bash
python main.py plot
```
This will output plots into the `figures` directory.

To run the training:
```bash
python main.py train
```
This will output saved model weights into the `models` directory.

If you find any problems with the code, please submit an [issue](/../../issues/). For other questions, email jussi.s.leinonen@jpl.caltech.edu.
