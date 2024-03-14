# CAE_comparison
A python script to evaluate the performance of Convolutif Auto-Encoders.

## Requirements
```angular2
pytorch
torchvision
skimage
pytorch_wavelets
```

Run code with cmd line:
```angular2
python run.py
```

You can change the parameters with this function in the run.py file:
```angular2
train.execute(model_name='CAE', depth=3, coeff_NB=4, database="MNIST", nb_epoch=10, pth_name="model.pth")
```
Parameters are detailed in the comments below
