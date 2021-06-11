## Assignment 6 : Experimentation with different types of normalization and regularization methods (Early submission of late assignment)



### Problem statement:

Experiment with different normalization and regularization methods on MNIST

- Network with Group Normalization + L1
- Network with Layer Normalization + L2
- Network with BatchNorm with L1 and L2



#### Code Structure:

model.py : Contains network structure

misclassified_images.py : Contains utility function to plot misclassified images from inference on test loader.

train_test_loader.py : Contains code for train test dataloader.

trainer.py : Contains training and testing code of the network



Understanding Normalization

There are mainly three types of Normalization techniques we will be discussing:-

- Batch Normalization
- Layer Normalization
- Group Normalization

![1_xf37Ts0mdBGiS-keOwviOQ](/home/adit/Downloads/1_xf37Ts0mdBGiS-keOwviOQ.png)

##### Batch Normalization

Rescaling the data points w.r.t each channel. 

##### Layer Normalization

Rescaling the data points w.r.t each image across all channels

##### Group Normalization

Rescaling the data points w.r.t specific group of layer in an image



##### Example

![Screenshot from 2021-06-12 03-08-11](/home/adit/Pictures/Screenshot from 2021-06-12 03-08-11.png)

### Model learning graphs



##### Training Loss

##### ![train_loss](/home/adit/Downloads/train_loss.png)

##### Test Loss

![test_loss](/home/adit/Downloads/test_loss.png)



##### Training Accuracy



![train_acc](/home/adit/Downloads/train_acc.png)



##### Testing Accuracy

![test_acc](/home/adit/Downloads/test_acc.png)



### Misclassified Images



#### Model 1 (Group Normalization + L1)

![grp_nrm_img](/home/adit/Downloads/grp_nrm_img.png)

#### Model 2 (Layer Normalization + L2)



![layer_norm_img](/home/adit/Downloads/layer_norm_img.png)

#### Model 3 (Batch Normalization + L1 + L2 )



![batch_norm_img](/home/adit/Downloads/batch_norm_img.png)