## Problem statement

We have to code a network to do the following task

- Prediction of number in binary format from image (MNIST dataset)
- Adding the MNIST prediction with a random number and output in binary format

**Example**

Input to the network ->  MNIST Image  , 5 (random number (0-9))                              

<img src="assests\dataset_sample.png" alt="sample_image" width="200" />


Expected output from the network

1. 00111 (Binary of 7)
2. 01100 (Binary of 12, Summation of 5 + 7)



## Solution approach



#### Dataset creation

First we downloaded MNIST data. 

```python
train_set = torchvision.datasets.MNIST('/data', train=True, download=True, transform=train_transforms)
train_set, val_set = torch.utils.data.random_split(train_set, [50000, 10000])
test_set = torchvision.datasets.MNIST('/data', train=False, download=True, transform=test_transforms)
```



As we are not passing plain MNIST data to the network we need to create custom dataset class.

```python
class CustomMNIST(Dataset):
  def __init__(self, MNIST):
    self.MNIST = MNIST
    self.binary_label_list = { 0:"00000", 1:"00001", 2:"00010", 3:"00011", 4:"00100", 5:"00101", 6:"00110", 7:"00111", 8:"01000", 9:"01001", 10:"01010", 11: "01011", 12:"01100", 13:"01101", 14:"01110", 15:"01111", 16:"10000", 17:"10001", 18:"10010"} 

  def get_binary_number(self,number):
    binary_number = self.binary_label_list[number]
    return binary_number

  def __getitem__(self, index):
    mnist_image = self.MNIST[index][0]
    label = self.MNIST[index][1]

    label_binary = self.get_binary_number(label)
    binary_target = targets_mnist[class_labels_mnist.index(label_binary)]


    random_number = random.randint(0,9)
    
    # One hot encoding vector for random number  
    one_hot_encoding_random_number = torch.nn.functional.one_hot(torch.arange(0, 10))

    #Summation
    sum_output = label + random_number

    sum_target = targets_summation[class_labels_summation.index(self.get_binary_number(sum_output))]
    # print("label binary target", label, binary_target)
    return mnist_image, binary_target , one_hot_encoding_random_number[random_number], sum_target

  def __len__(self):
    return len(self.MNIST)

```



In getitem method we are structuring the data. The following things we are doing

- Returning mnist image
- mnist label -> Binary format 
- Random number + mnist label -> Sum output -> Binary format



#### Training and validation curve



<img src="assests\training_curve.png" alt="sample_image" width="200" />

#### Results:

MNIST accuracy: 98.25

Summation accuracy: 93.6



#### Sample inference

<img src="assests\result_sample.png" alt="sample_image" style="zoom:25%;" />
