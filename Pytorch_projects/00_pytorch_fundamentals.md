## 00. PyTorch Fundamentals

Resource notebook: https://www.learnpytorch.io/00_pytorch_fundamentals/

For questions: https://github.com/mrdbourke/pytorch-deep-learning/discussions


```python
# fundamental data science packages
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
print(torch.__version__)
```

    2.5.1+cu121


## Introduction to Tensors

# Creating tensors

PyTorch tensors are created using `torch.Tensor()` = https://pytorch.org/docs/stable/tensors.html

### Scalar


```python
scalar = torch.tensor(7)
scalar
```




    tensor(7)




```python
scalar.ndim

# has no dimension bc it is a single number
```




    0




```python
# Get tensor back as Python int
scalar.item()
```




    7



### Vector


```python
vector = torch.tensor([7, 7])
vector
```




    tensor([7, 7])




```python
vector.ndim

# tipp: num of square bracket pairs in the beginning tell how many dimensions it has
```




    1




```python
vector.shape

# how many elements
```




    torch.Size([2])



### Matrix

Matrices and tensors are created in upper case


```python
MATRIX = torch.tensor([[7, 8],
                       [9, 10]])
MATRIX
```




    tensor([[ 7,  8],
            [ 9, 10]])




```python
MATRIX.ndim
```




    2




```python
# show specific dimensions of the matrix:
MATRIX[1]
```




    tensor([ 9, 10])




```python
MATRIX.shape
# 2x2 elements = 4
```




    torch.Size([2, 2])



### Tensor


```python
TENSOR = torch.tensor([[[1, 2, 3],
                        [3, 6, 9],
                        [2, 4, 5]]])
TENSOR
```




    tensor([[[1, 2, 3],
             [3, 6, 9],
             [2, 4, 5]]])




```python
TENSOR.ndim
```




    3




```python
TENSOR.shape

# the 1 tells how many elems are in the first bracket,
# the 3 tells how many are in the second bracket,
# the last 3 tells how many elems are in one inner bracket
```




    torch.Size([1, 3, 3])




```python
TENSOR[0]
```




    tensor([[1, 2, 3],
            [3, 6, 9],
            [2, 4, 5]])



### Random tensors

Why random tensors?

Random tensors are important bc the way many neural networks learn is that they start with tensors full of random numbers and then adjust those random numbers to better represent the data.

`Start with random numbers -> look at data -> update random numbers -> look at data -> update random numbers`

Torch random tensors - https://pytorch.org/docs/stable/generated/torch.rand.html


```python
# Create a random tensor of size (3, 4)
random_tensor = torch.rand(3, 4)
random_tensor
```




    tensor([[0.5094, 0.3575, 0.4705, 0.9525],
            [0.9872, 0.6594, 0.4554, 0.5705],
            [0.9707, 0.9505, 0.3900, 0.2835]])




```python
# Create a random tensor with similar shape to an img tensor
random_img_size_tensor = torch.rand(size=(224, 224, 3)) # height, width, col channels (R, G, B), col channel can also be at the front like (3, 224, 224)
random_img_size_tensor.shape, random_img_size_tensor.ndim
```




    (torch.Size([224, 224, 3]), 3)



### Zeros and Ones


```python
# Create a tensor of all zeros
zeros = torch.zeros(size=(3, 4))
zeros
```




    tensor([[0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.]])




```python
# used for masking (zero numbers out)
zeros * random_tensor
```




    tensor([[0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.]])




```python
# Create a tensor of all ones
ones = torch.ones(size=(3, 4))
ones
```




    tensor([[1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.]])




```python
ones.dtype # default data type
```




    torch.float32




```python
random_tensor.dtype
```




    torch.float32



### Creating a range of tensors and tensors-like


```python
# torch.range() is deprecated
# Use torch.arange
one_to_ten = torch.arange(start=1, end=11, step=1) # finishes at 11-1
one_to_ten
```




    tensor([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])




```python
# Creating tensors like
ten_zeros = torch.zeros_like(input=one_to_ten)
ten_zeros

# zeros in the same shape as one_to_ten
```




    tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])



### Tensor datatypes

**Note:** Tensor datatypes is one of the big errors you'll run into with PyTorch & Deep Learning:
1. Tensors not right datatype
2. Tensors not right shape
3. Tensors not on right device

Precision in Computing - https://en.wikipedia.org/wiki/Precision_(computer_science)


```python
# Float 32 tensor
float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=None,  # what datatype is the tensor (e.g. float32 (single precision), float16 (half precision, but calculate faster))
                               device=None, # what device is your tensor on, "cuda" or "cpu"
                               requires_grad=False) # whether or not to track gradients with this tensors operations
float_32_tensor
```




    tensor([3., 6., 9.])




```python
float_32_tensor.dtype # is float32 bc this is the default data type even if dtype is None
```




    torch.float32




```python
float_16_tensor = float_32_tensor.type(torch.float16)
float_16_tensor
```




    tensor([3., 6., 9.], dtype=torch.float16)




```python
float_16_tensor * float_32_tensor
```




    tensor([ 9., 36., 81.])




```python
int_32_tensor = torch.tensor([3, 6, 9], dtype=torch.int32)
int_32_tensor
```




    tensor([3, 6, 9], dtype=torch.int32)




```python
float_32_tensor * int_32_tensor
```




    tensor([ 9., 36., 81.])



### Getting information from tensors (tensor attributes)

1. Tensors not right datatype - to get datatype from a tensor, can use `tensor.dtype`
2. Tensors not right shape - to get shape from a tensor, can use `tensor.shape`
3. Tensors not on right device - to get device from a tensor, can use `tensor.device`


```python
# Create a tensor
some_tensor = torch.rand(3, 4)
some_tensor
```




    tensor([[0.4061, 0.0162, 0.3566, 0.8088],
            [0.6701, 0.8956, 0.7619, 0.9931],
            [0.7093, 0.3383, 0.1064, 0.9462]])




```python
some_tensor.size(), some_tensor.shape # gives same putput but size() is a func, shape is an attribute
```




    (torch.Size([3, 4]), torch.Size([3, 4]))




```python
# Find out details about some tensor
print(some_tensor)
print(f"Datatype of tensor: {some_tensor.dtype}")
print(f"Shape of tensor: {some_tensor.shape}")
print(f"Device tensor is on: {some_tensor.device}")
```

    tensor([[0.4061, 0.0162, 0.3566, 0.8088],
            [0.6701, 0.8956, 0.7619, 0.9931],
            [0.7093, 0.3383, 0.1064, 0.9462]])
    Datatype of tensor: torch.float32
    Shape of tensor: torch.Size([3, 4])
    Device tensor is on: cpu


### Manipulating Tensors (tensor operations)

Tensor operations include:
* Addition
* Subtraction
* Multiplication (element-wise)
* Division
* Matrix multiplication


```python
# Create a tensor and add 10 to it
tensor = torch.tensor([1, 2, 3])
tensor + 10
```




    tensor([11, 12, 13])




```python
# Multiply tensor by 10
tensor * 10
```




    tensor([10, 20, 30])




```python
tensor
```




    tensor([1, 2, 3])




```python
# Subtract 10
tensor - 10
```




    tensor([-9, -8, -7])




```python
# Try out PyTorch in-build functions
torch.mul(tensor, 10)
```




    tensor([10, 20, 30])




```python
torch.add(tensor, 10)
```




    tensor([11, 12, 13])



### Matrix multiplication

Two main ways of performing multiplication in neural networks and deep learning

1. Element-wise multiplication
2. Matrix multiplication (dot product)

More information on multiplying matrices - https://www.mathsisfun.com/algebra/matrix-multiplying.html

There are two main rules that performing matrix multiplication needs to satisfy:
1. The **inner dimensions** (the dims next to @) must match:
* `(3, 2) @ (3, 2)` won't work
* `(2, 3) @ (3, 2)` will work
* `(3, 2) @ (2, 3)` will work
2. The resulting matrix has the shape of the **outer dimensions**
* `(2, 3) @ (3, 2)` -> `(2, 2)`
* `(3, 2) @ (2, 3)` -> `(3, 3)`


```python
torch.matmul(torch.rand(8, 10), torch.rand(10, 8)).shape
```




    torch.Size([8, 8])




```python
# Element wise multiplication
print(tensor, "*", tensor)
print(f"Equals: {tensor * tensor}")
```

    tensor([1, 2, 3]) * tensor([1, 2, 3])
    Equals: tensor([1, 4, 9])



```python
# Matrix multiplication, @
torch.matmul(tensor, tensor)
```




    tensor(14)




```python
# Matrix multiplication by hand
1*1 + 2*2 + 3*3
```




    14




```python
%%time
value = 0
for i in range(len(tensor)):
  value += tensor[i] * tensor[i]
print(value)
```

    tensor(14)
    CPU times: user 1.68 ms, sys: 84 µs, total: 1.76 ms
    Wall time: 3.8 ms



```python
%%time
torch.matmul(tensor, tensor)

# 10 times faster
```

    CPU times: user 69 µs, sys: 8 µs, total: 77 µs
    Wall time: 81.3 µs





    tensor(14)



### One of the most common errors in deep learning: Shape errors


```python
# Shapes for matrix multiplication
tensor_A = torch.tensor([[1, 2],
                         [3, 4],
                         [5, 6]])
tensor_B = torch.tensor([[7, 10],
                         [8, 11],
                         [9, 12]])

torch.mm(tensor_A, tensor_B) # torch.mm is the same as torch.matmul (it's an alias for writing less code)

```


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    <ipython-input-46-ba7bd33faa73> in <cell line: 9>()
          7                          [9, 12]])
          8 
    ----> 9 torch.mm(tensor_A, tensor_B) # torch.mm is the same as torch.matmul (it's an alias for writing less code)
    

    RuntimeError: mat1 and mat2 shapes cannot be multiplied (3x2 and 3x2)



```python
tensor_A.shape, tensor_B.shape
```




    (torch.Size([3, 2]), torch.Size([3, 2]))



To fix our tensor shape issues, we can manipulate the shape of one of our tensors using a **transpose**

A **transpose** switches the axes or dimensions of a given tensor


```python
tensor_B, tensor_B.shape
```




    (tensor([[ 7, 10],
             [ 8, 11],
             [ 9, 12]]),
     torch.Size([3, 2]))




```python
tensor_B.T, tensor_B.T.shape
```




    (tensor([[ 7,  8,  9],
             [10, 11, 12]]),
     torch.Size([2, 3]))




```python
# The matrix multiplicaton operation works when tensor_B is transposed
print(f"Original shapes: tensor_A = {tensor_A.shape}, tensor_B = {tensor_B.shape}")
print(f"New shapes: tensor_A = {tensor_A.shape} (same shape as above), tensor_B.T = {tensor_B.T.shape}")
print(f"Multiplying: {tensor_A.shape} @ {tensor_B.T.shape} <- inner dimensions must match")
print("Output:\n")
output = torch.matmul(tensor_A, tensor_B.T)
print(output)
print(f"\nOutput shape: {output.shape}")
```

    Original shapes: tensor_A = torch.Size([3, 2]), tensor_B = torch.Size([3, 2])
    New shapes: tensor_A = torch.Size([3, 2]) (same shape as above), tensor_B.T = torch.Size([2, 3])
    Multiplying: torch.Size([3, 2]) @ torch.Size([2, 3]) <- inner dimensions must match
    Output:
    
    tensor([[ 27,  30,  33],
            [ 61,  68,  75],
            [ 95, 106, 117]])
    
    Output shape: torch.Size([3, 3])


## Finding the min, max, mean, sum (tensor aggregation)


```python
# Create a tensor
x = torch.arange(1, 100, 10)
x, x.dtype
```




    (tensor([ 1, 11, 21, 31, 41, 51, 61, 71, 81, 91]), torch.int64)




```python
# Find the min
torch.min(x), x.min()
```




    (tensor(1), tensor(1))




```python
# Find the max
torch.max(x), x.max
```




    (tensor(91), <function Tensor.max>)




```python
# Find the mean - note: the torch.mean() function requires a tensor of float32 datatype to work
torch.mean(x.type(torch.float32)), x.type(torch.float32).mean()
```




    (tensor(46.), tensor(46.))




```python
#Find the sum
torch.sum(x), x.sum()
```




    (tensor(460), tensor(460))



## Find the positional min and max



```python
x
```




    tensor([ 1, 11, 21, 31, 41, 51, 61, 71, 81, 91])




```python
# Find the position in tensor that has the minimum value with argmin() -> returns index position of target tensor where the minimum value occurs
x.argmin()
```




    tensor(0)




```python
x[0]
```




    tensor(1)




```python
# Find the position in tensor that has the maximum value with argmax()
x.argmax()
```




    tensor(9)




```python
x[9]
```




    tensor(91)



## Reshaping, stacking, squeezing and unsqueezing tensors

* Reshaping - reshapes an input tensor to a defined shape
* View - Return a view of an input tensor of certain shape but keep the same memory as  the original tensor
* Stacking - combine multiple tensors on top of each other (vstack) or side by side (hstack)
* Squeeze - removes all `1` dimensions from a tensor
* Unsqueeze - add a `1` dimension to a target tensor
* Permute - Return a view of the input with dimensions permuted (swapped) in a certain way


```python
# Create a tensor
import torch
x = torch.arange(1., 10.)
x, x.shape
```




    (tensor([1., 2., 3., 4., 5., 6., 7., 8., 9.]), torch.Size([9]))




```python
# Add an extra dimension
x_reshaped = x.reshape(1, 9) # has to be a shape for the same amount of elements
x_reshaped, x_reshaped.shape
```




    (tensor([[1., 2., 3., 4., 5., 6., 7., 8., 9.]]), torch.Size([1, 9]))




```python
# Change the view
z = x.view(1, 9)
z, z.shape
```




    (tensor([[1., 2., 3., 4., 5., 6., 7., 8., 9.]]), torch.Size([1, 9]))




```python
# Changing z changes x (bc a view of a tensor shares the same memory as the original input)
z[:, 0] = 5
z, x
```




    (tensor([[5., 2., 3., 4., 5., 6., 7., 8., 9.]]),
     tensor([5., 2., 3., 4., 5., 6., 7., 8., 9.]))




```python
# Stack tensors on top of each other
x_stacked = torch.stack([x, x, x, x], dim=0)
x_stacked
```




    tensor([[5., 2., 3., 4., 5., 6., 7., 8., 9.],
            [5., 2., 3., 4., 5., 6., 7., 8., 9.],
            [5., 2., 3., 4., 5., 6., 7., 8., 9.],
            [5., 2., 3., 4., 5., 6., 7., 8., 9.]])




```python
# vstack
x_vstacked = torch.vstack([x, x, x, x])
x_vstacked
```




    tensor([[5., 2., 3., 4., 5., 6., 7., 8., 9.],
            [5., 2., 3., 4., 5., 6., 7., 8., 9.],
            [5., 2., 3., 4., 5., 6., 7., 8., 9.],
            [5., 2., 3., 4., 5., 6., 7., 8., 9.]])




```python
# hstack
x_hstacked = torch.hstack([x, x, x, x])
x_hstacked
```




    tensor([5., 2., 3., 4., 5., 6., 7., 8., 9., 5., 2., 3., 4., 5., 6., 7., 8., 9.,
            5., 2., 3., 4., 5., 6., 7., 8., 9., 5., 2., 3., 4., 5., 6., 7., 8., 9.])




```python
# torch.squeeze() - removes all single dimensions from a target tensor
print(f"Previous tensor: {x_reshaped}")
print(f"Previous shape: {x_reshaped.shape}")

#Remove extra dimensions from x_reshaped
x_squeezed = x_reshaped.squeeze()
print(f"\nNew tensor: {x_squeezed}")
print(f"New shape: {x_squeezed.shape}")
```

    Previous tensor: tensor([[5., 2., 3., 4., 5., 6., 7., 8., 9.]])
    Previous shape: torch.Size([1, 9])
    
    New tensor: tensor([5., 2., 3., 4., 5., 6., 7., 8., 9.])
    New shape: torch.Size([9])



```python
#torch.unsqueeze() - adds a single dimension to a target tensor at a specific dim (dimension)
print(f"Previous target: {x_squeezed}")
print(f"Previous shape: {x_squeezed.shape}")

# Add an extra dimension with unsqueeze
x_unsqueezed = x_squeezed.unsqueeze(dim=0)
print(f"\nNew tensor: {x_unsqueezed}")
print(f"New shape: {x_unsqueezed.shape}")
```

    Previous target: tensor([5., 2., 3., 4., 5., 6., 7., 8., 9.])
    Previous shape: torch.Size([9])
    
    New tensor: tensor([[5., 2., 3., 4., 5., 6., 7., 8., 9.]])
    New shape: torch.Size([1, 9])



```python
# torch.permute - rearranges the simensions of a target tensor in a specified order
x_original = torch.rand(size=(224, 244, 3)) # [height, width, colour_channels]

#Permute the original tensor to rearrange the axis (or dim) order
x_permuted = x_original.permute(2, 0, 1) # shifts axis 0->1, 1->2, 2->0

print(f"Previous shape: {x_original.shape}")
print(f"New shape: {x_permuted.shape}") # [colour_channels, height, width]
```

    Previous shape: torch.Size([224, 244, 3])
    New shape: torch.Size([3, 224, 244])



```python
# permute creates a view of the original tensor

x_original[0, 0, 0] = 728218
x_original[0, 0, 0], x_permuted[0, 0, 0]
```




    (tensor(728218.), tensor(728218.))



## Indexing (selecting data from tensors)

Indexing with PyTorch is similar to indexing with NumPy


```python
# Create a tensor
import torch
x = torch.arange(1, 10).reshape(1, 3, 3)
x, x.shape
```




    (tensor([[[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]]]),
     torch.Size([1, 3, 3]))




```python
# Index on our new tensor
x[0]
```




    tensor([[1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]])




```python
# Index on the middle bracket (dim=1)
x[0][0]
```




    tensor([1, 2, 3])




```python
# Index on the most inner bracket (last dimension)
x[0][1][1]
```




    tensor(5)




```python
# You can also use ":" to select "all" of a target dimension
x[:, 0]
```




    tensor([[1, 2, 3]])




```python
# Get all values of 0th and 1st dimensions but only index 1 of 2nd dimension
x[:, :, 1]
```




    tensor([[2, 5, 8]])




```python
# Get all values of the 0 dimension but only the 1 index value of the 1st and 2nd dimension
x[:, 1, 1]
```




    tensor([5])




```python
# Get index 0 of 0th and 1st dimension and all values of 2nd dimension
x[0, 0, :]
```




    tensor([1, 2, 3])




```python
# Index on x to return 9
print(x[0][2][2])

# Index on x to return 3, 6, 9
print(x[:, :, 2])
```

    tensor(9)
    tensor([[3, 6, 9]])


## PyTorch tensors & NumPy

NumPy is a popular scientific Python numerical computing library.

Because of this, PyTorch has funcionality to interact with it.

* Data in NumPy, want in PyTorch tensor -> `torch.from_numpy(ndarray)` where ndarray is NumPy's main data type
* PyTorch tensor -> NumPy -> `torch.Tensor.numpy()`


```python
# NumPy array to tensor
import torch
import numpy as np

array = np.arange(1.0, 8.0)
tensor = torch.from_numpy(array) #.type(torch.float32) # warning: when converting from numpy -> pytorch, pytorch reflects numpy's default datatype of float64 unless specified otherwise
array, tensor
```




    (array([1., 2., 3., 4., 5., 6., 7.]),
     tensor([1., 2., 3., 4., 5., 6., 7.], dtype=torch.float64))




```python
# Change the value of array, what will this do to `tensor`?
array = array + 1
array, tensor
```




    (array([2., 3., 4., 5., 6., 7., 8.]),
     tensor([1., 2., 3., 4., 5., 6., 7.], dtype=torch.float64))




```python
# Tensor to NumPy array
tensor = torch.ones(7)
numpy_tensor = tensor.numpy() # it also reflects pytorch's default datatype of float32
tensor, numpy_tensor
```




    (tensor([1., 1., 1., 1., 1., 1., 1.]),
     array([1., 1., 1., 1., 1., 1., 1.], dtype=float32))




```python
# Change the tensor, what happens to `numpy_tensor`?
tensor = tensor + 1
numpy_tensor, tensor
```




    (array([1., 1., 1., 1., 1., 1., 1.], dtype=float32),
     tensor([2., 2., 2., 2., 2., 2., 2.]))



## Reproducibility (trying to take random out of random)

In short how a neural network learns:

`start with random numbers -> tensor operations -> update random numbers to try and make them better representations of the data -> again -> again -> again...`

To reduce the randomness in neural networks and PyTorch comes the concept of a **random seed**.

Essentially what the random seed does is "flavour" the randomness.


```python
import torch

# Create two random tensors
random_tensor_A = torch.rand(3, 4)
random_tensor_B = torch.rand(3, 4)

print(random_tensor_A)
print(random_tensor_B)
print(random_tensor_A == random_tensor_B)
```

    tensor([[0.8149, 0.7748, 0.7305, 0.2043],
            [0.0965, 0.4742, 0.4436, 0.4604],
            [0.0760, 0.4151, 0.5967, 0.5008]])
    tensor([[0.6524, 0.6775, 0.8430, 0.3630],
            [0.4615, 0.7154, 0.6135, 0.1010],
            [0.8771, 0.9271, 0.6470, 0.2771]])
    tensor([[False, False, False, False],
            [False, False, False, False],
            [False, False, False, False]])



```python
# Let's make some random but reproducible tensors
import torch

# Set the random seed
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
random_tensor_C = torch.rand(3, 4)

torch.manual_seed(RANDOM_SEED) # have to call this method for every new block of code
random_tensor_D = torch.rand(3, 4)

print(random_tensor_C)
print(random_tensor_D)
print(random_tensor_C == random_tensor_D)
```

    tensor([[0.8823, 0.9150, 0.3829, 0.9593],
            [0.3904, 0.6009, 0.2566, 0.7936],
            [0.9408, 0.1332, 0.9346, 0.5936]])
    tensor([[0.8823, 0.9150, 0.3829, 0.9593],
            [0.3904, 0.6009, 0.2566, 0.7936],
            [0.9408, 0.1332, 0.9346, 0.5936]])
    tensor([[True, True, True, True],
            [True, True, True, True],
            [True, True, True, True]])


Extra resources for Reproducibility:
* https://pytorch.org/docs/stable/notes/randomness.html
* https://en.wikipedia.org/wiki/Random_seed

## Running tensors and PyTorch objects on the GPUs (and making faster computations)

GPUs = faster computation on numbers, thanks to CUDA + NVIDIA hardware + PyTorch working bts to make everything hunky dory (good).

### 1. Getting a GPU

1. Easiest - Use Google Colab for a free GPU (options to upgrade as well)
2. Use your own GPU - takes a little bit of setup and requires the investment of purchasing a GPU, there's lots of options, see this what to get: https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/
3. Use cloud computing - GCP, AWS, Azure, these services allow you to rent computers on the cloud and access them

For 2, 3 PyTorch + GPU drivers (CUDA) takes a little bit of setting up, to do this, refer to PyTorch setup documentation: https://pytorch.org/get-started/locally/


```python
!nvidia-smi
```

    Sun Dec 15 14:08:55 2024       
    +---------------------------------------------------------------------------------------+
    | NVIDIA-SMI 535.104.05             Driver Version: 535.104.05   CUDA Version: 12.2     |
    |-----------------------------------------+----------------------+----------------------+
    | GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
    |                                         |                      |               MIG M. |
    |=========================================+======================+======================|
    |   0  Tesla T4                       Off | 00000000:00:04.0 Off |                    0 |
    | N/A   62C    P8               9W /  70W |      0MiB / 15360MiB |      0%      Default |
    |                                         |                      |                  N/A |
    +-----------------------------------------+----------------------+----------------------+
                                                                                             
    +---------------------------------------------------------------------------------------+
    | Processes:                                                                            |
    |  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
    |        ID   ID                                                             Usage      |
    |=======================================================================================|
    |  No running processes found                                                           |
    +---------------------------------------------------------------------------------------+


### 2. Check for GPU access with PyTorch


```python
# Check for GPU access with PyTorch
import torch
torch.cuda.is_available()
```




    True



For PyTorch since it's capable of running compute on the GPU or CPU, it's best practice to setup device agnostic code: https://pytorch.org/docs/stable/notes/cuda.html#best-practices

E.g. run on GPU if available, else default to CPU


```python
# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
device
```




    'cuda'




```python
# Count number of devices
torch.cuda.device_count()
```




    1



## 3. Putting tensors (and models) on the GPU

The reason we want our tensors/models on the GPU is bc using a GPU results in faster computations.


```python
# Create a tensor (default on the CPU)
tensor = torch.tensor([1, 2, 3])

# Tensor not on GPU
print(tensor, tensor.device)
```

    tensor([1, 2, 3]) cpu



```python
# Move tensor to GPU (if available)
tensor_on_gpu = tensor.to(device)
tensor_on_gpu
```




    tensor([1, 2, 3], device='cuda:0')



### 4. Moving tensors back to the CPU


```python
# If tensor is on GPU, can't transform it to NumPy
tensor_on_gpu.numpy()
# numpy only works on the CPU
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-9-1a55f76e8cda> in <cell line: 2>()
          1 # If tensor is on GPU, can't transform it to NumPy
    ----> 2 tensor_on_gpu.numpy()
          3 # numpy only works on the CPU


    TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.



```python
# To fix the GPU tensor with NumPy issue, we can first set it to the CPU
tensor_back_on_cpu = tensor_on_gpu.cpu().numpy()
tensor_back_on_cpu
```




    array([1, 2, 3])




```python
tensor_on_gpu
```




    tensor([1, 2, 3], device='cuda:0')




```python

```
