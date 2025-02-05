# 00. PyTorch Fundamentals Exercises

### 1. Documentation reading

A big part of deep learning (and learning to code in general) is getting familiar with the documentation of a certain framework you're using. We'll be using the PyTorch documentation a lot throughout the rest of this course. So I'd recommend spending 10-minutes reading the following (it's okay if you don't get some things for now, the focus is not yet full understanding, it's awareness):
  * The documentation on [`torch.Tensor`](https://pytorch.org/docs/stable/tensors.html#torch-tensor).
  * The documentation on [`torch.cuda`](https://pytorch.org/docs/master/notes/cuda.html#cuda-semantics).




```python
# No code solution (reading)
```

### 2. Create a random tensor with shape `(7, 7)`.



```python
import torch

tensor_A = torch.rand(7, 7)
tensor_A, tensor_A.shape
```




    (tensor([[0.3044, 0.5531, 0.1153, 0.1375, 0.0462, 0.2880, 0.8678],
             [0.8725, 0.6062, 0.5742, 0.6131, 0.2019, 0.6525, 0.5457],
             [0.9594, 0.4755, 0.8344, 0.0202, 0.9237, 0.0431, 0.2629],
             [0.4530, 0.4405, 0.1733, 0.3904, 0.3074, 0.0698, 0.4178],
             [0.8279, 0.9324, 0.3784, 0.3179, 0.7784, 0.7271, 0.2666],
             [0.4405, 0.1591, 0.0871, 0.4617, 0.6870, 0.6496, 0.4721],
             [0.1451, 0.9903, 0.9418, 0.8182, 0.4266, 0.9814, 0.1834]]),
     torch.Size([7, 7]))



### 3. Perform a matrix multiplication on the tensor from 2 with another random tensor with shape `(1, 7)` (hint: you may have to transpose the second tensor).


```python
tensor_B = torch.rand(1, 7)
print(tensor_B)

prod = torch.matmul(tensor_A, tensor_B.T)
print(prod)
```

    tensor([[0.5580, 0.7877, 0.9200, 0.9772, 0.1824, 0.1744, 0.4337]])
    tensor([[1.2810],
            [2.4790],
            [1.9873],
            [1.3902],
            [2.2396],
            [1.3458],
            [2.8556]])


### 4. Set the random seed to `0` and do 2 & 3 over again.

The output should be:
```
(tensor([[1.8542],
         [1.9611],
         [2.2884],
         [3.0481],
         [1.7067],
         [2.5290],
         [1.7989]]), torch.Size([7, 1]))
```


```python
# Set manual seed
RANDOM_SEED = 0
torch.manual_seed(RANDOM_SEED)
tensor_A_seed = torch.rand(7, 7)
tensor_B_seed = torch.rand(1, 7)

# Matrix multiply tensors
prod_seed = torch.matmul(tensor_A_seed, tensor_B_seed.T)
prod_seed, prod_seed.shape
```




    (tensor([[1.8542],
             [1.9611],
             [2.2884],
             [3.0481],
             [1.7067],
             [2.5290],
             [1.7989]]),
     torch.Size([7, 1]))



### 5. Speaking of random seeds, we saw how to set it with `torch.manual_seed()` but is there a GPU equivalent? (hint: you'll need to look into the documentation for `torch.cuda` for this one)
  * If there is, set the GPU random seed to `1234`.


```python
# Set random seed on the GPU
torch.cuda.manual_seed(1234)
```


### 6. Create two random tensors of shape `(2, 3)` and send them both to the GPU (you'll need access to a GPU for this). Set `torch.manual_seed(1234)` when creating the tensors (this doesn't have to be the GPU random seed). The output should be something like:

```
Device: cuda
(tensor([[0.0290, 0.4019, 0.2598],
         [0.3666, 0.0583, 0.7006]], device='cuda:0'),
 tensor([[0.0518, 0.4681, 0.6738],
         [0.3315, 0.7837, 0.5631]], device='cuda:0'))
```


```python
torch.manual_seed(1234)
tensor_C = torch.rand(2, 3)
tensor_D = torch.rand(2, 3)

if torch.cuda.is_available:
  device = "cuda"
  tensor_C_gpu = tensor_C.to(device)
  tensor_D_gpu = tensor_D.to(device)

print(f"Device: " + device)
print(tensor_C_gpu)
print(tensor_D_gpu)
```

    Device: cuda
    tensor([[0.0290, 0.4019, 0.2598],
            [0.3666, 0.0583, 0.7006]], device='cuda:0')
    tensor([[0.0518, 0.4681, 0.6738],
            [0.3315, 0.7837, 0.5631]], device='cuda:0')



### 7. Perform a matrix multiplication on the tensors you created in 6 (again, you may have to adjust the shapes of one of the tensors).

The output should look like:
```
(tensor([[0.3647, 0.4709],
         [0.5184, 0.5617]], device='cuda:0'), torch.Size([2, 2]))
```


```python
prod = torch.matmul(tensor_C_gpu, tensor_D_gpu.T)
prod, prod.shape
```




    (tensor([[0.3647, 0.4709],
             [0.5184, 0.5617]], device='cuda:0'),
     torch.Size([2, 2]))



### 8. Find the maximum and minimum values of the output of 7.


```python
# Find max
max = torch.max(prod)

# Find min
min = torch.min(prod)

max, min
```




    (tensor(0.5617, device='cuda:0'), tensor(0.3647, device='cuda:0'))



### 9. Find the maximum and minimum index values of the output of 7.


```python
# Find arg max
arg_max = torch.argmax(prod)

# Find arg min
arg_min = torch.argmin(prod)

arg_max, arg_min
```




    (tensor(3, device='cuda:0'), tensor(0, device='cuda:0'))




### 10. Make a random tensor with shape `(1, 1, 1, 10)` and then create a new tensor with all the `1` dimensions removed to be left with a tensor of shape `(10)`. Set the seed to `7` when you create it and print out the first tensor and it's shape as well as the second tensor and it's shape.

The output should look like:

```
tensor([[[[0.5349, 0.1988, 0.6592, 0.6569, 0.2328, 0.4251, 0.2071, 0.6297,
           0.3653, 0.8513]]]]) torch.Size([1, 1, 1, 10])
tensor([0.5349, 0.1988, 0.6592, 0.6569, 0.2328, 0.4251, 0.2071, 0.6297, 0.3653,
        0.8513]) torch.Size([10])
```


```python
torch.manual_seed(7)
tensor_E = torch.rand(1, 1, 1, 10)
print(tensor_E, tensor_E.shape)

tensor_E_squeezed = torch.squeeze(tensor_E)
print(tensor_E_squeezed, tensor_E_squeezed.shape)

```

    tensor([[[[0.5349, 0.1988, 0.6592, 0.6569, 0.2328, 0.4251, 0.2071, 0.6297,
               0.3653, 0.8513]]]]) torch.Size([1, 1, 1, 10])
    tensor([0.5349, 0.1988, 0.6592, 0.6569, 0.2328, 0.4251, 0.2071, 0.6297, 0.3653,
            0.8513]) torch.Size([10])



```python

```
