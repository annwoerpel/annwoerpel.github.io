<a href="https://colab.research.google.com/github/mrdbourke/pytorch-deep-learning/blob/main/extras/exercises/04_pytorch_custom_datasets_exercises.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# 04. PyTorch Custom Datasets Exercises

Welcome to the 04. PyTorch Custom Datasets exercise.

The best way to practice PyTorch code is to write more PyTorch code.

So read the original notebook and try to complete the exercises by writing code where it's required.

Feel free to reference the original resources whenever you need but should practice writing all of the code yourself.

## Resources

1. These exercises/solutions are based on [notebook 04 of the Learn PyTorch for Deep Learning course](https://www.learnpytorch.io/04_pytorch_custom_datasets/).
2. See a live [walkthrough of the solutions (errors and all) on YouTube](https://youtu.be/vsFMF9wqWx0).
3. See [other solutions on the course GitHub](https://github.com/mrdbourke/pytorch-deep-learning/tree/main/extras/solutions).


```python
# Check for GPU
!nvidia-smi
```

    Mon Jan 13 14:48:17 2025       
    +---------------------------------------------------------------------------------------+
    | NVIDIA-SMI 535.104.05             Driver Version: 535.104.05   CUDA Version: 12.2     |
    |-----------------------------------------+----------------------+----------------------+
    | GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
    |                                         |                      |               MIG M. |
    |=========================================+======================+======================|
    |   0  Tesla T4                       Off | 00000000:00:04.0 Off |                    0 |
    | N/A   52C    P8              10W /  70W |      0MiB / 15360MiB |      0%      Default |
    |                                         |                      |                  N/A |
    +-----------------------------------------+----------------------+----------------------+
                                                                                             
    +---------------------------------------------------------------------------------------+
    | Processes:                                                                            |
    |  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
    |        ID   ID                                                             Usage      |
    |=======================================================================================|
    |  No running processes found                                                           |
    +---------------------------------------------------------------------------------------+



```python
# Import torch
import torch
from torch import nn

# Exercises require PyTorch > 1.10.0
print(torch.__version__)

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
device
```

    2.5.1+cu121





    'cuda'



## 1. Our models are underperforming (not fitting the data well). What are 3 methods for preventing underfitting? Write them down and explain each with a sentence.

* Train for longer: The model gets more time to learn the patterns of the data. Add more epochs.
* Tweak learning rate: Lower the learning rate to prevent the weights being updated too much.
* Use Transfer Learning: Use patterns from a working model and adjust it to your own problem.

## 2. Recreate the data loading functions we built in [sections 1, 2, 3 and 4 of notebook 04](https://www.learnpytorch.io/04_pytorch_custom_datasets/). You should have train and test `DataLoader`'s ready to use.


```python
# 1. Get data
import requests
import zipfile
from pathlib import Path

# path to datafolder
data_path = Path("/data")
image_path = data_path / "pizza_steak_sushi"

if image_path.is_dir():
  print(f"{image_path} already exists. Skipping download")
else:
  print(f"{image_path} does not exist, creating one...")
  image_path.mkdir(parents=True, exist_ok=True)

with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
  request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/refs/heads/main/data/pizza_steak_sushi.zip")
  print(f"Downloading pizza, steak, sushi data...")
  f.write(request.content)

# unzip data
with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
  print("Unzipping pizza, steak and sushi data...")
  zip_ref.extractall(image_path)
```

    /data/pizza_steak_sushi does not exist, creating one...
    Downloading pizza, steak, sushi data...
    Unzipping pizza, steak and sushi data...



```python
# 2. Become one with the data
import os
def walk_through_dir(dir_path):
  """Walks through dir_path returning file counts of its contents."""
  for dirpath, dirnames, filenames in os.walk(dir_path):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

walk_through_dir(image_path)
```

    There are 2 directories and 0 images in '/data/pizza_steak_sushi'.
    There are 3 directories and 0 images in '/data/pizza_steak_sushi/test'.
    There are 0 directories and 25 images in '/data/pizza_steak_sushi/test/pizza'.
    There are 0 directories and 31 images in '/data/pizza_steak_sushi/test/sushi'.
    There are 0 directories and 19 images in '/data/pizza_steak_sushi/test/steak'.
    There are 3 directories and 0 images in '/data/pizza_steak_sushi/train'.
    There are 0 directories and 78 images in '/data/pizza_steak_sushi/train/pizza'.
    There are 0 directories and 72 images in '/data/pizza_steak_sushi/train/sushi'.
    There are 0 directories and 75 images in '/data/pizza_steak_sushi/train/steak'.



```python
# Setup train and testing paths
train_dir = image_path / "train"
test_dir = image_path / "test"

train_dir, test_dir
```




    (PosixPath('/data/pizza_steak_sushi/train'),
     PosixPath('/data/pizza_steak_sushi/test'))




```python
# Visualize an image
import random
from PIL import Image

image_path_list = list(image_path.glob("*/*/*.jpg"))
random_image_path = random.choice(image_path_list)
image_class = random_image_path.stem
img = Image.open(random_image_path)

print(f"Random image path: {random_image_path}")
print(f"Image class: {image_class}")
print(f"Image height: {img.height}")
print(f"Image width: {img.width}")
img
```

    Random image path: /data/pizza_steak_sushi/train/steak/2087958.jpg
    Image class: 2087958
    Image height: 512
    Image width: 512





    
![png](04_pytorch_custom_datasets_exercises_files/04_pytorch_custom_datasets_exercises_10_1.png)
    




```python
# Do the image visualization with matplotlib
import numpy as np
import matplotlib.pyplot as plt

img_as_array = np.asarray(img)

plt.figure(figsize=(10, 7))
plt.imshow(img_as_array)
plt.title(f"Image class: {image_class} | Image shape: {img_as_array.shape} - [height, width, color_channels]")
plt.axis(False);
```


    
![png](04_pytorch_custom_datasets_exercises_files/04_pytorch_custom_datasets_exercises_11_0.png)
    


We've got some images in our folders.

Now we need to make them compatible with PyTorch by:
1. Transform the data into tensors.
2. Turn the tensor data into a `torch.utils.data.Dataset` and later a `torch.utils.data.DataLoader`.


```python
# 3.1 Transforming data with torchvision.transforms
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
```


```python
# Write transform for turning images into tensors
data_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor()
])

data_transform(img)
```




    tensor([[[0.8118, 0.8118, 0.8078,  ..., 0.6471, 0.6314, 0.6235],
             [0.8078, 0.8157, 0.8118,  ..., 0.6392, 0.6353, 0.6314],
             [0.8235, 0.8196, 0.8118,  ..., 0.6431, 0.6431, 0.6431],
             ...,
             [0.7843, 0.7922, 0.7961,  ..., 0.8078, 0.8039, 0.8000],
             [0.7922, 0.7922, 0.7961,  ..., 0.8118, 0.8078, 0.8000],
             [0.7882, 0.7882, 0.7922,  ..., 0.8078, 0.8000, 0.8039]],
    
            [[0.6745, 0.6745, 0.6706,  ..., 0.4627, 0.4588, 0.4549],
             [0.6706, 0.6745, 0.6784,  ..., 0.4510, 0.4549, 0.4549],
             [0.6745, 0.6745, 0.6745,  ..., 0.4510, 0.4667, 0.4667],
             ...,
             [0.6275, 0.6275, 0.6314,  ..., 0.6431, 0.6471, 0.6431],
             [0.6275, 0.6275, 0.6275,  ..., 0.6510, 0.6510, 0.6510],
             [0.6275, 0.6392, 0.6275,  ..., 0.6588, 0.6549, 0.6510]],
    
            [[0.5216, 0.5137, 0.5137,  ..., 0.3020, 0.2941, 0.2745],
             [0.5098, 0.5098, 0.5176,  ..., 0.2941, 0.2902, 0.2784],
             [0.5137, 0.5098, 0.5098,  ..., 0.2902, 0.3020, 0.2980],
             ...,
             [0.4784, 0.4863, 0.4941,  ..., 0.4980, 0.5059, 0.4980],
             [0.4863, 0.4863, 0.4980,  ..., 0.4902, 0.5059, 0.5020],
             [0.4824, 0.4784, 0.4902,  ..., 0.4902, 0.4902, 0.4902]]])




```python
# Write a function to plot transformed images
def plot_transformed_images(image_paths: list, transform, n=3, seed=None):
  """
  Selects n random images from image_paths, transforms and then plots the original vs the transformed version.
  """
  if seed:
    random.seed(seed)
  random_image_paths = random.sample(image_paths, k=n)
  for image_path in random_image_paths:
    with Image.open(image_path) as f:
      fig, ax = plt.subplots(nrows=1, ncols=2)
      ax[0].imshow(f)
      ax[0].set_title(f"Original\nSize: {f.size}")
      ax[0].axis(False)

      transformed_image = transform(f).permute(1, 2, 0)
      ax[1].imshow(transformed_image)
      ax[1].set_title(f"Transformed\nShape: {transformed_image.shape}")
      ax[1].axis(False)

      fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)

plot_transformed_images(image_paths=image_path_list,
                        transform=data_transform,
                        seed=42)
```


    
![png](04_pytorch_custom_datasets_exercises_files/04_pytorch_custom_datasets_exercises_15_0.png)
    



    
![png](04_pytorch_custom_datasets_exercises_files/04_pytorch_custom_datasets_exercises_15_1.png)
    



    
![png](04_pytorch_custom_datasets_exercises_files/04_pytorch_custom_datasets_exercises_15_2.png)
    


### Load image data using `ImageFolder`


```python
# Use ImageFolder to create dataset(s)
from torchvision import datasets
train_data = datasets.ImageFolder(root=train_dir,
                                  transform=data_transform)

test_data = datasets.ImageFolder(root=test_dir,
                                 transform=data_transform)

train_data, test_data
```




    (Dataset ImageFolder
         Number of datapoints: 225
         Root location: /data/pizza_steak_sushi/train
         StandardTransform
     Transform: Compose(
                    Resize(size=(64, 64), interpolation=bilinear, max_size=None, antialias=True)
                    ToTensor()
                ),
     Dataset ImageFolder
         Number of datapoints: 75
         Root location: /data/pizza_steak_sushi/test
         StandardTransform
     Transform: Compose(
                    Resize(size=(64, 64), interpolation=bilinear, max_size=None, antialias=True)
                    ToTensor()
                ))




```python
# Get class names as a list
class_names = train_data.classes
class_names
```




    ['pizza', 'steak', 'sushi']




```python
# Can also get class names as a dict
class_dict = train_data.class_to_idx
class_dict
```




    {'pizza': 0, 'steak': 1, 'sushi': 2}




```python
# Check the lengths of each dataset
len(train_data), len(test_data)
```




    (225, 75)




```python
# Turn train and test Datasets into DataLoaders
BATCH_SIZE = 1
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              num_workers=1,
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
                             num_workers=1,
                             shuffle=False)

train_dataloader, test_dataloader
```




    (<torch.utils.data.dataloader.DataLoader at 0x7a86bf0fea40>,
     <torch.utils.data.dataloader.DataLoader at 0x7a86bf0fdc30>)




```python
# How many batches of images are in our data loaders?
len(train_dataloader), len(test_dataloader)
```




    (225, 75)



## 3. Recreate `model_0` we built in section 7 of notebook 04.


```python
class TinyVGG(nn.Module):
  def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
    super().__init__()
    self.conv_block_1 = nn.Sequential(
        nn.Conv2d(in_channels=input_shape,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=0),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=0),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,
                     stride=2)
    )
    self.conv_block_2 = nn.Sequential(
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=0),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=0),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,
                     stride=2)
    )
    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=hidden_units*13*13,
                  out_features=output_shape)
    )

  def forward(self, x):
    return self.classifier(self.conv_block_2(self.conv_block_1(x)))
```


```python
torch.manual_seed(42)
model_0 = TinyVGG(input_shape=3,
                  hidden_units=10,
                  output_shape=len(class_names)).to(device)

model_0
```




    TinyVGG(
      (conv_block_1): Sequential(
        (0): Conv2d(3, 10, kernel_size=(3, 3), stride=(1, 1))
        (1): ReLU()
        (2): Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1))
        (3): ReLU()
        (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (conv_block_2): Sequential(
        (0): Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1))
        (1): ReLU()
        (2): Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1))
        (3): ReLU()
        (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (classifier): Sequential(
        (0): Flatten(start_dim=1, end_dim=-1)
        (1): Linear(in_features=1690, out_features=3, bias=True)
      )
    )



## 4. Create training and testing functions for `model_0`.


```python
def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer):

  # Put the model in train mode
  model.train()

  # Setup train loss and train accuracy values
  train_loss, train_acc = 0, 0

  # Loop through data loader and data batches
  for batch, (X, y) in enumerate(dataloader):
    # Send data to target device
    X, y = X.to(device), y.to(device)

    # 1. Forward pass
    y_pred = model(X)

    # 2. Calculate and accumulate loss
    loss = loss_fn(y_pred, y)
    train_loss += loss.item()

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backward
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    # Calculate and accumualte accuracy metric across all batches
    y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
    train_acc += (y_pred_class==y).sum().item()/len(y_pred)

  # Adjust metrics to get average loss and average accuracy per batch
  train_loss = train_loss / len(dataloader)
  train_acc = train_acc / len(dataloader)
  return train_loss, train_acc
```


```python
def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module):

  # Put model in eval mode
  model.eval()

  # Setup the test loss and test accuracy values
  test_loss, test_acc = 0, 0

  # Turn on inference context manager
  with torch.inference_mode():
    # Loop through DataLoader batches
    for batch, (X, y) in enumerate(dataloader):
      # Send data to target device
      X, y = X.to(device), y.to(device)

      # 1. Forward pass
      test_pred_logits = model(X)

      # 2. Calculuate and accumulate loss
      loss = loss_fn(test_pred_logits, y)
      test_loss += loss.item()

      # Calculate and accumulate accuracy
      test_pred_labels = test_pred_logits.argmax(dim=1)
      test_acc += ((test_pred_labels==y).sum().item()/len(test_pred_labels))


  # Adjust metrics to get average loss and accuracy per batch
  test_loss = test_loss / len(dataloader)
  test_acc = test_acc / len(dataloader)
  return test_loss, test_acc
```


```python
from tqdm.auto import tqdm

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5):

  # Create results dictionary
  results = {"train_loss": [],
             "train_acc": [],
             "test_loss": [],
             "test_acc": []}

  # Loop through the training and testing steps for a number of epochs
  for epoch in tqdm(range(epochs)):
    # Train step
    train_loss, train_acc = train_step(model=model,
                                       dataloader=train_dataloader,
                                       loss_fn=loss_fn,
                                       optimizer=optimizer)
    # Test step
    test_loss, test_acc = test_step(model=model,
                                    dataloader=test_dataloader,
                                    loss_fn=loss_fn)

    # Print out what's happening
    print(f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}"
    )

    # Update the results dictionary
    results["train_loss"].append(train_loss)
    results["train_acc"].append(train_acc)
    results["test_loss"].append(test_loss)
    results["test_acc"].append(test_acc)

  # Return the results dictionary
  return results
```

## 5. Try training the model you made in exercise 3 for 5, 20 and 50 epochs, what happens to the results?
* Use `torch.optim.Adam()` with a learning rate of 0.001 as the optimizer.


```python
# Train for 5 epochs
torch.manual_seed(42)
torch.cuda.manual_seed(42)

model_0 = TinyVGG(input_shape=3,
                  hidden_units=10,
                  output_shape=len(class_names)).to(device)

NUM_EPOCHS = 5

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_0.parameters(),
                             lr=0.001)

model_0_results = train(model=model_0,
                        train_dataloader=train_dataloader,
                        test_dataloader=test_dataloader,
                        loss_fn=loss_fn,
                        optimizer=optimizer,
                        epochs=NUM_EPOCHS)
```


      0%|          | 0/5 [00:00<?, ?it/s]


    Epoch: 1 | train_loss: 1.0923 | train_acc: 0.3911 | test_loss: 1.0726 | test_acc: 0.4133
    Epoch: 2 | train_loss: 1.0264 | train_acc: 0.5156 | test_loss: 1.0161 | test_acc: 0.4267
    Epoch: 3 | train_loss: 0.9602 | train_acc: 0.5200 | test_loss: 0.9910 | test_acc: 0.4667
    Epoch: 4 | train_loss: 0.9163 | train_acc: 0.5644 | test_loss: 0.9770 | test_acc: 0.4267
    Epoch: 5 | train_loss: 0.8869 | train_acc: 0.6000 | test_loss: 0.9822 | test_acc: 0.4933



```python
# Train for 20 epochs
torch.manual_seed(42)
torch.cuda.manual_seed(42)

model_1 = TinyVGG(input_shape=3,
                  hidden_units=10,
                  output_shape=len(class_names)).to(device)

NUM_EPOCHS = 20

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_1.parameters(),
                             lr=0.001)

model_1_results = train(model=model_1,
                        train_dataloader=train_dataloader,
                        test_dataloader=test_dataloader,
                        loss_fn=loss_fn,
                        optimizer=optimizer,
                        epochs=NUM_EPOCHS)
```


      0%|          | 0/20 [00:00<?, ?it/s]


    Epoch: 1 | train_loss: 1.0923 | train_acc: 0.3911 | test_loss: 1.0726 | test_acc: 0.4133
    Epoch: 2 | train_loss: 1.0264 | train_acc: 0.5156 | test_loss: 1.0161 | test_acc: 0.4267
    Epoch: 3 | train_loss: 0.9601 | train_acc: 0.5200 | test_loss: 0.9916 | test_acc: 0.4667
    Epoch: 4 | train_loss: 0.9155 | train_acc: 0.5644 | test_loss: 0.9758 | test_acc: 0.4267
    Epoch: 5 | train_loss: 0.8890 | train_acc: 0.6044 | test_loss: 0.9814 | test_acc: 0.4800
    Epoch: 6 | train_loss: 0.8377 | train_acc: 0.6178 | test_loss: 1.0055 | test_acc: 0.4933
    Epoch: 7 | train_loss: 0.7853 | train_acc: 0.6711 | test_loss: 1.1314 | test_acc: 0.4000
    Epoch: 8 | train_loss: 0.6944 | train_acc: 0.6978 | test_loss: 1.0330 | test_acc: 0.4800
    Epoch: 9 | train_loss: 0.5858 | train_acc: 0.7822 | test_loss: 1.2654 | test_acc: 0.4267
    Epoch: 10 | train_loss: 0.4522 | train_acc: 0.8089 | test_loss: 1.3857 | test_acc: 0.4400
    Epoch: 11 | train_loss: 0.3387 | train_acc: 0.8800 | test_loss: 1.7953 | test_acc: 0.4400
    Epoch: 12 | train_loss: 0.2916 | train_acc: 0.8933 | test_loss: 1.8747 | test_acc: 0.4133
    Epoch: 13 | train_loss: 0.2033 | train_acc: 0.9156 | test_loss: 2.0878 | test_acc: 0.4400
    Epoch: 14 | train_loss: 0.1666 | train_acc: 0.9600 | test_loss: 2.8420 | test_acc: 0.4000
    Epoch: 15 | train_loss: 0.0481 | train_acc: 0.9911 | test_loss: 3.4573 | test_acc: 0.4667
    Epoch: 16 | train_loss: 0.0396 | train_acc: 0.9911 | test_loss: 3.6603 | test_acc: 0.4267
    Epoch: 17 | train_loss: 0.0096 | train_acc: 1.0000 | test_loss: 4.2559 | test_acc: 0.4133
    Epoch: 18 | train_loss: 0.0039 | train_acc: 1.0000 | test_loss: 4.3974 | test_acc: 0.4267
    Epoch: 19 | train_loss: 0.0023 | train_acc: 1.0000 | test_loss: 4.6287 | test_acc: 0.4133
    Epoch: 20 | train_loss: 0.0015 | train_acc: 1.0000 | test_loss: 4.8020 | test_acc: 0.4133



```python
# Train for 50 epochs
torch.manual_seed(42)
torch.cuda.manual_seed(42)

model_2 = TinyVGG(input_shape=3,
                  hidden_units=10,
                  output_shape=len(class_names)).to(device)

NUM_EPOCHS = 50

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_2.parameters(),
                             lr=0.001)

model_2_results = train(model=model_2,
                        train_dataloader=train_dataloader,
                        test_dataloader=test_dataloader,
                        loss_fn=loss_fn,
                        optimizer=optimizer,
                        epochs=NUM_EPOCHS)
```


      0%|          | 0/50 [00:00<?, ?it/s]


    Epoch: 1 | train_loss: 1.0922 | train_acc: 0.3911 | test_loss: 1.0730 | test_acc: 0.4133
    Epoch: 2 | train_loss: 1.0263 | train_acc: 0.5111 | test_loss: 1.0177 | test_acc: 0.4267
    Epoch: 3 | train_loss: 0.9623 | train_acc: 0.5156 | test_loss: 0.9918 | test_acc: 0.4533
    Epoch: 4 | train_loss: 0.9180 | train_acc: 0.5644 | test_loss: 0.9830 | test_acc: 0.4400
    Epoch: 5 | train_loss: 0.8919 | train_acc: 0.5956 | test_loss: 0.9797 | test_acc: 0.4800
    Epoch: 6 | train_loss: 0.8405 | train_acc: 0.6222 | test_loss: 1.0190 | test_acc: 0.5333
    Epoch: 7 | train_loss: 0.8063 | train_acc: 0.6489 | test_loss: 1.0334 | test_acc: 0.4133
    Epoch: 8 | train_loss: 0.7621 | train_acc: 0.6711 | test_loss: 1.0092 | test_acc: 0.5067
    Epoch: 9 | train_loss: 0.6718 | train_acc: 0.7422 | test_loss: 1.2770 | test_acc: 0.5067
    Epoch: 10 | train_loss: 0.5529 | train_acc: 0.7911 | test_loss: 1.2803 | test_acc: 0.4533
    Epoch: 11 | train_loss: 0.4341 | train_acc: 0.8533 | test_loss: 1.6123 | test_acc: 0.4533
    Epoch: 12 | train_loss: 0.3799 | train_acc: 0.8489 | test_loss: 1.5082 | test_acc: 0.4667
    Epoch: 13 | train_loss: 0.2898 | train_acc: 0.8978 | test_loss: 1.8473 | test_acc: 0.4533
    Epoch: 14 | train_loss: 0.2372 | train_acc: 0.9289 | test_loss: 2.2145 | test_acc: 0.4800
    Epoch: 15 | train_loss: 0.1375 | train_acc: 0.9556 | test_loss: 2.5776 | test_acc: 0.4800
    Epoch: 16 | train_loss: 0.1228 | train_acc: 0.9556 | test_loss: 2.4386 | test_acc: 0.4667
    Epoch: 17 | train_loss: 0.0926 | train_acc: 0.9600 | test_loss: 2.6927 | test_acc: 0.4800
    Epoch: 18 | train_loss: 0.0736 | train_acc: 0.9733 | test_loss: 2.8344 | test_acc: 0.5067
    Epoch: 19 | train_loss: 0.0649 | train_acc: 0.9778 | test_loss: 4.5856 | test_acc: 0.4267
    Epoch: 20 | train_loss: 0.0518 | train_acc: 0.9778 | test_loss: 3.9949 | test_acc: 0.4267
    Epoch: 21 | train_loss: 0.0828 | train_acc: 0.9644 | test_loss: 3.6983 | test_acc: 0.4267
    Epoch: 22 | train_loss: 0.0459 | train_acc: 0.9822 | test_loss: 4.3096 | test_acc: 0.4400
    Epoch: 23 | train_loss: 0.0209 | train_acc: 0.9911 | test_loss: 4.2077 | test_acc: 0.4667
    Epoch: 24 | train_loss: 0.0073 | train_acc: 1.0000 | test_loss: 4.5461 | test_acc: 0.4400
    Epoch: 25 | train_loss: 0.0033 | train_acc: 1.0000 | test_loss: 4.7687 | test_acc: 0.4533
    Epoch: 26 | train_loss: 0.0019 | train_acc: 1.0000 | test_loss: 4.9486 | test_acc: 0.4400
    Epoch: 27 | train_loss: 0.0013 | train_acc: 1.0000 | test_loss: 5.0631 | test_acc: 0.4400
    Epoch: 28 | train_loss: 0.0009 | train_acc: 1.0000 | test_loss: 5.2536 | test_acc: 0.4400
    Epoch: 29 | train_loss: 0.0007 | train_acc: 1.0000 | test_loss: 5.2889 | test_acc: 0.4400
    Epoch: 30 | train_loss: 0.0006 | train_acc: 1.0000 | test_loss: 5.4083 | test_acc: 0.4400
    Epoch: 31 | train_loss: 0.0005 | train_acc: 1.0000 | test_loss: 5.5611 | test_acc: 0.4400
    Epoch: 32 | train_loss: 0.0004 | train_acc: 1.0000 | test_loss: 5.7413 | test_acc: 0.4400
    Epoch: 33 | train_loss: 0.0003 | train_acc: 1.0000 | test_loss: 5.7963 | test_acc: 0.4400
    Epoch: 34 | train_loss: 0.0002 | train_acc: 1.0000 | test_loss: 5.8993 | test_acc: 0.4400
    Epoch: 35 | train_loss: 0.0002 | train_acc: 1.0000 | test_loss: 5.9619 | test_acc: 0.4400
    Epoch: 36 | train_loss: 0.0002 | train_acc: 1.0000 | test_loss: 6.0571 | test_acc: 0.4400
    Epoch: 37 | train_loss: 0.0001 | train_acc: 1.0000 | test_loss: 6.1313 | test_acc: 0.4400
    Epoch: 38 | train_loss: 0.0001 | train_acc: 1.0000 | test_loss: 6.1917 | test_acc: 0.4400
    Epoch: 39 | train_loss: 0.0001 | train_acc: 1.0000 | test_loss: 6.2818 | test_acc: 0.4400
    Epoch: 40 | train_loss: 0.0001 | train_acc: 1.0000 | test_loss: 6.3486 | test_acc: 0.4400
    Epoch: 41 | train_loss: 0.0001 | train_acc: 1.0000 | test_loss: 6.4157 | test_acc: 0.4400
    Epoch: 42 | train_loss: 0.0001 | train_acc: 1.0000 | test_loss: 6.5184 | test_acc: 0.4400
    Epoch: 43 | train_loss: 0.0001 | train_acc: 1.0000 | test_loss: 6.5731 | test_acc: 0.4400
    Epoch: 44 | train_loss: 0.0000 | train_acc: 1.0000 | test_loss: 6.6545 | test_acc: 0.4400
    Epoch: 45 | train_loss: 0.0000 | train_acc: 1.0000 | test_loss: 6.7256 | test_acc: 0.4400
    Epoch: 46 | train_loss: 0.0000 | train_acc: 1.0000 | test_loss: 6.7845 | test_acc: 0.4400
    Epoch: 47 | train_loss: 0.0000 | train_acc: 1.0000 | test_loss: 6.8361 | test_acc: 0.4400
    Epoch: 48 | train_loss: 0.0000 | train_acc: 1.0000 | test_loss: 6.9059 | test_acc: 0.4400
    Epoch: 49 | train_loss: 0.0000 | train_acc: 1.0000 | test_loss: 6.9910 | test_acc: 0.4400
    Epoch: 50 | train_loss: 0.0000 | train_acc: 1.0000 | test_loss: 7.0610 | test_acc: 0.4400


It looks like our model is starting to overfit towards the end (performing far better on the training data than on the testing data).

In order to fix this, we'd have to introduce ways of preventing overfitting.

## 6. Double the number of hidden units in your model and train it for 20 epochs, what happens to the results?


```python
# Double the number of hidden units and train for 20 epochs
torch.manual_seed(42)
torch.cuda.manual_seed(42)

model_3 = TinyVGG(input_shape=3,
                  hidden_units=20,
                  output_shape=len(class_names)).to(device)

NUM_EPOCHS = 20

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_3.parameters(),
                             lr=0.001)

model_3_results = train(model=model_3,
                        train_dataloader=train_dataloader,
                        test_dataloader=test_dataloader,
                        loss_fn=loss_fn,
                        optimizer=optimizer,
                        epochs=NUM_EPOCHS)
```


      0%|          | 0/20 [00:00<?, ?it/s]


    Epoch: 1 | train_loss: 1.1102 | train_acc: 0.2933 | test_loss: 1.0993 | test_acc: 0.3333
    Epoch: 2 | train_loss: 1.0993 | train_acc: 0.2933 | test_loss: 1.0999 | test_acc: 0.3333
    Epoch: 3 | train_loss: 1.0991 | train_acc: 0.3467 | test_loss: 1.1002 | test_acc: 0.3333
    Epoch: 4 | train_loss: 1.0989 | train_acc: 0.3467 | test_loss: 1.1003 | test_acc: 0.3333
    Epoch: 5 | train_loss: 1.0988 | train_acc: 0.3333 | test_loss: 1.1008 | test_acc: 0.3333
    Epoch: 6 | train_loss: 1.0990 | train_acc: 0.3467 | test_loss: 1.1011 | test_acc: 0.3333
    Epoch: 7 | train_loss: 1.0987 | train_acc: 0.3467 | test_loss: 1.1011 | test_acc: 0.3333
    Epoch: 8 | train_loss: 1.0987 | train_acc: 0.3467 | test_loss: 1.1011 | test_acc: 0.3333
    Epoch: 9 | train_loss: 1.0986 | train_acc: 0.3467 | test_loss: 1.1017 | test_acc: 0.3333
    Epoch: 10 | train_loss: 1.0987 | train_acc: 0.3467 | test_loss: 1.1019 | test_acc: 0.3333
    Epoch: 11 | train_loss: 1.0986 | train_acc: 0.3467 | test_loss: 1.1016 | test_acc: 0.3333
    Epoch: 12 | train_loss: 1.0988 | train_acc: 0.3467 | test_loss: 1.1020 | test_acc: 0.3333
    Epoch: 13 | train_loss: 1.0988 | train_acc: 0.3467 | test_loss: 1.1017 | test_acc: 0.3333
    Epoch: 14 | train_loss: 1.0987 | train_acc: 0.3467 | test_loss: 1.1020 | test_acc: 0.3333
    Epoch: 15 | train_loss: 1.0987 | train_acc: 0.3467 | test_loss: 1.1025 | test_acc: 0.3333
    Epoch: 16 | train_loss: 1.0986 | train_acc: 0.3467 | test_loss: 1.1026 | test_acc: 0.3333
    Epoch: 17 | train_loss: 1.0988 | train_acc: 0.3467 | test_loss: 1.1017 | test_acc: 0.3333
    Epoch: 18 | train_loss: 1.0987 | train_acc: 0.3467 | test_loss: 1.1022 | test_acc: 0.3333
    Epoch: 19 | train_loss: 1.0987 | train_acc: 0.3467 | test_loss: 1.1022 | test_acc: 0.3333
    Epoch: 20 | train_loss: 1.0988 | train_acc: 0.3467 | test_loss: 1.1016 | test_acc: 0.3333


It looks like the model is still overfitting, even when changing the number of hidden units.

To fix this, we'd have to look at ways to prevent overfitting with our model.

## 7. Double the data you're using with your model from step 6 and train it for 20 epochs, what happens to the results?
* **Note:** You can use the [custom data creation notebook](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/extras/04_custom_data_creation.ipynb) to scale up your Food101 dataset.
* You can also find the [already formatted double data (20% instead of 10% subset) dataset on GitHub](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/data/pizza_steak_sushi_20_percent.zip), you will need to write download code like in exercise 2 to get it into this notebook.


```python
# Download 20% data for Pizza/Steak/Sushi from GitHub
import requests
import zipfile
from pathlib import Path

# Setup path to data folder
data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi_20_percent"

# If the image folder doesn't exist, download it and prepare it...
if image_path.is_dir():
    print(f"{image_path} directory exists.")
else:
    print(f"Did not find {image_path} directory, creating one...")
    image_path.mkdir(parents=True, exist_ok=True)

# Download pizza, steak, sushi data
with open(data_path / "pizza_steak_sushi_20_percent.zip", "wb") as f:
    request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi_20_percent.zip")
    print("Downloading pizza, steak, sushi 20% data...")
    f.write(request.content)

# Unzip pizza, steak, sushi data
with zipfile.ZipFile(data_path / "pizza_steak_sushi_20_percent.zip", "r") as zip_ref:
    print("Unzipping pizza, steak, sushi 20% data...")
    zip_ref.extractall(image_path)
```

    data/pizza_steak_sushi_20_percent directory exists.
    Downloading pizza, steak, sushi 20% data...
    Unzipping pizza, steak, sushi 20% data...



```python
# See how many images we have
walk_through_dir(image_path)
```

    There are 2 directories and 0 images in 'data/pizza_steak_sushi_20_percent'.
    There are 3 directories and 0 images in 'data/pizza_steak_sushi_20_percent/test'.
    There are 0 directories and 46 images in 'data/pizza_steak_sushi_20_percent/test/pizza'.
    There are 0 directories and 46 images in 'data/pizza_steak_sushi_20_percent/test/sushi'.
    There are 0 directories and 58 images in 'data/pizza_steak_sushi_20_percent/test/steak'.
    There are 3 directories and 0 images in 'data/pizza_steak_sushi_20_percent/train'.
    There are 0 directories and 154 images in 'data/pizza_steak_sushi_20_percent/train/pizza'.
    There are 0 directories and 150 images in 'data/pizza_steak_sushi_20_percent/train/sushi'.
    There are 0 directories and 146 images in 'data/pizza_steak_sushi_20_percent/train/steak'.


Excellent, we now have double the training and testing images...


```python
# Create the train and test paths
train_data_20_percent_path = image_path / "train"
test_data_20_percent_path = image_path / "test"

train_data_20_percent_path, test_data_20_percent_path
```




    (PosixPath('data/pizza_steak_sushi_20_percent/train'),
     PosixPath('data/pizza_steak_sushi_20_percent/test'))




```python
# Turn the 20 percent datapaths into Datasets and DataLoaders
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

simple_transform = transforms.Compose([
  transforms.Resize((64, 64)),
  transforms.ToTensor()
])

# Create datasets
train_data_20 = datasets.ImageFolder(root=train_data_20_percent_path,
                                     transform=simple_transform)

test_data_20 = datasets.ImageFolder(root=test_data_20_percent_path,
                                     transform=simple_transform)


# Create dataloaders
BATCH_SIZE = 1

train_dataloader_20 = DataLoader(dataset=train_data_20,
                              batch_size=BATCH_SIZE,
                              num_workers=1,
                              shuffle=True)

test_dataloader_20 = DataLoader(dataset=test_data_20,
                             batch_size=BATCH_SIZE,
                             num_workers=1,
                             shuffle=False)
```


```python
# Train a model with increased amount of data
torch.manual_seed(42)
torch.cuda.manual_seed(42)

model_4 = TinyVGG(input_shape=3,
                  hidden_units=20,
                  output_shape=len(class_names)).to(device)

NUM_EPOCHS = 20

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_4.parameters(),
                             lr=0.001)

model_4_results = train(model=model_4,
                        train_dataloader=train_dataloader_20,
                        test_dataloader=test_dataloader_20,
                        loss_fn=loss_fn,
                        optimizer=optimizer,
                        epochs=NUM_EPOCHS)
```


      0%|          | 0/20 [00:00<?, ?it/s]


    Epoch: 1 | train_loss: 1.0970 | train_acc: 0.3756 | test_loss: 0.9609 | test_acc: 0.5800
    Epoch: 2 | train_loss: 1.0192 | train_acc: 0.5089 | test_loss: 1.0931 | test_acc: 0.3867
    Epoch: 3 | train_loss: 0.9488 | train_acc: 0.5244 | test_loss: 0.9496 | test_acc: 0.4667
    Epoch: 4 | train_loss: 0.9030 | train_acc: 0.5978 | test_loss: 0.9243 | test_acc: 0.4867
    Epoch: 5 | train_loss: 0.8870 | train_acc: 0.6156 | test_loss: 0.9147 | test_acc: 0.6000
    Epoch: 6 | train_loss: 0.8527 | train_acc: 0.6089 | test_loss: 0.9239 | test_acc: 0.5667
    Epoch: 7 | train_loss: 0.8089 | train_acc: 0.6533 | test_loss: 0.8989 | test_acc: 0.6133
    Epoch: 8 | train_loss: 0.7549 | train_acc: 0.6756 | test_loss: 0.8939 | test_acc: 0.6067
    Epoch: 9 | train_loss: 0.7154 | train_acc: 0.6978 | test_loss: 1.0133 | test_acc: 0.5267
    Epoch: 10 | train_loss: 0.6079 | train_acc: 0.7622 | test_loss: 0.9608 | test_acc: 0.5400
    Epoch: 11 | train_loss: 0.5281 | train_acc: 0.7867 | test_loss: 1.3195 | test_acc: 0.5067
    Epoch: 12 | train_loss: 0.4748 | train_acc: 0.8067 | test_loss: 1.2857 | test_acc: 0.5333
    Epoch: 13 | train_loss: 0.3884 | train_acc: 0.8489 | test_loss: 1.4144 | test_acc: 0.5133
    Epoch: 14 | train_loss: 0.2670 | train_acc: 0.9089 | test_loss: 1.8023 | test_acc: 0.4933
    Epoch: 15 | train_loss: 0.1945 | train_acc: 0.9244 | test_loss: 2.1829 | test_acc: 0.5067
    Epoch: 16 | train_loss: 0.1532 | train_acc: 0.9556 | test_loss: 2.5215 | test_acc: 0.4600
    Epoch: 17 | train_loss: 0.1425 | train_acc: 0.9511 | test_loss: 2.3725 | test_acc: 0.5067
    Epoch: 18 | train_loss: 0.0684 | train_acc: 0.9822 | test_loss: 3.3134 | test_acc: 0.5067
    Epoch: 19 | train_loss: 0.0556 | train_acc: 0.9844 | test_loss: 4.0109 | test_acc: 0.4867
    Epoch: 20 | train_loss: 0.0987 | train_acc: 0.9622 | test_loss: 3.5666 | test_acc: 0.5067


## 8. Make a prediction on your own custom image of pizza/steak/sushi (you could even download one from the internet) with your trained model from exercise 7 and share your prediction.
* Does the model you trained in exercise 7 get it right?
* If not, what do you think you could do to improve it?


```python
custom_image_path = data_path / "04-pizza-dad.jpg"

if not custom_image_path.is_file():
  with open(custom_image_path, "wb") as f:
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/refs/heads/main/images/04-pizza-dad.jpeg")
    print(f"Downloading {custom_image_path}...")
    f.write(request.content)
else:
  print(f"{custom_image_path} already exists, skipping download...")
```

    Downloading data/04-pizza-dad.jpg...



```python
from torchvision import transforms

custom_img_transform = transforms.Compose([
    transforms.Resize(size=(64, 64))
])
```


```python
import torchvision

def pred_plot_img(model: torch.nn.Module,
                  image_path: str,
                  class_names=None,
                  transform=None,
                  device=device):
  img = torchvision.io.decode_image(custom_image_path).type(torch.float32) / 255

  if transform:
    img = transform(img)

  model.to(device)

  model.eval()
  with torch.inference_mode():
    img = img.unsqueeze(0)
    img_pred = model(img.to(device))

  img_pred_probs = torch.softmax(img_pred, dim=1)
  img_pred_label = torch.argmax(img_pred_probs, dim=1)

  plt.imshow(img.squeeze().permute(1, 2, 0))
  if class_names:
    title = f"Pred: {class_names[img_pred_label.cpu()]} | Prob: {img_pred_probs.max().cpu():.3f}"
  else:
    title = f"Pred: {img_pred_label} | Prob: {img_pred_probs.max().cpu():.3f}"
  plt.title(title)
  plt.axis(False)
```


```python
pred_plot_img(model=model_4,
              image_path=custom_image_path,
              class_names=class_names,
              transform=custom_img_transform,
              device=device)
```


    
![png](04_pytorch_custom_datasets_exercises_files/04_pytorch_custom_datasets_exercises_49_0.png)
    


It predicted much better than the model from chapter 04. But it is also
still overfitting. We could test other optimizers and different learning rates.
