# alexnet_parallel_oneflow
 A distributed parallel implementation of AlexNet, including DP, TP and PP. The dataset uses a small CIFAR10.
 
## Download dataset
```
training_data = flowvision.datasets.CIFAR10(
    root="data",
    train=True,
    transform=transforms.ToTensor(),
    download=False,
)
```
If you want to test with a larger dataset, the OFRecord code for the ImageNet dataset is also provided. See: [alexnet_1d_ofrecord](https://github.com/lixiang007666/alexnet_parallel_oneflow/blob/main/alexnet_1d_ofrecord.py).

## Result
**BATCH_SIZE = 128**
|  | Training time (s) | CUDAMemoryUsed (MB)  |
|--|--|--|
| 1d | 96.46 | GPU0: 2158.0 |
| dp | 109.48 | GPU0: 1944.0, GPU1: 1954.0 |
| tp | 104.03 | GPU0: 1835.0, GPU1: 1843.0 |
| pp | 104.26 | GPU0: 2020.0, GPU1: 1192.0 |

NOTE: Among them, the tensor parallel only splits the Linear layer. See the code for details on the tensor parallel.
