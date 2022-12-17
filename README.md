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
If you want to test with a larger dataset, the OFRecord code for the imagenet dataset is also provided. See: [alexnet_1d_ofrecord](https://github.com/lixiang007666/alexnet_parallel_oneflow/blob/main/alexnet_1d_ofrecord.py)

