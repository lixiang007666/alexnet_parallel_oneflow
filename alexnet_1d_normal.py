import oneflow as flow
import oneflow.nn as nn
import flowvision
import flowvision.transforms as transforms

from model import alexnet
import time

BATCH_SIZE = 16
EPOCH_NUM = 5

DEVICE = "cuda" if flow.cuda.is_available() else "cpu"
print("Using {} device".format(DEVICE))

training_data = flowvision.datasets.CIFAR10(
    root="data",
    train=True,
    transform=transforms.ToTensor(),
    download=False,
)

train_dataloader = flow.utils.data.DataLoader(
    training_data, BATCH_SIZE, shuffle=True
)

model = alexnet()
print(model)
model = model.to(DEVICE)

loss_fn = nn.CrossEntropyLoss().to(DEVICE)
optimizer = flow.optim.SGD(model.parameters(), lr=1e-3)

start_t = time.time()

for t in range(EPOCH_NUM):
    print(f"Epoch {t+1}\n-------------------------------")
    size = len(train_dataloader.dataset)
    for batch, (x, y) in enumerate(train_dataloader):
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        # Compute prediction error
        pred = model(x)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        current = batch * BATCH_SIZE
        if batch % 5 == 0:
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        if batch % 50 == 0:
            cur_used = flow._oneflow_internal.GetCUDAMemoryUsed()
            print("cur_used:",cur_used,"MB!")

end_t = time.time()
print("train time : {}".format(end_t - start_t))
