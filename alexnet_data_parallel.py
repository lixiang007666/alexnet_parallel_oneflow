import oneflow as flow
import oneflow.nn as nn
import flowvision
import flowvision.transforms as transforms

from model import alexnet
import time

BATCH_SIZE = 16
EPOCH_NUM = 5

PLACEMENT = flow.placement("cuda", [0,1])
S0 = flow.sbp.split(0)
B = flow.sbp.broadcast

DEVICE = "cuda" if flow.cuda.is_available() else "cpu"
print("Using {} device".format(DEVICE))

training_data = flowvision.datasets.CIFAR10(
    root="data",
    train=True,
    transform=transforms.ToTensor(),
    download=False,
)

train_dataloader = flow.utils.data.DataLoader(
    training_data, BATCH_SIZE, shuffle=True, drop_last=True
)

model = alexnet()
print(model)
model.train()
model = model.to_global(placement=PLACEMENT, sbp=B)

loss_fn = nn.CrossEntropyLoss().to(DEVICE)
optimizer = flow.optim.SGD(model.parameters(), lr=1e-3)

class AlexNetGraph(flow.nn.Graph):
    def __init__(self):
        super().__init__()
        self.alexnet = model
        self.cross_entropy = loss_fn
        self.add_optimizer(optimizer)

    def build(self,image,label):
        y_pred = self.alexnet(image)
        loss = self.cross_entropy(y_pred, label)
        loss.backward()
        return loss

graph = AlexNetGraph()

start_t = time.time()

for t in range(EPOCH_NUM):
    print(f"Epoch {t+1}\n-------------------------------")
    size = len(train_dataloader.dataset)
    for batch, (x, y) in enumerate(train_dataloader):
        x = x.to_global(placement=PLACEMENT, sbp=S0)
        y = y.to_global(placement=PLACEMENT, sbp=S0)

        loss = graph(x, y)

        current = batch * BATCH_SIZE
        if batch % 5 == 0:
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        if batch % 50 == 0:
            cur_used = flow._oneflow_internal.GetCUDAMemoryUsed()
            print("cur_used:",cur_used,"MB!")

end_t = time.time()
print("train time : {}".format(end_t - start_t))
