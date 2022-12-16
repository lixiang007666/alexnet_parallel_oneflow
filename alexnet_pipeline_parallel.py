import oneflow as flow
import oneflow.nn as nn
import flowvision
import flowvision.transforms as transforms

from model import alexnet
import time

BATCH_SIZE = 16
EPOCH_NUM = 5

BROADCAST = [flow.sbp.broadcast]
P0 = flow.placement("cuda", ranks=[0])
P1 = flow.placement("cuda", ranks=[1])

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


class Stage0Module(flow.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = alexnet().layer1

    def forward(self, x):
        out = self.layer1(x)
        return out

class Stage1Module(flow.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer2 = alexnet().layer2

    def forward(self, x):
        out = self.layer2(x)
        return out


class PipelineModule(flow.nn.Module):
    def __init__(self):
        super().__init__()
        self.m_stage0 = Stage0Module()
        self.m_stage1 = Stage1Module()

        self.m_stage0.to_global(placement=P0, sbp=BROADCAST)
        self.m_stage1.to_global(placement=P1, sbp=BROADCAST)

    def forward(self, x):
        out_stage0 = self.m_stage0(x)
        in_stage1 = out_stage0.to_global(placement=P1, sbp=BROADCAST)
        out_stage1 = self.m_stage1(in_stage1)
        return out_stage1

module_pipeline = PipelineModule()
sgd = flow.optim.SGD(module_pipeline.parameters(), lr=0.001)

class PipelineGraph(flow.nn.Graph):
    def __init__(self):
        super().__init__()
        self.module_pipeline = module_pipeline
        self.module_pipeline.m_stage0.to(nn.graph.GraphModule).set_stage(stage_id=0, placement=P0)
        self.module_pipeline.m_stage1.to(nn.graph.GraphModule).set_stage(stage_id=1, placement=P1)
        self.loss_fn = flow.nn.CrossEntropyLoss()
        self.config.set_gradient_accumulation_steps(2)
        self.add_optimizer(sgd)

    def build(self, x, y):
        out = self.module_pipeline(x)
        loss = self.loss_fn(out, y)
        loss.backward()
        return loss

graph = PipelineGraph()

start_t = time.time()

for t in range(EPOCH_NUM):
    print(f"Epoch {t+1}\n-------------------------------")
    size = len(train_dataloader.dataset)
    for batch, (x, y) in enumerate(train_dataloader):
        x = x.to_global(P0, BROADCAST)
        y = y.to_global(P1, BROADCAST)

        loss = graph(x, y)

        current = batch * BATCH_SIZE
        if batch % 5 == 0:
            print(f"loss: {loss.to_local()}  [{current:>5d}/{size:>5d}]")

        if batch % 50 == 0:
            cur_used = flow._oneflow_internal.GetCUDAMemoryUsed()
            print("cur_used:",cur_used,"MB!")

end_t = time.time()
print("train time : {}".format(end_t - start_t))

