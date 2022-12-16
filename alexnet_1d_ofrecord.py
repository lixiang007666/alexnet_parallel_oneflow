import oneflow as flow
import oneflow.nn as nn
import flowvision
import flowvision.transforms as transforms

from ofrecord_data_utils import OFRecordDataLoader
import numpy as np
import time

BATCH_SIZE= 128
EPOCH_NUM = 3

PLACEMENT = flow.placement("cuda", [0,1])
S0 = flow.sbp.split(0)
B = flow.sbp.broadcast

DEVICE = "cuda" if flow.cuda.is_available() else "cpu"
print("Using {} device".format(DEVICE))

train_dataloader = OFRecordDataLoader(
            ofrecord_root="/dataset/imagenette/ofrecord",
            mode="train",
            dataset_size=4000,
            batch_size=BATCH_SIZE,
        )

val_data_loader = OFRecordDataLoader(
            ofrecord_root="/dataset/imagenette/ofrecord",
            mode="val",
            dataset_size=400,
            batch_size=BATCH_SIZE,
)


model = flowvision.models.alexnet(pretrained=False, progress=True).to(DEVICE)

print(model)
loss_fn = nn.CrossEntropyLoss().to(DEVICE)
optimizer = flow.optim.SGD(model.parameters(), lr=1e-3)


class AlexNetGraph(flow.nn.Graph):
    def __init__(self):
        super().__init__()
        self.train_data_loader = train_dataloader
        self.alexnet = model
        self.cross_entropy = loss_fn
        self.add_optimizer(optimizer)

    def build(self,image,label):
        logits = self.alexnet(image)
        loss = self.cross_entropy(logits, label)
        loss.backward()
        return loss

alexnet_graph = AlexNetGraph()

class AlexNetEvalGraph(flow.nn.Graph):
    def __init__(self):
        super().__init__()
        self.val_data_loader = val_data_loader
        self.alexnet = model

    def build(self,image):
        with flow.no_grad():
            logits = self.alexnet(image)
            predictions = logits.softmax()
        return predictions, label

alexnet_eval_graph = AlexNetEvalGraph()

of_losses = []
all_samples = len(val_data_loader) * BATCH_SIZE
print_interval = 10

start_t = time.time()

for epoch in range(EPOCH_NUM):
    model.train()

    for b in range(len(train_dataloader)):
        # oneflow graph train
        image, label = train_dataloader()
        image = image.to(DEVICE)
        label = label.to(DEVICE)
        loss = alexnet_graph(image,label)
        if b % print_interval == 0:
            l = loss.numpy()
            of_losses.append(l)
            print(
                "epoch {} train iter {} oneflow loss {}".format(
                    epoch, b, l
                )
            )

end_t = time.time()
print("train time : {}".format(end_t - start_t))

    # print("epoch %d train done, start validation" % epoch)

    # model.eval()
    # correct_of = 0.0
    # for b in range(len(val_data_loader)):
    #     image, label = val_data_loader()

    #     start_t = time.time()
    #     image = image.to(DEVICE)
    #     predictions, label = alexnet_eval_graph(image)
    #     of_predictions = predictions.numpy()
    #     clsidxs = np.argmax(of_predictions, axis=1)

    #     label_nd = label.numpy()
    #     for i in range(BATCH_SIZE):
    #         if clsidxs[i] == label_nd[i]:
    #             correct_of += 1
    #     end_t = time.time()

    # print("epoch %d, oneflow top1 val acc: %f" % (epoch, correct_of / all_samples))
