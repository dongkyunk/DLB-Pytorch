import torch
import torchvision
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning import Trainer
from model.dlb_model import DLBModel
from config import Config
from dataset.transform import train_transform, test_transform

seed_everything(Config.seed)

backbone = torchvision.models.vgg16(pretrained=False, progress=True)

dlb_model = DLBModel(Config, backbone)

trainer = Trainer(gpus=1, max_epochs=Config.epochs)


trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=Config.train_batch_size,
                                          shuffle=True, num_workers=Config.num_workers)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=Config.val_batch_size,
                                         shuffle=False, num_workers=Config.num_workers)

trainer.fit(dlb_model, trainloader, testloader)