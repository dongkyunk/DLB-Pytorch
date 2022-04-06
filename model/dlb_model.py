import torch
import torchmetrics
import torch.nn as nn
import pytorch_lightning as pl
from torch_optimizer import MADGRAD


class DLBModel(pl.LightningModule):
    """Self-Distillation from the Last Mini-Batch for Consistency Regularization
    """    
    def __init__(self, cfg, model):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.fc = nn.Linear(1000, cfg.num_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.lr = cfg.lr

        self._init_last_batch_memory()

    def _init_last_batch_memory(self):
        self.last_images = None
        self.last_logits = None

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        return x

    def shared_step(self, batch, batch_idx):
        images, target = batch
        batch_size = images.shape[0]

        if self.last_images is not None:
            images = torch.cat([images, self.last_images], dim=0)

            logit = self(images)
            
            logit, logit_last = logit[:batch_size], logit[batch_size:]

            loss_org = self.criterion(logit, target)
            loss_dlb = self.criterion(logit_last, self.last_logits)
        else:
            logit = self(images)
            loss_org = self.criterion(logit, target)
            loss_dlb = torch.tensor(0.0)

        # Update last
        self.last_images = images[:batch_size].detach()
        self.last_logits = torch.softmax(logit.detach(), dim=1)

        loss = loss_org + loss_dlb

        return dict(
            loss=loss,
            logit=logit.detach(),
            target=target,
            logs=dict(
                loss_org=loss_org.detach(),
                loss_dlb=loss_dlb.detach(),
            )
        )
    def on_train_epoch_end(self):
        self._init_last_batch_memory()

    
    def on_validation_end(self):
        self._init_last_batch_memory()

    def training_step(self, batch, batch_idx):
        outputs = self.shared_step(batch, batch_idx)
        self.train_acc(outputs['logit'], outputs['target'])
        outputs['logs']['train_acc'] = self.train_acc
        self.log_dict(outputs['logs'], prog_bar=True)
        return outputs

    def validation_step(self, batch, batch_idx):
        outputs = self.shared_step(batch, batch_idx)
        self.val_acc(outputs['logit'], outputs['target'])
        outputs['logs']['val_acc'] = self.val_acc
        self.log_dict(outputs['logs'], prog_bar=True)

    def configure_optimizers(self):
        optimizer = MADGRAD(self.parameters(), lr=self.lr)
        return optimizer
    
