import torch
import torchmetrics
import torch.nn as nn
import pytorch_lightning as pl
from torch_optimizer import MADGRAD


def criterion_barlow(z1, z2, lambd=0.0051):
    # empirical cross-correlation matrix
    c = z1.T @ z2

    # sum the cross-correlation matrix between all gpus
    c.div_(z1.shape[0])
    # torch.distributed.all_reduce(c)

    def off_diagonal(x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = off_diagonal(c).pow_(2).sum()
    loss = on_diag + lambd * off_diag
    return loss


class DLBModel(pl.LightningModule):
    """Self-Distillation from the Last Mini-Batch for Consistency Regularization
    """    
    def __init__(self, cfg, model):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.fc = nn.Linear(1000, cfg.num_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.criterion_barlow = criterion_barlow
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.lr = cfg.lr

        self._init_last_batch_memory()

    def _init_last_batch_memory(self):
        self.last_images = None
        self.last_logits = None
        self.last_embds = None

    def forward(self, x):
        embd = self.model(x)
        logit = self.fc(embd)
        return logit, embd

    def shared_step(self, batch, batch_idx):
        images, target = batch
        batch_size = images.shape[0]

        if self.last_images is not None:
            images = torch.cat([images, self.last_images], dim=0)

            logit, embd = self(images)
            
            logit, logit_last = logit[:batch_size], logit[batch_size:]
            embd, embd_last = embd[:batch_size], embd[batch_size:]

            loss_org = self.criterion(logit, target)
            loss_dlb = self.criterion(logit_last, self.last_logits)
            loss_barlow = self.criterion_barlow(embd_last, self.last_embds)*1e-3
        else:
            logit, embd = self(images)
            loss_org = self.criterion(logit, target)
            loss_dlb = torch.tensor(0.0)
            loss_barlow = torch.tensor(0.0)

        # Update last
        self.last_images = images[:batch_size].detach()
        self.last_logits = torch.softmax(logit.detach(), dim=1)
        self.last_embds = embd.detach()

        loss = loss_org + loss_dlb + loss_barlow

        return dict(
            loss=loss,
            logit=logit.detach(),
            target=target,
            logs=dict(
                loss_org=loss_org.detach(),
                loss_dlb=loss_dlb.detach(),
                loss_barlow=loss_barlow.detach(),
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
    
