from typing import Any
import yaml
import torch
import torch.nn as nn
from torch.optim import Adam
from torchmetrics import F1Score, Precision, Recall, MetricCollection, Accuracy
from lightning.pytorch import LightningModule

all_labels = []
all_preds = []

class PhoBERTModel(LightningModule):
    def __init__(self, model, num_labels:int, loss_weight=None):
        super(PhoBERTModel, self).__init__()
        self.model = model
        self.train_loss_fn = nn.CrossEntropyLoss(weight=loss_weight)
        self.loss_fn = nn.CrossEntropyLoss()
        
        self.test_metrics =  MetricCollection([
          Accuracy(task = 'multiclass',average="weighted", num_classes=num_labels),
          Precision(task = 'multiclass', average="weighted", num_classes=num_labels),
          Recall(task = 'multiclass',average="weighted", num_classes=num_labels),
          F1Score(task = 'multiclass',average="weighted", num_classes=num_labels)])
        self.val_acc_fn = Accuracy(task = 'multiclass',average="weighted", num_classes=num_labels)

    def forward(self, input_ids, attn_mask):
        return self.model(input_ids, attn_mask)

    def training_step(self, batch):
        input_ids, attn_mask, label = batch

        logits = self.model(input_ids, attn_mask)
        pred = torch.nn.functional.log_softmax(logits, dim=1)

        loss = self.train_loss_fn(pred, label)

        self.log("train_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        return loss
    def predict_step(self, batch) :
        input_ids, attn_mask, label = batch

        logits = self.model(input_ids, attn_mask)
        pred = torch.nn.functional.log_softmax(logits, dim=1)
        pred = torch.argmax(pred, dim=1)
    
        return pred
    def test_step(self, batch, batch_idx):
        input_ids, attn_mask, label = batch

        logits = self.model(input_ids, attn_mask)
        pred = torch.nn.functional.log_softmax(logits, dim=1)
        loss = self.loss_fn(pred, label)

        pred = torch.argmax(pred, dim=1)
        all_labels.extend(label.tolist())
        all_preds.extend(pred.tolist())
        metrics = self.test_metrics(pred, label)
        self.log("test_loss", loss)
        self.log_dict(metrics)

    def validation_step(self, batch, batch_idx):
        input_ids, attn_mask, label = batch

        logits = self.model(input_ids, attn_mask)
        pred = torch.nn.functional.log_softmax(logits, dim=1)

        loss = self.loss_fn(pred, label)
        self.val_acc_fn.update(pred, label)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self,):
        val_acc = self.val_acc_fn.compute()
        self.log("val_acc", val_acc, prog_bar=True)
        self.val_acc_fn.reset()

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=float(1e-4), eps=1e-6, weight_decay=0.01)
        return optimizer