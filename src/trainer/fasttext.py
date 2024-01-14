import yaml
import torch
import torch.nn as nn
from torch.optim import Adam
from src.model.BiLSTM import BiLSTM

from torch.optim import Adam
from torchmetrics import Accuracy, F1Score, Precision, Recall, MetricCollection
from lightning.pytorch import LightningModule

all_labels = []
all_preds = []
class FastTextLSTMModel(LightningModule):
    def __init__(self, num_labels: int, hidden_size:int, dropout:float , loss_weight=None):
        super(FastTextLSTMModel, self).__init__()
        self.model = BiLSTM(vector_size=hidden_size,num_labels=num_labels, drop_out=dropout)
        self.train_loss_fn = nn.CrossEntropyLoss(weight=loss_weight)
        self.loss_fn = nn.CrossEntropyLoss()
        self.test_metrics =  MetricCollection([
          Accuracy(task = 'multiclass',average="weighted", num_classes=num_labels),
          Precision(task = 'multiclass', average="weighted", num_classes=num_labels),
          Recall(task = 'multiclass',average="weighted", num_classes=num_labels),
          F1Score(task = 'multiclass',average="weighted", num_classes=num_labels)])
        self.val_acc_fn = Accuracy(task = 'multiclass',average="weighted", num_classes=num_labels)

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch):
        x, y = batch

        logits = self.model(x)
        pred = nn.functional.log_softmax(logits, dim=1)

        loss = self.train_loss_fn(pred, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch

        logits = self.model(x)
        pred = nn.functional.log_softmax(logits, dim=1)
        pred = torch.argmax(pred, dim=1)
        loss = self.loss_fn(pred, y)
        metrics = self.test_metrics(pred, y)
        all_labels.extend(y.tolist())
        all_preds.extend(pred.tolist())
        self.log("test_loss", loss)
        self.log_dict(metrics)

    def validation_step(self, batch, batch_idx):
        x, y = batch

        logits = self.model(x)
        pred = nn.functional.log_softmax(logits, dim=1)

        loss = self.loss_fn(pred, y)

        self.val_acc_fn.update(pred, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        val_acc = self.val_acc_fn.compute()
        self.log("val_acc", val_acc, prog_bar=True)
        self.val_acc_fn.reset()
    
    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=1e-4, eps=1e-6, weight_decay=0.01)
        
        return optimizer