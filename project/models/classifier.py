import hydra
import pytorch_lightning as pl
import torch
import torchmetrics
from torch.nn import functional as F


class LitClassifier(pl.LightningModule):
    def __init__(self, optim, input_dim, output_dim, hidden_dim=128, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.l1 = torch.nn.Linear(input_dim, hidden_dim)
        self.l2 = torch.nn.Linear(hidden_dim, output_dim)

        self.val_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()

        self.example_input_array = torch.randn([input_dim, input_dim])

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('validation_loss', loss)
        accuracy = self.val_accuracy(torch.softmax(y_hat, dim=1), y)
        self.log('validation_accuracy', accuracy)

    def validation_epoch_end(self, outputs):
        self.log("validation_accuracy", self.val_accuracy.compute())
        pass

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss)
        test_accuracy = self.test_accuracy(torch.softmax(y_hat, dim=1), y)
        self.log('test_accuracy', test_accuracy)
        return test_accuracy

    def test_epoch_end(self, outputs):
        self.log('test_accuracy', self.test_accuracy.compute())
        pass

    def configure_optimizers(self):
        return hydra.utils.instantiate(self.hparams.optim, params=self.parameters())

    def on_train_start(self):
        # Proper logging of hyperparams and metrics in TB
        self.logger.log_hyperparams(self.hparams)