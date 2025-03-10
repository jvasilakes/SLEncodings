import torch.nn as nn
import torch.optim as optim
from transformers import BertModel

from .resnet import ResBlock


ENCODER_REGISTRY = {}


def register_encoder(*dataset_names):
    def add_to_registry(cls):
        for ds_name in dataset_names:
            ENCODER_REGISTRY[ds_name] = cls
        return cls
    return add_to_registry


@register_encoder("synthetic")
class LinearEncoder(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.encoder = nn.Sequential(
                nn.Linear(input_size, self.hidden_size),
                nn.ReLU()
        )

    def forward(self, batch):
        inputs = batch['X']
        return self.encoder(inputs)


@register_encoder("cifar10", "cifar10s", "cifar10h")
class ResNetEncoder(nn.Module):

    def __init__(self, n=7, res_option='A', use_dropout=False):
        super().__init__()
        self.res_option = res_option
        self.use_dropout = use_dropout
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)
        self.layers1 = self._make_layer(n, 16, 16, 1)
        self.layers2 = self._make_layer(n, 32, 16, 2)
        self.layers3 = self._make_layer(n, 64, 32, 2)
        self.avgpool = nn.AvgPool2d(8)
        # self.linear = nn.Linear(64, 10)
        self.hidden_size = 64

    def _make_layer(self, layer_count, channels, channels_in, stride):
        return nn.Sequential(
            ResBlock(channels, channels_in, stride, res_option=self.res_option,
                     use_dropout=self.use_dropout),
            *[ResBlock(channels) for _ in range(layer_count-1)])

    def forward(self, batch):
        x = batch['X']
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)
        out = self.layers1(out)
        out = self.layers2(out)
        out = self.layers3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        # out = self.linear(out)
        return out

    def configure_optimizer(self):
        opt_cls = optim.SGD
        opt_kwargs = {
            "momentum": 0.9,
            "weight_decay": 0.0001}
        return (opt_cls, opt_kwargs)

    def configure_lr_scheduler(self, num_train_batches, epochs):
        """
        num_train_batches and epochs are unused here.
        """
        sched_cls = optim.lr_scheduler.MultiStepLR
        sched_kwargs = {
            "milestones": [50, 55],
            "gamma": 0.1
        }
        update_freq = "epoch"
        return (sched_cls, sched_kwargs, update_freq)


@register_encoder("mre")
class BERTEncoder(nn.Module):

    def __init__(self, bert_name_or_path="bert-base-uncased"):
        super().__init__()
        self.bert_name_or_path = bert_name_or_path
        self.bert = BertModel.from_pretrained(self.bert_name_or_path)
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.hidden_size = self.bert.config.hidden_size

    def forward(self, batch):
        x = batch['X']
        encoded = self.bert(**x).pooler_output
        dropped = self.dropout(encoded)
        return dropped

    def configure_optimizer(self):
        opt_cls = optim.AdamW
        opt_kwargs = {
            "eps": 1e-8,
            "weight_decay": 0.0
        }
        return (opt_cls, opt_kwargs)

    def configure_lr_scheduler(self, num_train_batches, epochs):
        sched_cls = optim.lr_scheduler.LinearLR
        sched_kwargs = {
            "start_factor": 1.0,
            "end_factor": 0.0,
            "total_iters": num_train_batches * epochs
        }
        update_freq = "batch"
        return (sched_cls, sched_kwargs, update_freq)
