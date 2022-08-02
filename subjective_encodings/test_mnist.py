"""
Credit given to
https://gist.github.com/kdubovikov/eb2a4c3ecadd5295f68c126542e59f0a
for the original code.
"""
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D
import torch.nn.functional as F

from tqdm import tqdm
from torchvision import datasets, transforms

import subjective_encodings as sle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-S", "--subjective",
                        action="store_true", default=False)
    return parser.parse_args()


class CNNClassifier(nn.Module):
    """Custom module for a simple convnet classifier"""
    def __init__(self, subjective=False):
        super(CNNClassifier, self).__init__()
        self.subjective = subjective
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        if subjective is True:
            self.pred_layer = sle.DirichletLayer(50, 10)
        else:
            self.pred_layer = nn.Linear(50, 10)

    def forward(self, x):
        # input is 28x28x1
        # conv1(kernel=5, filters=10) 28x28x10 -> 24x24x10
        # max_pool(kernel=2) 24x24x10 -> 12x12x10
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # conv2(kernel=5, filters=20) 12x12x20 -> 8x8x20
        # max_pool(kernel=2) 8x8x20 -> 4x4x20
        x = F.relu(F.max_pool2d(self.dropout(self.conv2(x)), 2))
        # flatten 4x4x20 = 320
        x = x.view(-1, 320)
        # 320 -> 50
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        # 50 -> 10
        y_hat = self.pred_layer(x)
        return y_hat

    def compute_loss(self, preds, targets):
        if self.subjective is True:
            preds = preds.rsample()
            targets = targets.rsample()
            #loss = D.kl_divergence(preds, targets).mean()
        loss = F.cross_entropy(preds, targets)
        return loss


def get_data(subjective=False):
    # download and transform train dataset
    mnist_transform = transforms.Compose(
        [transforms.ToTensor(),  # first, convert image to PyTorch tensor
         transforms.Normalize((0.1307,), (0.3081,))]  # normalize inputs
         )
    if subjective is True:
        sle_transform = lambda y: sle.encode_one(y, 10)  # noqa
        collate_fn = sle_collate_fn
    else:
        sle_transform = None
        collate_fn = None

    train_dataset = datasets.MNIST('../mnist_data', download=True, train=True,
                                   transform=mnist_transform,
                                   target_transform=sle_transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=10, shuffle=True, collate_fn=collate_fn)

    # download and transform test dataset
    test_dataset = datasets.MNIST('../mnist_data', download=True, train=False,
                                  transform=mnist_transform,
                                  target_transform=sle_transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=10, shuffle=False, collate_fn=collate_fn)

    return train_loader, test_loader


def sle_collate_fn(batch):
    data = []
    targets = []
    for (x, y) in batch:
        data.append(x)
        targets.append(y)
    collated_x = torch.stack(data)
    collated_y = sle.collate_sle_labels(targets)
    return collated_x, collated_y


def train(clf, opt, train_loader):
    loss_history = []
    clf.train()
    pbar = tqdm(train_loader)
    for (i, (data, target)) in enumerate(pbar):
        opt.zero_grad()
        preds = clf(data)
        loss = clf.compute_loss(preds, target)
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(clf.parameters(), 5.)
        loss_history.append(loss.item())
        opt.step()
        if i % 100 == 0:
            avg_loss = torch.mean(torch.tensor(loss_history))
            pbar.set_description(f"Avg. Train Loss: {avg_loss:.4f}")


def test(clf, test_loader):
    clf.eval()  # set model in inference mode (need this because of dropout)
    test_loss = 0
    correct = 0

    print("Testing...")
    for (data, target) in tqdm(test_loader):
        output = clf(data)
        test_loss += clf.compute_loss(output, target).item()
        pred = output.argmax(1)  # get the index of the max log-prob
        correct += pred.eq(target).cpu().sum()

    test_loss = test_loss
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print("  Avg. Test Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))


if __name__ == "__main__":
    args = parse_args()
    # create classifier and optimizer objects
    clf = CNNClassifier(subjective=args.subjective)
    opt = optim.SGD(clf.parameters(), lr=0.01, momentum=0.5)
    train_loader, test_loader = get_data(subjective=args.subjective)

    for epoch in range(0, 3):
        print("Epoch %d" % epoch)
        train(clf, opt, train_loader)
        test(clf, test_loader)
