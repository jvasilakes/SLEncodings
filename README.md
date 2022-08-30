# Subjective Logic Encodings (SLEs)


## Loading datasets

Dataset-specific loading logic is implemented in `src.dataloaders`.

`torch.utils.data.Dataset` subclasses specialized for crowdsourced
datasets are implemented in `src.datasets`.

For example, let's load CIFAR-10S.
Download the CIFAR-10 images using https://pytorch.org/vision/master/generated/torchvision.datasets.CIFAR10.html
Download the CIFAR-10S annotations from https://github.com/cambridge-mlg/cifar-10s

The load the data using the following code.

```python
import src.dataloaders as DL
import src.datasets as DS

cifar10s = SL.CIFAR10SDataLoader("path/to/cifar10-pytorch", "path/to/cifar-10s_t2clamp_redist10.json")

# The basic class for crowdsourced annotations
# Each example is paired with all its annotations.
train = DS.MultiAnnotatorDataset(**cifar.train)
val = DS.MultiAnnotatorDataset(**cifar.val)
test = DS.MultiAnnotatorDataset(**cifar.test)

# Use it like so
for (x, y) in train:
	print(x.shape)
	print(y.shape)
# torch.Size([3, 32, 32])  img dims
# torch.Size([7, 10]) (annotators, labels)

# Keep each (img, annotation) as a separate example.
# This means each img with occur multiple times in the dataset.
train = DS.NonAggregatedDataset(**cifar.train)
for (x, y) in train:
	print(x.shape)
	print(y.shape)
# torch.Size([3, 32, 32])  img dims
# torch.Size([10]) (labels,)

# Aggregate labels with majority voting
train = DS.VotingAggregatedDataset(**cifar.train)
for (x, y) in train:
	print(x.shape)
	print(y.shape)
# torch.Size([3, 32, 32])  img dims
# torch.Size([10]) (labels,)

# Like MultiAnnotatorDataset, but each annotation is an SLE
train = DS.SubjectiveLogicDataset(**cifar.train)

# Like NonAggregatedDataset, but each annotation is an SLE
train = DS.NonAggregatedSLDataset(**cifar.train)

# Aggregated labels with cumulative fusion
# Instead of a one-hot or probabilistic label,
#   target is a Dirichlet distribution.
train = DS.CumulativeFusionDataset(**cifar.train)
for (x, y) in train:
	print(x.shape)
	print(y)
# torch.Size([3, 32, 32])  img dims
# SLDirichlet([0., 0., 0., 0., 0.954, 0.029, 0., 0.017, 0., 0.], 0.)
```


## Synthetic Data

Generate the data using the `generate_data.py` script. For example,
```
python generate_data.py --outdir data/synthetic/perfect --random-seed 0 
                        --n-examples 1000 --n-features 2 --n-annotators 10 
                        --reliability "perfect" --certainty perfect
```

Load the data as follows.

```python
import src.dataloaders as DL
import src.datasets as DS

synth = DL.SyntheticDataLoader("data/synthetic/perfect/")
train = DS.VotingAggregatedDataset(**synth.train)
# etc.
```
