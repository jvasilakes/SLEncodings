import os
import json
import yaml
import argparse

import torch
import numpy as np
from tqdm import tqdm

from src import dataloaders, datasets, models
from sle import collate, SLBeta, SLDirichlet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, required=True,
                        help="Where to save log files.")
    parser.add_argument("--dataset-name", type=str, required=True,
                        help="""Dataset to load. See `src.dataloaders`
                                for possible values.""")
    parser.add_argument("--datadirs", type=str, nargs='+', required=True,
                        help="""Path(s) to directory containing
                                dataset files to load.""")
    parser.add_argument("--model-config", type=str, required=True,
                        help="Path to config.yaml.")
    parser.add_argument("--label-type", type=str, default="discrete",
                        choices=["discrete", "sle"])
    parser.add_argument("--label-aggregation", type=str, default=None,
                        choices=["vote", "fuse", "none"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--n-train", type=int, default=-1,
                        help="Number of training examples to load.""")
    parser.add_argument("--random_seed", type=int, default=0)
    return parser.parse_args()


def main(args):
    np.random.seed(0)
    if args.label_aggregation == "none":
        args.label_aggregation = None

    # Load the dataset
    dataset = dataloaders.load(args.dataset_name, *args.datadirs)
    # (Optionally) encode and aggregate the labels.
    aggregator = get_data_aggregator(args.label_type, args.label_aggregation)
    train = aggregator(**dataset.train)
    val = aggregator(**dataset.val)
    test = aggregator(**dataset.test)

    collate_fn = None
    if args.label_type == "sle":
        collate_fn = collate.sle_default_collate

    # Get data ready for model training.
    train_loader = torch.utils.data.DataLoader(
            train, batch_size=args.batch_size, shuffle=True,
            collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(
            val, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(
            test, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_fn)

    os.makedirs(args.outdir, exist_ok=False)

    with open(args.model_config) as config_file:
        model_args = yaml.safe_load(config_file)
    # synthetic: linear network
    # cifar10: resnet
    # etc.
    encoder = models.load_encoder_by_dataset_name(
        args.dataset_name, **model_args["encoder"])
    # discrete: linear -> softmax
    # sle: SLELayer
    decoder = models.load_decoder_by_label_type(
        args.label_type, **model_args["decoder"])
    model = models.CombinedModule(encoder, decoder, lr=args.lr)
    model.train()

    save_cmdline_args(args, args.outdir)
    model_outputs_dir = os.path.join(args.outdir, "model_outputs")
    os.makedirs(model_outputs_dir)

    losses = []
    for epoch in range(args.epochs):
        if (epoch+1) % 10 == 0:
            save_model_outputs(
                model, train, epoch=epoch, outdir=model_outputs_dir)
            accuracy = run_validate(epoch, model, val_loader)
            print(f"Accuracy: {accuracy:.4f}")
        epoch_losses = run_train(epoch, model, train_loader)
        losses.append(epoch_losses)

    model.eval()
    save_model_outputs(model, train, epoch=epoch, outdir=model_outputs_dir)
    save_model_outputs(model, val, epoch=epoch,
                       outdir=model_outputs_dir, split="val")

    losslog = os.path.join(args.outdir, "losses.json")
    with open(losslog, 'w') as outF:
        json.dump(losses, outF)


def run_train(epoch, model, dataloader):
    model.train()
    losses = []
    pbar = tqdm(dataloader)
    for (n, batch) in enumerate(dataloader):
        output = model(batch)
        loss = model.compute_loss(output, batch)
        loss.backward()
        model.optimizer.step()
        model.optimizer.zero_grad()
        losses.append(loss.item())
        if n % 100 == 0:
            avg_loss = torch.mean(torch.tensor(losses))
            pbar.set_description(f"({epoch}) Avg. Train Loss: {avg_loss:.4f}")  # noqa
        pbar.update()
    return losses


def run_validate(epoch, model, dataloader):
    model.eval()
    losses = []
    all_preds = []
    all_golds = []

    pbar = tqdm(dataloader)
    for (n, batch) in enumerate(dataloader):
        output = model(batch)
        loss = model.compute_loss(output, batch)
        losses.append(loss.item())
        preds = model.predict(output)
        all_preds.extend(preds)
        all_golds.extend(batch["gold_y"].argmax(dim=1))

        if n % 100 == 0:
            avg_loss = torch.mean(torch.tensor(losses))
            pbar.set_description(f"(Val {epoch}) Avg. Loss: {avg_loss:.4f}")  # noqa
        pbar.update()

    all_preds = torch.as_tensor(all_preds)
    all_golds = torch.as_tensor(all_golds)
    accuracy = (all_preds == all_golds).sum() / len(all_golds)
    return accuracy


def save_cmdline_args(args, outdir):
    outpath = os.path.join(outdir, "cmdlog.json")
    with open(outpath, 'w') as outF:
        json.dump(args.__dict__, outF, indent=2)


def get_data_aggregator(label_type, label_aggregation):
    lookup = {
            ("discrete", None): datasets.NonAggregatedDataset,
            ("discrete", "vote"): datasets.VotingAggregatedDataset,
            ("sle", None): datasets.NonAggregatedSLDataset,
            ("sle", "fuse"): datasets.CumulativeFusionDataset,
            }
    try:
        return lookup[(label_type, label_aggregation)]
    except KeyError:
        raise KeyError(f"Unsupported (label_type, label_aggregation): ({label_type}, {label_aggregation})")  # noqa


def save_model_outputs(model, dataset, epoch=None, outdir=None, split="train"):
    outputs = []
    model.eval()
    pbar = tqdm(dataset)
    pbar.set_description("Saving model outputs...")
    for datum in pbar:
        output = model(datum)
        datum_cp = dict(datum)
        if isinstance(output, (SLBeta, SLDirichlet)):
            output = output.parameters()
            datum_cp['Y'] = datum_cp['Y'].parameters()
        datum_cp["model_output"] = output
        datum_cp = convert_tensors_to_items(datum_cp)
        outputs.append(datum_cp)
        pbar.update()

    outputs = sorted(outputs, key=lambda x: x["metadata"][0]["example_id"])
    if outdir is None:
        outdir = '.'
    outpath = os.path.join(outdir, f"{split}_epoch={epoch}.json")
    with open(outpath, 'w') as outF:
        json.dump(outputs, outF)


def convert_tensors_to_items(collection):
    if torch.is_tensor(collection):
        if collection.dim() > 0:
            return collection.tolist()
        else:
            return collection.item()

    if isinstance(collection, dict):
        cp = dict(collection)
        for key in cp.keys():
            cp[key] = convert_tensors_to_items(cp[key])
    elif isinstance(collection, (list, tuple, set)):
        cp = type(collection)(collection)
        for i in range(len(cp)):
            cp[i] = convert_tensors_to_items(cp[i])
    else:
        return collection

    return cp


if __name__ == "__main__":
    args = parse_args()
    main(args)
