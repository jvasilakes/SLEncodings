import os
import json
import argparse

import torch
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import sle
from sle import collate
import data as D
from models import LinearNet, SLNet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, required=True,
                        help="Where to save log files.")
    parser.add_argument("--datadir", type=str, default=None,
                        help="""Path to directory containing
                                {train,val,test}.json""")
    parser.add_argument("--label-type", type=str, default="discrete",
                        choices=["discrete", "sl"])
    parser.add_argument("--label-aggregation", type=str, default=None,
                        choices=["fuse", "vote", "freq", "sample"])
    parser.add_argument("--n-features", type=int, default=5)
    parser.add_argument("--n-examples", type=int, default=10)
    parser.add_argument("--n-annotators", type=int, default=3)
    parser.add_argument('-T', "--annotator-trustworthiness", type=str,
                        choices=["perfect", "high", "high-outlier", "medium",
                                 "low", "low-outlier"], default="high",
                        help="Distribution of annotator trustworthiness.")
    parser.add_argument("--hidden-dim", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--n-train", type=int, default=-1,
                        help="Number of training examples to load.""")
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--generate-data-only", action="store_true",
                        help="""If set, generate and save data, but
                                don't train a model.""")
    return parser.parse_args()


def main(args):
    np.random.seed(0)
    dataclass = get_data_class(args.label_type, args.label_aggregation)
    os.makedirs(args.outdir, exist_ok=False)
    if args.datadir is not None:
        train_path = os.path.join(args.datadir, "train.json")
        train_dataset = dataclass.from_file(train_path, n=args.n_train)
        val_path = os.path.join(args.datadir, "val.json")
        val_dataset = dataclass.from_file(val_path)
        test_path = os.path.join(args.datadir, "test.json")
        test_dataset = dataclass.from_file(test_path)
    else:
        full_dataset = dataclass(
                args.n_features, args.n_examples,
                annotators=args.n_annotators,
                trustworthiness=args.annotator_trustworthiness,
                random_seed=args.random_seed)
        train_dataset, val_dataset, test_dataset = split_dataset(full_dataset)

        train_data_outpath = os.path.join(args.outdir, "train.json")
        train_dataset.save(train_data_outpath)
        train_dataset.plot(savepath=os.path.join(args.outdir, "train.png"))
        val_data_outpath = os.path.join(args.outdir, "val.json")
        val_dataset.save(val_data_outpath)
        test_data_outpath = os.path.join(args.outdir, "test.json")
        test_dataset.save(test_data_outpath)

    if args.generate_data_only is True:
        return

    collate_fn = None
    if isinstance(train_dataset[0]['y'], (sle.SLBeta, sle.SLDirichlet)):
        collate_fn = collate.sle_default_collate
    train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            collate_fn=collate_fn)
    val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_fn)
    test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_fn)

    modelclass = get_model_class(args.label_type, args.label_aggregation)
    model = modelclass(train_dataset.n_features, args.hidden_dim,
                       train_dataset.label_dim, lr=args.lr)
    print(model)
    model.train()

    save_cmdline_args(args, args.outdir)

    losses = []
    for epoch in range(args.epochs):
        if (epoch+1) % 10 == 0:
            save_model_outputs(model, train_dataset,
                               epoch=epoch, outdir=args.outdir)
            accuracy = run_validate(epoch, model, val_dataloader)
            print(f"Accuracy: {accuracy:.4f}")
        epoch_losses = run_train(epoch, model, train_dataloader)
        losses.append(epoch_losses)

    model.eval()
    save_model_outputs(model, train_dataset, epoch=epoch, outdir=args.outdir)

    losslog = os.path.join(args.outdir, "losses.json")
    with open(losslog, 'w') as outF:
        json.dump(losses, outF)


def run_train(epoch, model, dataloader):
    model.train()
    losses = []
    pbar = tqdm(dataloader)
    for (n, batch) in enumerate(dataloader):
        output = model(batch["x"])
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
        output = model(batch["x"])
        loss = model.compute_loss(output, batch)
        losses.append(loss.item())
        preds = model.predict(output)
        all_preds.extend(preds)
        golds = batch["preferred_y"]
        all_golds.extend(golds)

        if n % 100 == 0:
            avg_loss = torch.mean(torch.tensor(losses))
            pbar.set_description(f"(Val {epoch}) Avg. Loss: {avg_loss:.4f}")  # noqa
        pbar.update()

    all_preds = torch.as_tensor(all_preds)
    all_golds = torch.as_tensor(all_golds)
    accuracy = (all_preds == all_golds).sum() / len(all_golds)
    return accuracy


def split_dataset(dataset):
    example_ids = list(set([ex["example_id"] for ex in dataset]))
    train_ids, eval_ids = train_test_split(
            example_ids, train_size=0.8, random_state=0)
    val_ids, test_ids = train_test_split(
            eval_ids, test_size=0.5, random_state=0)

    train_idxs = []
    val_idxs = []
    test_idxs = []
    for (i, ex) in enumerate(dataset):
        if ex["example_id"] in train_ids:
            train_idxs.append(i)
        elif ex["example_id"] in val_ids:
            val_idxs.append(i)
        elif ex["example_id"] in test_ids:
            test_idxs.append(i)
    assert len(train_idxs) + len(val_idxs) + len(test_idxs) == len(dataset)
    train_ds = dataset.subset(train_idxs)
    val_ds = dataset.subset(val_idxs)
    test_ds = dataset.subset(test_idxs)
    return train_ds, val_ds, test_ds


def save_cmdline_args(args, outdir):
    outpath = os.path.join(outdir, "cmdlog.json")
    with open(outpath, 'w') as outF:
        json.dump(args.__dict__, outF, indent=2)


def get_data_class(label_type, label_aggregation):
    lookup = {
            ("discrete", None): D.MultiAnnotatorDataset,
            ("discrete", "vote"): D.VotingAggregatedDataset,
            ("discrete", "freq"): D.FrequencyAggregatedDataset,
            ("discrete", "sample"): D.CatSampleAggregatedDataset,
            ("sl", None): D.SubjectiveLogicDataset,
            ("sl", "fuse"): D.CumulativeFusionDataset,
            ("sl", "sample"): D.SLSampleAggregatedDataset,
            }
    try:
        return lookup[(label_type, label_aggregation)]
    except KeyError:
        raise KeyError(f"Unsupported (label_type, label_aggregation): ({label_type}, {label_aggregation})")  # noqa


def get_model_class(label_type, label_aggregation):
    lookup = {
            ("discrete", None): LinearNet,
            ("discrete", "vote"): LinearNet,
            ("discrete", "freq"): LinearNet,
            ("discrete", "sample"): LinearNet,
            ("sl", None): SLNet,
            ("sl", "fuse"): SLNet,
            ("sl", "sample"): LinearNet,
            }
    try:
        return lookup[(label_type, label_aggregation)]
    except KeyError:
        raise KeyError(f"Unsupported (label_type, label_aggregation): ({label_type}, {label_aggregation})")  # noqa


def save_model_outputs(model, dataset, epoch=None, outdir=None):
    outputs = {}
    model.eval()
    seen = set()
    pbar = tqdm(dataset)
    pbar.set_description("Saving model outputs...")
    for datum in pbar:
        datum_cp = dict(datum)
        uuid = datum_cp.pop("uuid")
        if uuid in seen:
            continue
        seen.add(uuid)
        output = model(datum["x"])
        if "distribution" in output:
            output = {k: v.squeeze() for (k, v)
                      in output["distribution"].parameters().items()}
            datum_cp['y'] = datum_cp['y'].parameters()
        datum_cp["model_output"] = output
        datum_cp = convert_tensors_to_items(datum_cp)
        outputs[uuid] = datum_cp
        pbar.update()

    if outdir is None:
        outdir = '.'
    outpath = os.path.join(outdir, f"outputs_epoch={epoch}.json")
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
