import os
import json
import yaml
import random
import argparse
from glob import glob

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from src import data, aggregators, models
from sle import collate, SLBeta, SLDirichlet


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, required=True,
                        help="Where to save log files.")
    parser.add_argument("--dataset-name", type=str, required=True,
                        help="""Dataset to load. See `src.data`
                                for possible values.""")
    parser.add_argument("--datadirs", type=str, nargs='+', default=None,
                        help="""Path(s) to directory containing
                                dataset files to load.""")
    parser.add_argument("--split-indices-dir", type=str, default=None,
                        help="""Path to dir containing
                                {train,val,test}_indices.npy files.""")
    parser.add_argument("--data-pickle-dir", type=str, default=None,
                        help="""If specified without --datadirs,
                                load aggregated datasets from this directory.
                                If specified *with* --datadirs,
                                save aggregated datasets to this directory.""")
    parser.add_argument("--model-config", type=str, required=True,
                        help="Path to config.yaml.")
    parser.add_argument("--label-type", type=str, default="discrete",
                        choices=["discrete", "sle"])
    parser.add_argument("--label-aggregation", type=str, default="none",
                        choices=["vote", "soft", "fuse", "none"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--val-freq", type=int, default=10,
                        help="Run validation every val-freq epochs.")
    parser.add_argument("--n-train", type=int, default=-1,
                        help="Number of training examples to load.")
    parser.add_argument("--no-train", default=False, action="store_true",
                        help="Don't run training. Default False (run train)")
    parser.add_argument("--run-test", default=False, action="store_true",
                        help="Run testing. Default True.")
    parser.add_argument("--random-seed", type=int, default=0)
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_indices(dirpath):
    train_idxs = np.load(os.path.join(dirpath, "train_indices.npy"))
    val_idxs = np.load(os.path.join(dirpath, "val_indices.npy"))
    test_idxs = np.load(os.path.join(dirpath, "test_indices.npy"))
    return train_idxs, val_idxs, test_idxs


def main(args):
    run_train = not args.no_train

    if args.label_aggregation == "none":
        args.label_aggregation = None

    # ======== Load the dataset ========
    set_seed(args.random_seed)
    aggregator = get_data_aggregator(
        args.label_type, args.label_aggregation)
    if args.datadirs is not None:
        train_idxs = val_idxs = test_idxs = None
        if args.split_indices_dir is not None:
            train_idxs, val_idxs, test_idxs = load_indices(args.split_indices_dir)  # noqa
        dataset = data.load(args.dataset_name, *args.datadirs,
                                   n_train=args.n_train,
                                   train_idxs=train_idxs,
                                   val_idxs=val_idxs,
                                   test_idxs=test_idxs,
                                   random_seed=args.random_seed)
        # (Optionally) encode and aggregate the labels.
        trainset = aggregator(**dataset.train)
        valset = aggregator(**dataset.val)
        # val = aggregator(**dataset.train); warnings.warn("Validating on train set!")  # noqa
        if args.data_pickle_dir is not None:
            os.makedirs(args.data_pickle_dir, exist_ok=False)
            train_path = os.path.join(args.data_pickle_dir, "train.pkl")
            trainset.save(train_path)
            val_path = os.path.join(args.data_pickle_dir, "val.pkl")
            valset.save(val_path)
    elif args.data_pickle_dir is not None:
        train_path = os.path.join(args.data_pickle_dir, "train.pkl")
        trainset = aggregator.load(train_path)
        val_path = os.path.join(args.data_pickle_dir, "val.pkl")
        valset = aggregator.load(val_path)
    else:
        raise ValueError("You must specify one or both of --datadirs or --data-pickle-dir")  # noqa

    # ==== Get data ready for model training ====
    collate_fn = None
    if args.label_type == "sle":
        collate_fn = collate.sle_default_collate

    set_seed(args.random_seed)
    train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True,
            collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(
            valset, batch_size=100, shuffle=False,
            collate_fn=collate_fn)

    if args.run_test is True:
        if args.datadirs is not None:
            testset = aggregator(**dataset.test)
            if args.data_pickle_dir is not None:
                test_path = os.path.join(args.data_pickle_dir, "test.pkl")
                testset.save(test_path)
        else:
            test_path = os.path.join(args.data_pickle_dir, "test.pkl")
            testset = aggregator.load(test_path)
        test_loader = torch.utils.data.DataLoader(
                testset, batch_size=args.batch_size, shuffle=False,
                collate_fn=collate_fn)

    if run_train is True:
        os.makedirs(args.outdir, exist_ok=False)

    # ======= Load the model ========
    with open(args.model_config) as config_file:
        model_args = yaml.safe_load(config_file)
    print("Model:")
    print(json.dumps(model_args, indent=2))
    # synthetic: linear network
    # cifar10: resnet
    # etc.
    set_seed(args.random_seed)
    encoder = models.load_encoder_by_dataset_name(
        args.dataset_name, **model_args["encoder"])
    # discrete: linear
    # sle: SLELayer
    decoder = models.load_decoder_by_label_type(
        args.label_type, **model_args["decoder"])
    model = models.CombinedModule(encoder, decoder, lr=args.lr)
    model.to(DEVICE)
    model.train()

    # ==== Get optimizer and lr scheduler ====
    opt_cls, opt_kwargs = model.configure_optimizer()
    optimizer = opt_cls(model.parameters(), lr=args.lr, **opt_kwargs)

    lr_scheduler = None
    sch_cls, sch_kwargs = model.configure_lr_scheduler()
    if sch_cls is not None:
        lr_scheduler = sch_cls(optimizer, **sch_kwargs)

    print("Optimizer: ", optimizer)
    print("LR Scheduler: ", lr_scheduler.milestones, lr_scheduler.gamma)
    print("# Params: ", get_param_count(model))

    save_cmdline_args(args, args.outdir)
    model_ckpt_dir = os.path.join(args.outdir, "model_checkpoints")
    model_outputs_dir = os.path.join(args.outdir, "model_outputs")
    os.makedirs(model_ckpt_dir, exist_ok=True)
    os.makedirs(model_outputs_dir, exist_ok=True)

    # ======= Run training =======
    if run_train is True:
        train_losses = {}
        val_losses = {}
        # best_val_loss = torch.inf
        best_val_acc = 0.0
        set_seed(args.random_seed)
        for epoch in range(args.epochs):
            train_acc, train_loss = train(
                epoch, model, train_loader, optimizer)
            train_losses[epoch] = train_loss
            if lr_scheduler is not None:
                lr_scheduler.step()

            if (epoch+1) % args.val_freq == 0:
                val_acc, val_loss = validate(epoch, model, val_loader)
                val_losses[epoch] = val_loss
                print(f"(Val {epoch}) Accuracy: {val_acc:.4f}, Loss: {val_loss:.4f}")  # noqa
                # if val_loss < best_val_loss:
                #     print(f"Val loss improved {best_val_loss:.4f} -> {val_loss:.4f}")  # noqa
                #     best_val_loss = val_loss
                if val_acc > best_val_acc:
                    print("Saving new best model.")
                    print(f"Val accuracy improved {best_val_acc:.4f} -> {val_acc:.4f}")  # noqa
                    best_val_acc = val_acc
                    save_model(model, epoch=epoch, outdir=model_ckpt_dir)
                    save_model_outputs(model, val_loader, epoch=epoch,
                                       outdir=model_outputs_dir, split="val")

        # ====== Save outputs and run testing ======
        losslog = os.path.join(args.outdir, "train_losses.json")
        with open(losslog, 'w') as outF:
            json.dump(train_losses, outF)
        losslog = os.path.join(args.outdir, "val_losses.json")
        with open(losslog, 'w') as outF:
            json.dump(val_losses, outF)

    if args.run_test is True:
        print("Loading best performing model on validation set...")
        epoch = -1
        model = load_model(model_ckpt_dir, epoch=epoch)
        test_acc, test_loss = validate(epoch, model, test_loader)
        print(f"(Test) Accuracy: {test_acc:.4f}, Loss: {test_loss:.4f}")
        save_model_outputs(model, test_loader, epoch=epoch,
                           outdir=model_outputs_dir, split="test")


def train(epoch, model, dataloader, optimizer):
    model.train()
    losses = []
    correct_preds = []

    pbar = tqdm(dataloader)
    for (n, batch) in enumerate(dataloader):
        batch = send_to_device(batch, DEVICE)
        output = model(batch)
        loss = model.compute_loss(output, batch)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # plot_grad_flow(model.named_parameters())
        optimizer.step()
        losses.append(loss.item())

        with torch.no_grad():
            preds = model.predict(output).detach().cpu()
        golds = torch.as_tensor([gold.argmax() for gold in batch["gold_y"]])
        correct_preds.extend(preds == golds)

        if n % 10 == 0:
            avg_loss = np.mean(losses)
            acc = sum(correct_preds) / len(correct_preds)
            desc = f"({epoch}) Avg. Train Loss: {avg_loss:.4f}, Acc: {acc:.4f}"
            pbar.set_description(desc)
        pbar.update()

    accuracy = sum(correct_preds) / len(correct_preds)
    return accuracy, np.mean(losses)


def validate(epoch, model, dataloader):
    model.eval()
    losses = []
    correct_preds = []

    for (n, batch) in enumerate(dataloader):
        batch = send_to_device(batch, DEVICE)
        output = model(batch)
        loss = model.compute_loss(output, batch)
        losses.append(loss.item())
        preds = model.predict(output).detach().cpu()
        golds = torch.as_tensor([gold.argmax() for gold in batch["gold_y"]])
        correct_preds.extend(preds == golds)

    accuracy = sum(correct_preds) / len(correct_preds)
    return accuracy, np.mean(losses)


def save_cmdline_args(args, outdir):
    outpath = os.path.join(outdir, "cmdlog.json")
    with open(outpath, 'w') as outF:
        json.dump(args.__dict__, outF, indent=2)


def get_data_aggregator(label_type, label_aggregation):
    lookup = {
            ("discrete", None): aggregators.NonAggregatedDataset,
            ("discrete", "vote"): aggregators.VotingAggregatedDataset,
            ("discrete", "soft"): aggregators.SoftVotingAggregatedDataset,
            ("sle", None): aggregators.NonAggregatedSLDataset,
            ("sle", "fuse"): aggregators.CumulativeFusionDataset,
            }
    try:
        return lookup[(label_type, label_aggregation)]
    except KeyError:
        raise KeyError(f"Unsupported (label_type, label_aggregation): ({label_type}, {label_aggregation})")  # noqa


def save_model(model, epoch, outdir):
    outpath = os.path.join(outdir, f"epoch={epoch}.pt")
    torch.save(model, outpath)


def load_model(ckpt_dir, epoch=-1):
    """
    If epoch == -1, load the latest epoch.
    """
    ckpt_glob = os.path.join(ckpt_dir, "*.pt")
    ckpt_files = glob(ckpt_glob)
    ckpt_by_epoch = {}
    for fpath in ckpt_files:
        fname = os.path.basename(fpath)
        epoch_num = int(fname.strip('.pt').split('=')[1])
        ckpt_by_epoch[epoch_num] = fpath
    if epoch == -1:
        ckpt_path = sorted(ckpt_by_epoch.items(), key=lambda x: x[0])[-1][1]
    else:
        ckpt_path = ckpt_by_epoch[epoch]
    model = torch.load(ckpt_path)
    model.eval()
    return model


def save_model_outputs(model, dataloader, epoch=None,
                       outdir=None, split="train"):
    outputs = []
    model.eval()
    pbar = tqdm(dataloader)
    pbar.set_description(f"Saving {split} outputs...")
    for batch in dataloader:
        batch = send_to_device(batch, DEVICE)
        output = model(batch)
        out_datum = send_to_device(dict(
            (key, val) for (key, val) in batch.items() if key != 'X'), "cpu")
        if isinstance(output, (SLBeta, SLDirichlet)):
            output = output.parameters()
            out_datum['Y'] = out_datum['Y'].parameters()
        out_datum["model_output"] = output
        out_datum = convert_tensors_to_items(out_datum, squeeze=False)
        out_examples = uncollate(out_datum)
        outputs.extend(out_examples)

        pbar.update()

    if outdir is None:
        outdir = '.'
    outpath = os.path.join(outdir, f"{split}_epoch={epoch}.json")
    with open(outpath, 'w') as outF:
        for example in outputs:
            json.dump(example, outF)
            outF.write('\n')


def uncollate(batch):
    """
    Modified from
    https://lightning-flash.readthedocs.io/en/stable/_modules/flash/core/data/batch.html#default_uncollate  # noqa

    This function is used to uncollate a batch into samples.
    The following conditions are used:

    - if the ``batch`` is a ``dict``, the result will be a list of dicts
    - if the ``batch`` is list-like, the result is guaranteed to be a list

    Args:
        batch: The batch of outputs to be uncollated.

    Returns:
        The uncollated list of predictions.

    Raises:
        ValueError: If input ``dict`` values are not all list-like.
        ValueError: If input ``dict`` values are not all the same length.
        ValueError: If the input is not a ``dict`` or list-like.
    """
    def _is_list_like_excluding_str(x):
        if isinstance(x, str):
            return False
        try:
            iter(x)
        except TypeError:
            return False
        return True

    if isinstance(batch, dict):
        if any(not _is_list_like_excluding_str(sub_batch)
               for sub_batch in batch.values()):
            raise ValueError("When uncollating a dict, all sub-batches (values) are expected to be list-like.")  # noqa
        uncollated_vals = [uncollate(val) for val in batch.values()]
        if len(set([len(v) for v in uncollated_vals])) > 1:
            uncollated_keys_vals = [(key, uncollate(val))
                                    for (key, val) in batch.items()]
            print([(k, len(v)) for (k, v) in uncollated_keys_vals])
            raise ValueError("When uncollating a dict, all sub-batches (values) are expected to have the same length.")  # noqa
        elements = list(zip(*uncollated_vals))
        return [dict(zip(batch.keys(), element)) for element in elements]
    if isinstance(batch, (list, tuple, torch.Tensor)):
        return list(batch)
    raise ValueError(
        "The batch of outputs to be uncollated is expected to be a `dict` or list-like "  # noqa
        f"(e.g. `Tensor`, `list`, `tuple`, etc.), but got input of type: {type(batch)}"  # noqa
    )


def convert_tensors_to_items(collection, squeeze=False):
    if torch.is_tensor(collection):
        if squeeze is True:
            collection = collection.squeeze()
        if collection.dim() > 0:
            return collection.cpu().tolist()
        return collection.cpu().item()

    if isinstance(collection, dict):
        cp = dict(collection)
        for key in cp.keys():
            cp[key] = convert_tensors_to_items(cp[key], squeeze=squeeze)
    elif isinstance(collection, (list, tuple, set)):
        cp = type(collection)(collection)
        for i in range(len(cp)):
            cp[i] = convert_tensors_to_items(cp[i], squeeze=squeeze)
    else:
        return collection

    return cp


def send_to_device(collection, device):
    if torch.is_tensor(collection) or isinstance(collection, (SLBeta, SLDirichlet)):  # noqa
        if collection.device != device:
            return collection.to(device)

    if isinstance(collection, dict):
        for key in collection.keys():
            collection[key] = send_to_device(collection[key], device)
    elif isinstance(collection, (list, tuple, set)):
        for i in range(len(collection)):
            collection[i] = send_to_device(collection[i], device)
    return collection


def get_param_count(model):
    return np.sum([np.prod(p.size()) for p in model.parameters()])


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during
    training. Can be used for checking for possible gradient vanishing /
    exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())"
    to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([plt.Line2D([0], [0], color="c", lw=4),
                plt.Line2D([0], [0], color="b", lw=4),
                plt.Line2D([0], [0], color="k", lw=4)],
               ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    main(args)
