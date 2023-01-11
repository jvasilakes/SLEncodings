import os
from glob import glob
from collections import defaultdict

import numpy as np
import pandas as pd
import pandas.api.types as ptypes
from transformers import AutoTokenizer

from src.data.utils import register, onehot


@register("mre")
class MREDataLoader(object):
    """
    datadir: Directory containing "ground_truth*.csv" and raw/
    split_num: Int 0 to 4 representing the cross validation split.
    bert_name_or_path: Which BERT tokenizer to load from transformers.
    max_seq_len: Maximum sequence length for BERT tokenizer.
    """

    def __init__(self, datadir, split_num=0,
                 bert_name_or_path="bert-base-uncased",
                 max_seq_len=200):
        self.datadir = datadir
        assert split_num < 5
        self.split_num = split_num
        self.bert_name_or_path = bert_name_or_path
        self.max_seq_len = max_seq_len

        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_name_or_path,
                                                       use_fast=True)
        self.train, self.val = self.load()
        self.test = None

    def load(self):
        gold_file = os.path.join(self.datadir, "ground_truth_cause.csv")
        gold_data = self._load_gold_data(gold_file)

        crowddir = os.path.join(self.datadir, "raw", "RelEx")
        csvglob = os.path.join(crowddir, "*.csv")
        crowd_data = defaultdict(list)
        for fpath in glob(csvglob):
            filedata = self._load_single_file(fpath, gold_data)
            for (_, datum) in filedata.iterrows():
                crowd_data[datum.SID].append(datum)

        X = []
        Y = []
        gold_y = []
        metadata = []
        for (_, datum) in gold_data.iterrows():
            ys = []
            this_metadata = []
            for ann in crowd_data[datum.SID]:
                ys.append(onehot(ann.causes, 2))
                this_metadata.append({"annotator_id": ann._worker_id,
                                      "sentence_id": datum.SID})
            Y.append(np.array(ys))
            metadata.append(this_metadata)

            enc = self.tokenizer(datum.sentence, max_length=self.max_seq_len,
                                 padding="max_length", return_tensors="pt")
            X.append(enc)
            # Convert from [-1., 1.] to [0, 1]
            lab = int(datum.expert == 1.0)
            gold_y.append(onehot(lab, 2))

        # Split data according to self.split_num
        train, val = self._split_data(X, Y, gold_y, metadata)
        return train, val

    def _load_gold_data(self, gold_file):
        data = pd.read_csv(gold_file)
        data = data[~data.expert.isna()]
        return data[["SID", "sentence", "b1", "e1", "b2", "e2",
                     "crowd", "expert"]]

    def _load_single_file(self, filepath, gold_data):
        data = pd.read_csv(filepath)
        if ptypes.is_string_dtype(data.sent_id):
            data["SID"] = data.sent_id.apply(lambda s: np.int(s.split('-')[0]))
        else:
            data["SID"] = data.sent_id
        data = data[data.SID.isin(gold_data.SID)]
        data["causes"] = data.relations.str.contains("[CAUSES]", regex=False)
        data["causes"] = data["causes"].astype(int)
        return data[["SID", "_worker_id", "causes"]]

    def _split_data(self, X, Y, gold_y, metadata):
        """
        Reproduces the cross validation logic in
        https://github.com/AlexandraUma/dali-learning-with-disagreement/blob/758b5a6b47bcb6da073550bd55710a57ab5029d3/mre/mre_mtl.py#L470  # noqa
        """
        idxs = list(range(len(gold_y)))
        start_i = 0 + (195 * self.split_num)
        end_i = 195 + (195 * self.split_num)
        val_idxs = idxs[start_i:end_i]

        train = {'X': [], 'Y': [], "gold_y": [], "metadata": []}
        val = {'X': [], 'Y': [], "gold_y": [], "metadata": []}
        for i in idxs:
            split = train
            if i in val_idxs:
                split = val
            split['X'].append(X[i])
            split['Y'].append(Y[i])
            split["gold_y"].append(gold_y[i])
            split["metadata"].append(metadata[i])
        return train, val
