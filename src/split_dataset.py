import argparse
import logging
import os
from random import shuffle
import os.path as osp

from utils.data_helper import load_jsonl, to_jsonl


parser = argparse.ArgumentParser()
parser.add_argument("--input_data_file", type=str)
parser.add_argument("--output_data_folder", type=str)
parser.add_argument("--num_replica", type=int)
parser.add_argument("--mode", type=str, default='no-action',
                    help="cv-10 or tdt-8-1-1")

logging.basicConfig(filename='log/split_dataset.log',
                    filemode="wt",
                    level=logging.INFO)


def num_dict_parse(prefetched_num_dict):
    num_dict = {}
    for key in prefetched_num_dict:
        _, l, k = key.strip('[]').split(":")
        if l == "1":
            num_dict[f"M{k}"] = prefetched_num_dict[key]['v']
        elif l == "2":
            num_dict[f"N{k}"] = prefetched_num_dict[key]['v']
    assert len(num_dict)
    return num_dict


def make_train_dev_test(datalist, output_folder, rtrain, rdev, rtest):
    shuffle(datalist)
    total_size = len(datalist)
    train_begin = 0
    train_end = train_begin + int(rtrain/(rtrain+rdev+rtest) * total_size)
    dev_begin = train_end
    dev_end = dev_begin + int(rdev/(rtrain+rdev+rtest) * total_size)
    test_begin = dev_end
    test_end = total_size
    train_dl = datalist[train_begin: train_end]
    dev_dl = datalist[dev_begin: dev_end]
    test_dl = datalist[test_begin: test_end]
    to_jsonl(train_dl, osp.join(output_folder, "train.jsonl"))
    to_jsonl(dev_dl, osp.join(output_folder, "dev.jsonl"))
    to_jsonl(test_dl, osp.join(output_folder, "test.jsonl"))


def make_cross_validation(datalist, num_folds):
    shuffle(datalist)
    size = len(datalist)
    fold_size = int(size / num_folds)
    folds = []
    for i in range(num_folds-1):
        folds.append(datalist[i * fold_size: (i+1) * fold_size])
    folds.append(datalist[(i+1) * fold_size:])

    return folds


if __name__ == "__main__":
    args = parser.parse_args()

    datalist = load_jsonl(args.input_data_file)

    mode, *mode_params = args.mode.split('-')
    if mode == 'tdt':
        for i in range(args.num_replica):

            output_folder = f"{args.output_data_folder}-{args.mode}-replica-{i}"
            os.makedirs(output_folder, exist_ok=True)
            rtrain, rdev, rtest = [float(p) for p in mode_params]
            make_train_dev_test(datalist, output_folder, rtrain, rdev, rtest)

    if mode == 'cv':
        num_folds = int(mode_params[0])
        folds = make_cross_validation(datalist, num_folds)

        for i in range(num_folds):
            test_dl = folds[i]
            train_dev_fold = []
            for j in range(num_folds):
                if i == j:
                    continue
                else:
                    train_dev_fold += folds[j]

            shuffle(train_dev_fold)
            dev_size = len(train_dev_fold) // 5
            dev_dl = train_dev_fold[:dev_size]
            train_dl = train_dev_fold[dev_size:]

            output_folder = f"{args.output_data_folder}-{args.mode}-replica-{i}"
            os.makedirs(output_folder)
            to_jsonl(train_dl, osp.join(output_folder, "train.jsonl"))
            to_jsonl(dev_dl, osp.join(output_folder, "dev.jsonl"))
            to_jsonl(test_dl, osp.join(output_folder, "test.jsonl"))
