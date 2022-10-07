import collections
import os
from collections import defaultdict
import argparse
from utils.meta_helper import load_config, dump_config
from utils.data_helper import load_jsonl
from utils.dataset import get_dataloader

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str)


def linearize_dict(d, prefix_k="", output_d={}):
    for k in d:
        v = d[k]
        if isinstance(v, dict):
            linearize_dict(v, k, output_d)
        else:
            if prefix_k:
                key = prefix_k + "." + k
            else:
                key = k
            output_d[key] = v
    return output_d


def recover_the_info_bart_forms_3way(dataset_file, badcase_file, num_labels=2):
    dataloader = get_dataloader(
        [dataset_file], shuffle=False, batch_size=1, num_labels=num_labels)
    full_case_dict_raw = {}
    for batch_info in dataloader:
        for i, case_id in enumerate(batch_info['case_id']):
            full_case_dict_raw[case_id] = {
                k: batch_info[k][i] for k in batch_info
            }

    df = pd.read_csv(badcase_file, sep='\t')

    step, accuracy = [], []
    for s, step_df in df.groupby('global_step'):
        step.append(s)
        # recover results
        bad_case_dict_inference = dict()
        for _, row in step_df.iterrows():
            bad_case_dict_inference[row.case_id] = row.to_dict()

        full_case_id_set = set(full_case_dict_raw.keys())
        bad_case_id_set = set(bad_case_dict_inference.keys())
        good_case_id_set = full_case_id_set.difference(bad_case_id_set)

        bad_case_ids = list(bad_case_id_set)
        good_case_ids = list(good_case_id_set)

        prediction_tuple_list = []
        for case_id in good_case_ids:
            pred = full_case_dict_raw[case_id]['gold_label']
            prediction_tuple_list.append((case_id, pred))

        for case_id in bad_case_ids:
            pred = bad_case_dict_inference[case_id]['prediction']
            prediction_tuple_list.append((case_id, pred))

        correct_vec = []
        for case_id, pred in prediction_tuple_list:
            label = full_case_dict_raw[case_id]['gold_label']
            if label == 2:
                label = 1
            if label == pred:
                correct_vec.append(1)
            else:
                correct_vec.append(0)

        accuracy.append(np.mean(correct_vec))

    print(step, accuracy)
    return zip(step, accuracy)


def recover_the_info_shared_encoder_2way(dataset_file, badcase_file, num_labels=2):
    dataloader = get_dataloader(
        [dataset_file], shuffle=False, batch_size=1, num_labels=num_labels)
    full_case_dict_raw = {}
    for batch_info in dataloader:
        for i, case_id in enumerate(batch_info['case_id']):
            full_case_dict_raw[case_id] = {
                k: batch_info[k][i] for k in batch_info
            }

    df = pd.read_csv(badcase_file, sep='\t')

    step, accuracy = [], []
    for s, step_df in df.groupby('global_step'):
        step.append(s)
        # recover results
        bad_case_dict_inference = dict()
        for _, row in step_df.iterrows():
            bad_case_dict_inference[row.case_id] = row.to_dict()

        full_case_id_set = set(full_case_dict_raw.keys())
        bad_case_id_set = set(bad_case_dict_inference.keys())
        good_case_id_set = full_case_id_set.difference(bad_case_id_set)

        bad_case_ids = list(bad_case_id_set)
        good_case_ids = list(good_case_id_set)

        prediction_tuple_list = []
        for case_id in good_case_ids:
            pred = full_case_dict_raw[case_id]['gold_label']
            prediction_tuple_list.append((case_id, pred))

        for case_id in bad_case_ids:
            pred = bad_case_dict_inference[case_id]['symbolic_prediction']
            prediction_tuple_list.append((case_id, pred))

        correct_vec = []
        for case_id, pred in prediction_tuple_list:
            label = full_case_dict_raw[case_id]['gold_label']
            if label == 1:
                label = 2
            if label == pred:
                correct_vec.append(1)
            else:
                correct_vec.append(0)

        accuracy.append(np.mean(correct_vec))

    print(step, accuracy)
    return zip(step, accuracy)


def parse_folder(exp_folder):
    data = {}

    with open(os.path.join(exp_folder, 'STATUS'), 'rt') as f:
        status = f.readline().strip()
    data['status'] = status

    exp_config = load_config(os.path.join(exp_folder, 'config.yaml'))
    exp_config = linearize_dict(exp_config)
    for k in exp_config:
        if k.endswith('.filename_list'):
            data[k] = exp_config[k][0]
        else:
            data[k] = exp_config[k]
    try:
        with open(os.path.join(exp_folder, 'BESTCKPT'), 'rt') as f:
            best_ckpt = f.readline().strip()
            data['best_ckpt'] = best_ckpt
    except:
        data['best_ckpt'] = "NA"

    if exp_config['model_params.model_name'] == 'bart_forms_3way_dropit':
        dev_dataset = exp_config['dev_dataset.filename_list'][0]
        ret = recover_the_info_bart_forms_3way(
            dataset_file=dev_dataset,
            badcase_file=os.path.join(exp_folder, 'dev_badcase.tsv'),
            num_labels=3)

        dev_dict = {k: v for k, v in ret}

        test_dataset = exp_config['test_dataset.filename_list'][0]
        ret = recover_the_info_bart_forms_3way(
            dataset_file=test_dataset,
            badcase_file=os.path.join(exp_folder, 'test_badcase.tsv'),
            num_labels=3)

        test_dict = {k: v for k, v in ret}

    elif exp_config['model_params.model_name'] == 'shared_encoder_2way_dropit':
        dev_dataset = exp_config['dev_dataset.filename_list'][0]
        ret = recover_the_info_shared_encoder_2way(
            dataset_file=dev_dataset,
            badcase_file=os.path.join(exp_folder, 'dev_badcase.tsv'),
            num_labels=2)

        dev_dict = {k: v for k, v in ret}

        test_dataset = exp_config['test_dataset.filename_list'][0]
        ret = recover_the_info_shared_encoder_2way(
            dataset_file=test_dataset,
            badcase_file=os.path.join(exp_folder, 'test_badcase.tsv'),
            num_labels=2)

        test_dict = {k: v for k, v in ret}

    else:

        accuracy_key = 'symbolic_accuracy'

        dev_info = os.path.join(exp_folder, "dev_info.tsv")
        dev_info_df = pd.read_csv(dev_info, sep='\t')

        if accuracy_key not in dev_info_df.columns:
            accuracy_key = 'nli_accuracy'
            assert accuracy_key in dev_info_df.columns

        dev_dict = dict(zip(dev_info_df.global_step,
                            dev_info_df[accuracy_key]))

        test_info = os.path.join(exp_folder, "test_info.tsv")
        test_info_df = pd.read_csv(test_info, sep='\t')
        test_dict = dict(zip(test_info_df.global_step,
                             test_info_df[accuracy_key]))

    best_dev_acc = -1
    best_test_acc = -1
    best_step = 0

    for k in test_dict:
        dev_acc = dev_dict[k]
        test_acc = test_dict[k]

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            best_test_acc = test_acc
            best_step = k

    data['best_dev_acc'] = best_dev_acc
    data['best_test_acc'] = best_test_acc
    data['best_global_step'] = best_step

    # bad case analysis
    # if exp_config['model_params.model_name'] in [
    #                                              'shared_encoder_2way',
    #                                              'shared_encoder_3way']:

    #     test_badcase = pd.read_csv(os.path.join(
    #         exp_folder, 'test_badcase.tsv'), sep='\t')

    #     dev_badcase = pd.read_csv(os.path.join(
    #         exp_folder, 'dev_badcase.tsv'), sep='\t')

    #     test_badcase = test_badcase[test_badcase.global_step == best_step]
    #     num_neural_badcase = np.sum(test_badcase.nlicls_correct_vec == False)
    #     num_symbolic_badcase = np.sum(
    #         test_badcase.symbolic_correct_vec == False)
    #     both_bad_case = np.sum((test_badcase.symbolic_correct_vec ==
    #                             False) & (test_badcase.nlicls_correct_vec == False))

    #     data['num_neural_wrong'] = num_neural_badcase
    #     data['num_symbolic_wrong'] = num_symbolic_badcase
    #     data['both_wrong'] = both_bad_case

    #     test_files = exp_config['test_dataset.filename_list']
    #     num_test_cases = 0
    #     for tf in test_files:
    #         num_test_cases += len(load_jsonl(tf))

    #     data['num_neural_right'] = num_test_cases - num_neural_badcase
    #     data['num_symbolic_right'] = num_test_cases - num_symbolic_badcase
    #     data['num_both_right'] = \
    #         num_test_cases - num_neural_badcase - num_symbolic_badcase + both_bad_case
    #     data['num_test_cases'] = num_test_cases
    return data


def main():
    args = parser.parse_args()
    os.chdir(os.getcwd())
    exp_datalist = []
    columns = set()
    target_folder = args.folder
    os.makedirs('failure_config', exist_ok=True)
    f_counter = 0
    for p, dirnames, filenames in os.walk(target_folder):
        for exp_folder_name in dirnames:
            exp_folder = os.path.join(target_folder, exp_folder_name)
            try:
                d = parse_folder(exp_folder)
                d['exp_folder'] = exp_folder
                columns.update(d.keys())
                exp_datalist.append(d)
            except:
                print("fail parse folder", exp_folder)
                config = load_config(os.path.join(exp_folder, 'config.yaml'))
                dump_config(config, os.path.join(
                    'failure_config', f'{f_counter}.yaml'))
                f_counter += 1

    data = defaultdict(list)
    for exp_data in exp_datalist:
        for key in columns:
            data[key].append(exp_data.get(key, None))

    df = pd.DataFrame(data)
    df.to_csv(f"{target_folder}/experiments.csv", index=False)

    summarize_name = 'best_test_acc num_neural_wrong	num_symbolic_wrong	both_wrong	num_neural_right	num_symbolic_right	num_both_right'.split()
    for key, subdf in df.groupby('train_dataset.filename_list'):
        for name in summarize_name:
            ave = np.mean(subdf[name])
            std = np.std(subdf[name])
            print(key, name, ave, std)


if __name__ == "__main__":
    # folder = 'experiment/10fold/20211201-Wed-073724_605184d837e911f141fb294ea883c1a5001d3b86b90684da81e2bd46dec595f2'
    # recover_the_info_bart_forms_3way(dataset_file='dataset/AWPNLI-cv-10-replica-6/dev.jsonl',
    #  badcase_file=os.path.join(folder, 'dev_badcase.tsv'))
    main()
