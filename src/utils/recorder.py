import os
import json

# from torch.utils.tenSummsorboard import SummaryWriter

import numpy as np


class Recorder:
    def __init__(self, log_file):
        assert not os.path.exists(log_file)
        # writer_dir = f"{os.path.dirname(log_file)}"
        # print(writer_dir)
        # self.tbwriter = SummaryWriter(writer_dir)
        self.file_handler = open(log_file, 'wt')
        self.schema = None
        self.num = 0

    def close(self):
        self.file_handler.close()

    def _write_tsv(self, data_list):
        self.file_handler.write(
            "\t".join([str(d) for d in data_list]) + "\n"
        )
        self.num += 1
        self.file_handler.flush()

    def _write_jsonl(self, data_dict):
        self.file_handler.write(
            json.dumps(data_dict) + "\n"
        )
        self.num += 1
        self.file_handler.flush()

    def save_tb(self, info_dict):
        for k, v in info_dict.items():
            self.tbwriter.add_scalar(k, v)

    def save_info(self, info_dict: dict, mode='tsv'):

        if mode == 'tsv':
            if self.schema is None:
                self.schema = list(info_dict.keys())
                assert self.num == 0
                self._write_tsv(self.schema)
            self._write_tsv(
                data_list=[info_dict.get(k, None) for k in self.schema]
            )
        else:
            self._write_jsonl(info_dict)

    def save_batch_badcase(self, batch_info_dict, meta_info_dict, mode='tsv'):
        correct_vec_labels = []
        for k in batch_info_dict:
            if k.endswith('correct_vec'):
                correct_vec_labels.append(k)

        correct_vec = np.ones(
            len(batch_info_dict[correct_vec_labels[0]])
        )

        for k in correct_vec_labels:
            correct_vec *= np.asarray(batch_info_dict[k])

        size = len(correct_vec)
        for i in range(size):
            if correct_vec[i] == 0:
                badcase_info = {}
                badcase_info.update(meta_info_dict)
                for k in batch_info_dict:
                    badcase_info[k] = batch_info_dict[k][i]
                self.save_info(badcase_info, mode=mode)


class RecorderGroup:
    def __init__(self, name_file_dict):
        self.recorder_dict = {}
        for k in name_file_dict:
            self.recorder_dict[k] = Recorder(log_file=name_file_dict[k])

    def __getattr__(self, name):
        return self.recorder_dict[name]

    def close(self):
        for k in self.recorder_dict:
            self.recorder_dict[k].close()


def get_recorders(recorder_name_file_dict):
    return RecorderGroup(recorder_name_file_dict)
