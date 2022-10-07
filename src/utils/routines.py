"""
The routines for train and evaluate the model
"""
from abc import abstractmethod
import logging
import json
from utils.recorder import Recorder
from torch.utils.data.dataloader import DataLoader
import torch
from tqdm import tqdm

import torch


class TrainStep:
    def __init__(self,
                 max_global_steps: int,
                 dataloader: DataLoader,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 badcase_recorder: Recorder = None,
                 info_recorder: Recorder = None,
                 step_lr_size=None,
                 step_lr_gamma=None,
                 device: str = "cpu"):
        self.max_global_steps = max_global_steps
        self.dataloader = dataloader
        self.model = model
        self.optimizer = optimizer
        if step_lr_size is not None and step_lr_gamma is not None:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=step_lr_size,
                gamma=step_lr_gamma)
        else:
            self.scheduler = None

        self.badcase_recorder = badcase_recorder
        self.info_recorder = info_recorder
        self.device = device

        self.data_iterator = None
        self.states = {}
        self.global_step = 0
        self.epoch = 0

    @abstractmethod
    def _step(self, batch_tuple):
        pass

    @abstractmethod
    def get_epoch_info(self):
        epoch_info = {
            'level': "epoch",
            'epoch': self.epoch,
            'global_step': self.global_step
        }
        return epoch_info

    def save_info(self, info):
        if self.info_recorder is not None:
            self.info_recorder.save_info(info, mode='jsonl')

    def step(self):
        if self.global_step > self.max_global_steps:
            print("finish iteration")

        if self.data_iterator is None:
            self.data_iterator = iter(self.dataloader)

        try:
            batch_dict = next(self.data_iterator)
        except StopIteration:
            self.epoch += 1
            print(f"begin next epoch {self.epoch}")
            logging.debug(json.dumps(self.states))
            epoch_info = self.get_epoch_info()
            self.save_info(epoch_info)
            self.states = {}

            self.data_iterator = iter(self.dataloader)
            batch_dict = next(self.data_iterator)

        print(self.epoch, self.global_step, end='\r')
        self._step(batch_dict)

        if self.scheduler:
            self.scheduler.step()

        self.global_step += 1
        return

class NLITrainStep(TrainStep):

    def _step(self, batch_dict):
        self.model.train()

        self.optimizer.zero_grad()

        self.model(batch_dict)
        loss = self.model.compute_loss()

        loss.backward()
        self.optimizer.step()

        batch_info_dict = self.model.compute_info()

        step_info_dict = {
            "level": "step",
            "global_step": self.global_step,
            "epoch": self.epoch,
            "batch_loss": loss.item(),
        }

        if hasattr(self.model, 'loss') and isinstance(self.model.loss, dict):
            for k, v in self.model.loss.items():
                step_info_dict[f'loss_{k}'] = v

        self.badcase_recorder.save_batch_badcase(
            batch_info_dict, step_info_dict)

        self.save_info(step_info_dict)

        if 'total_loss' in self.states:
            self.states['total_loss'] += loss.item()
        else:
            self.states['total_loss'] = loss.item()

        if 'total_count' in self.states:
            self.states['total_count'] += len(batch_info_dict['case_id'])
        else:
            self.states['total_count'] = len(batch_info_dict['case_id'])

        if 'count' in self.states:
            self.states['count'] += 1
        else:
            self.states['count'] = 1

        logging.debug(json.dumps(self.states))

    @abstractmethod
    def get_epoch_info(self):
        epoch_info = super().get_epoch_info()
        epoch_info['average_loss'] = self.states['total_loss'] / \
            self.states['count']

        return epoch_info


def eval_neural_NLI(meta_info_dict, dataloader, model, device, info_recorder, badcase_recorder, max_step=-1):
    model.eval()
    correct_vec_keys = None
    correct_count_dict = {}
    total_count = 0
    with tqdm(enumerate(dataloader), desc=f"{meta_info_dict}") as t:
        for i, batch_dict in t:
            if max_step > 0 and i > max_step:
                break

            model(batch_dict)
            batch_info_dict = model.compute_info()
            if correct_vec_keys is None:
                correct_vec_keys = []
                for k in batch_info_dict:
                    if k.endswith('correct_vec'):
                        correct_vec_keys.append(k)
                        correct_count_dict[k] = 0

            for key in correct_vec_keys:
                correct_vec = batch_info_dict[key]
                correct_count_dict[key] += correct_vec.sum()

            total_count += len(batch_info_dict[correct_vec_keys[0]])

            postfix_dict = {"cases": total_count}
            for k in correct_vec_keys:
                postfix_dict[k.replace(
                    'correct_vec', 'accuracy')] = correct_count_dict[k] / total_count

            t.set_postfix(postfix_dict)

            badcase_recorder.save_batch_badcase(
                batch_info_dict, meta_info_dict)

    for k in correct_vec_keys:
        meta_info_dict[k.replace('correct_vec', 'accuracy')
                       ] = correct_count_dict[k] / total_count
    info_recorder.save_info(meta_info_dict)


def get_train_step(step_type,
                   max_global_steps: int,
                   dataloader: DataLoader,
                   model: torch.nn.Module,
                   optimizer: torch.optim.Optimizer,
                   badcase_recorder: Recorder = None,
                   info_recorder: Recorder = None,
                   device: str = "cpu"):
    if step_type.lower() == 'nli':
        return NLITrainStep(max_global_steps,
                            dataloader,
                            model,
                            optimizer,
                            badcase_recorder,
                            info_recorder,
                            device)


def get_eval_routine(routine_type):
    if routine_type.lower() == 'nli':
        return eval_neural_NLI
    else:
        raise NotImplementedError
