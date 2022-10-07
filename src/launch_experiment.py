import argparse
import logging
import os
import os.path as osp
from datetime import datetime
from shutil import rmtree

import torch

from model import get_model
from utils.dataset import get_dataloader
from utils.meta_helper import (get_exp_id, load_config,
                               prepare_the_config_file,
                               prepare_the_version_file,
                               set_status, set_best_ckpt)
from utils.routines import TrainStep, get_eval_routine, get_train_step
from utils.recorder import get_recorders


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file",
                        type=str, default="config/template.yaml")
    parser.add_argument("-t", "--test",
                        action="store_true")
    parser.add_argument("--cuda", default=0, type=int)
    return parser


if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()
    os.chdir(os.getcwd())

    # preprare the config
    config_dict = load_config(args.config_file)
    time_now = datetime.now().strftime("%Y%m%d-%a-%H%M%S")
    config_dict['start_time'] = time_now

    exp_id = get_exp_id(config_dict)
    config_dict['exp_id'] = exp_id

    folder_name = f"{time_now}_{exp_id}"

    # prepare the experiment folder
    exp_folder = osp.join("experiment", folder_name)
    os.makedirs(exp_folder)
    prepare_the_config_file(config_dict, exp_folder)
    prepare_the_version_file(exp_folder)
    set_status(exp_folder, "partial")

    # prepare the ckpt folder
    ckpt_folder = osp.join("checkpoint", folder_name)
    os.makedirs(ckpt_folder)

    # prepare the logger
    logging.basicConfig(filename=osp.join(exp_folder, 'exp.log'),
                        filemode='wt',
                        level=logging.DEBUG)  # TODO: maybe better formatting
    logging.info(config_dict)
    recorder_params = config_dict['recorder_params']
    recorder_name_file_dict = {
        name: osp.join(exp_folder, f"{name}.tsv") for name in recorder_params['names']
    }
    recorders = get_recorders(recorder_name_file_dict)

    # prepare the model
    if args.cuda >= 0 and torch.cuda.is_available():
        device = f"cuda:{args.cuda}"
    else:
        device = "cpu"
    model_params = config_dict['model_params']
    model = get_model(**model_params, device=device)

    # prepare the dataloaders
    dataset_params = config_dict['dataset_params']
    if 'train_dataset' in dataset_params:
        train_params = dataset_params['train_dataset']
        train_dataloader = get_dataloader(**train_params)
    else:
        train_dataloader = None

    if 'dev_dataset' in dataset_params:
        dev_params = dataset_params['dev_dataset']
        dev_params['batch_size'] = 8
        dev_dataloader = get_dataloader(**dev_params)
    else:
        dev_dataloader = None

    if 'test_dataset' in dataset_params:
        test_params = dataset_params['test_dataset']
        test_params['batch_size'] = 8
        test_dataloader = get_dataloader(**test_params)
    else:
        test_dataloader = None

    # learning
    learning_params = config_dict['learning_params']

    optimizer_class = getattr(torch.optim, learning_params["optimizer_name"])
    print(model)
    learning_params["optimizer_params"]["lr"] = float(learning_params["optimizer_params"]["lr"])
    optimizer = optimizer_class(
        model.parameters(), **learning_params["optimizer_params"])

    max_global_steps = learning_params['max_global_steps']
    if device == 'cpu':
        max_global_steps = 1
    eval_every = learning_params['eval_every']
    save_every = learning_params['save_every']

    if train_dataloader is not None:
        train_stepper = get_train_step(step_type=learning_params['train_step_type'],
                                       max_global_steps=max_global_steps,
                                       dataloader=train_dataloader,
                                       model=model,
                                       optimizer=optimizer,
                                       info_recorder=recorders.train_info,
                                       badcase_recorder=recorders.train_badcase,
                                       device=device)
        logging.debug("get the train stepper")
    else:
        train_stepper = None

    if dev_dataloader is not None or test_dataloader is not None:
        eval_routine = get_eval_routine(learning_params['eval_routine_type'])

    best_dev_acc = 0

    if train_stepper is not None:
        logging.debug("begin training")
        while train_stepper.global_step < max_global_steps:
            train_stepper.step()
            if train_stepper.global_step % eval_every == 0:
                dev_meta_info_dict = {'global_step': train_stepper.global_step}
                eval_routine(dev_meta_info_dict, dev_dataloader, model,
                             device, recorders.dev_info, recorders.dev_badcase)
                test_meta_info_dict = {
                    'global_step': train_stepper.global_step}
                eval_routine(test_meta_info_dict, test_dataloader, model,
                             device, recorders.test_info, recorders.test_badcase)

            if train_stepper.global_step % save_every == 0:
                ckpt_path = osp.join(
                    ckpt_folder, f"{train_stepper.global_step}.ckpt")
                torch.save(model.state_dict(), ckpt_path)
                logging.info(f"save ckpt to {ckpt_path}")
                if best_dev_acc < dev_meta_info_dict.get('nli_accuracy', -1):
                    best_dev_acc = dev_meta_info_dict['nli_accuracy']
                    set_best_ckpt(exp_folder, ckpt_path)

            if args.test:
                dev_meta_info_dict = {'global_step': train_stepper.global_step}
                eval_routine(dev_meta_info_dict, dev_dataloader, model,
                             device, recorders.dev_info, recorders.dev_badcase)
                print("one step done")
                exit()

    # process finished
    recorders.close()
    set_status(exp_folder, "finished")

    # sync to hdfs storage if necessary
    rmtree(ckpt_folder)
