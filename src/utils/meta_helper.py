import json
import os
import shutil
import subprocess
from hashlib import sha256
from typing import Dict

import yaml

template_config_path = "config/template.yaml"
version_file = "current_version"


def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()


def load_config(config_file_path) -> Dict:
    with open(config_file_path, 'rt') as f:
        config_dict = yaml.safe_load(f)
    return config_dict


def dump_config(config_dict: Dict, config_file_path) -> None:
    with open(config_file_path, 'wt') as f:
        yaml.safe_dump(config_dict, f)


def get_exp_id(config_dict, version_file="current_version") -> str:
    """
    do the hash from the json str and version string
    """
    m = sha256()
    m.update(json.dumps(config_dict).encode())
    if os.path.exists(version_file):
        with open(version_file, 'rt') as f:
            for s in f.readlines():
                m.update(s.strip().encode())
    else:
        m.update(get_git_revision_hash().encode())
    exp_id = m.hexdigest()
    return exp_id


def prepare_the_config_file(config_dict, exp_folder):
    dump_config(config_dict, os.path.join(exp_folder, "config.yaml"))


def prepare_the_version_file(exp_folder):
    target_path = os.path.join(exp_folder, "VERSION")
    if os.path.exists(version_file):
        shutil.copyfile(version_file, target_path)
    else:
        git_hash = get_git_revision_hash()
        with open(target_path, 'wt') as f:
            f.write(f"revision:{git_hash}\n")


def set_status(exp_folder, status):
    target_path = os.path.join(exp_folder, "STATUS")

    if status.lower() == "partial":
        with open(target_path, 'wt') as f:
            f.write("PARTIAL\n")

    if status.lower() == "finished":
        assert os.path.exists(target_path)
        with open(target_path, 'rt') as f:
            assert f.read().strip() == "PARTIAL"
        with open(target_path, 'wt') as f:
            f.write("FINISHED\n")


def set_best_ckpt(exp_folder, best_ckpt):
    target_path = os.path.join(exp_folder, "BESTCKPT")

    with open(target_path, 'wt') as f:
        f.write(f"{best_ckpt}\n")


if __name__ == "__main__":
    print(get_git_revision_hash())
    config_dict = load_config(template_config_path)
    print(config_dict)
    exp_id = get_exp_id(config_dict)
    print(exp_id)
