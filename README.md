# Neural Symbolic Number Reasoning for Natural Language Inference

NLI task implementation for paper
[A Neural-Symbolic Approach to Natural Language Understanding](https://arxiv.org/abs/2203.10557)

## Usage

0. install the dependency in `requirement.txt`
1. install the spacy small en language model.
2. split the labeled AWP dataset, see `labeled_data/AWPNLI.jsonl`
```
python3 src/split_dataset.py  \
        --input_data_file labeled_data/AWPNLI.jsonl  \
        --output_data_folder dataset/AWPNLI  \
        --mode=cv-10
```
3. run the specific model on the specific split by setting the config file in the `config` folder. e.g.
```
python3 src/launch_experiment.py -c $config_file --cuda $cuda
```
4. gather results by in the default output folder `experiment`
```
python3 src/log_reader.py --folder experiment
```

Step 2-4 can also be found in `run.sh` file

## Support model
- `{bart/roberta}_cls_3way`: neural NLI classifier
- `bart_forms_3way`: symbolic execution
- `shared_encoder_3way`: neural symbolic with shared encoder


## Citation

```
@inproceedings{liu2022a,
  abbr={Findings-EMNLP},
  title={A Neural-Symbolic Approach to Natural Language Understanding},
  author={Liu, Zhixuan and Wang, Zihao and Lin, Yuan and Li, Hang},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2022},
  year={2022}
}
```
