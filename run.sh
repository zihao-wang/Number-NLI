mkdir log

# split datset
python3 src/split_dataset.py  \
        --input_data_file labeled_data/AWPNLI.jsonl  \
        --output_data_folder dataset/AWPNLI  \
        --mode=cv-10

# run cases
for rep in 0 1 2 3 4 5 6 7 8 9;
do
    for model in 'bart_cls_3way' 'roberta_cls_3way' 'bart_forms_3way' 'shared_encoder_3way';
    do
        config_file=config/${model}_dataset=AWPNLI_cv-10_replica=$rep
        echo $config_file
        python3 src/launch_experiment.py -c $config_file --cuda 0 --test
    done
done

# summarize results
python3 src/log_reader.py --folder experiment
