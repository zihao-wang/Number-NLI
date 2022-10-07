from .model_with_tok import get_model_with_tok
from .shared_encoder import MultiTaskTransformer


def get_model(model_name, device):
    if model_name == 'bart_cls_3way':
        return get_model_with_tok(model_name='bart_cls', pretrain_ckpt='facebook/bart-large-mnli').to(device)

    elif model_name == 'roberta_cls_3way':
        return get_model_with_tok(model_name='roberta_cls', pretrain_ckpt='roberta-large-mnli').to(device)

    elif model_name == 'bart_forms_3way':
        return get_model_with_tok(model_name='bart_forms', pretrain_ckpt='facebook/bart-base', num_labels=3).to(device)

    elif model_name == 'shared_encoder_3way':
        return MultiTaskTransformer.create(task_params_dict={
            'nlicls': {
                'model_type': 'BartForSequenceClassification',
                'pretrain_ckpt': 'facebook/bart-base',
                'task_model': 'SequenceClassifier'
            },
            'formgen_e': {
                'model_type': 'BartForConditionalGeneration',
                'pretrain_ckpt': 'facebook/bart-base',
                'task_model': 'SequenceGenerator',
                'task_model_params': {
                    'target_key': 'elisp'
                },
                'loss_contribution': 10
            },
            'formgen_c': {
                'model_type': 'BartForConditionalGeneration',
                'pretrain_ckpt': 'facebook/bart-base',
                'task_model': 'SequenceGenerator',
                'task_model_params': {
                    'target_key': 'clisp'
                },
                'loss_contribution': 10
            },
        }).to(device)
