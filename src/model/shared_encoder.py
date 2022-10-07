import functools
from abc import abstractmethod
from typing import Dict

import numpy as np
import torch
import transformers
from torch import nn
from transformers.modeling_utils import PreTrainedModel
from transformers.models.bart.tokenization_bart import BartTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer


from .executor import nli_label_decider

def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

# using wonder's beautiful simplification: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


tokenizer_related_params = {
    "padding": True,
    "truncation": True,
    "max_length": 512,
    "return_tensors": "pt"
}

number_tokens = [
    "[num:1]", "[num:2]"] + [
    f"[num:{k}:{i}]" for i in range(1, 10) for k in range(1, 3)]

lisp_special_tokens = ["(", ")", ",",
                       "add", "sub", "mul", "div",
                       "=", ">", "<", "!=", "<=", ">=",
                       "&", "|", "!",
                       "NA", "!ENTAIL"]

additional_special_tokens = number_tokens + lisp_special_tokens


class TaskModel(PreTrainedModel):
    def __init__(self, task_name, backbone_model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        super().__init__(transformers.PretrainedConfig())
        self.task_name = task_name
        self.backbone_transformer = backbone_model
        self.tokenizer = tokenizer

    @abstractmethod
    def forward(self, batch_dict):
        pass

    def compute_loss(self):
        if 'loss' in self.outputs:
            return self.outputs['loss']
        else:
            return torch.tensor(0)

    @abstractmethod
    def compute_info(self, batch_dict):
        pass


def input_to_device(inputs, device):
    for k in inputs:
        inputs[k] = inputs[k].to(device)
    return inputs


class SequenceClassifier(TaskModel):
    def __init__(self, task_name, backbone_model, tokenizer):
        super().__init__(task_name=task_name,
                         backbone_model=backbone_model,
                         tokenizer=tokenizer)

    def forward(self, batch_dict):
        self.batch_dict = batch_dict
        inputs = self.tokenizer(batch_dict['s1raw'],
                                batch_dict['s2raw'],
                                **tokenizer_related_params)
        inputs['labels'] = torch.tensor(batch_dict['gold_label'])
        inputs = input_to_device(inputs, self.backbone_transformer.device)

        self.outputs = self.backbone_transformer(**inputs)
        return self.outputs

    def compute_pred(self):
        self.logits = self.outputs['logits']
        return self.outputs['logits'].argmax(-1).cpu().numpy()

    def compute_info(self):
        preds = self.compute_pred()
        label = np.asarray(self.batch_dict['gold_label'])
        correct_vec = preds == label

        batch_info = {}
        batch_info['prediction'] = preds
        batch_info['labels'] = label
        batch_info['case_id'] = self.batch_dict['case_id']
        batch_info['correct_vec'] = correct_vec
        batch_info['nli_correct_vec'] = correct_vec
        return batch_info


class SequenceGenerator(TaskModel):
    def __init__(self,
                 task_name,
                 backbone_model,
                 tokenizer,
                 bitext_keys=['s1numrec', 's2numrec'],
                 target_key='clisp',
                 num_beams=5):
        super().__init__(task_name=task_name,
                         backbone_model=backbone_model,
                         tokenizer=tokenizer)
        self.bitext_keys = bitext_keys
        self.target_key = target_key
        self.num_beams = num_beams

    def decode_generated_ids_to_str_list(self, generated_ids):
        size = len(generated_ids)
        ans = []
        for j in range(size):
            s = self.tokenizer.decode(
                generated_ids[j].squeeze(), skip_special_tokens=False)
            s = s.replace("<s>", "").replace(
                "<pad>", "").replace("</s>", "").replace(" ", "")
            ans.append(s)
        return ans

    def forward(self, batch_dict, generation=False, **kwargs):
        self.batch_dict = batch_dict

        for btk in self.bitext_keys:
            assert btk in batch_dict

        sent1_list = batch_dict[self.bitext_keys[0]]
        sent2_list = batch_dict[self.bitext_keys[1]]
        inputs = self.tokenizer(sent1_list, sent2_list,
                                **tokenizer_related_params)

        if (not self.training) or \
           (generation is True) or \
           (self.target_key not in batch_dict):
            # trigger the generation mode
            # beam search
            inputs = input_to_device(inputs, self.backbone_transformer.device)

            self.outputs = self.backbone_transformer.generate(
                input_ids=inputs.input_ids,
                num_beams=self.num_beams,
                num_return_sequences=self.num_beams,
                output_scores=True,
                output_hidden_states=True,
                return_dict_in_generate=True,
                **kwargs
            )
            # decode the beam search results
            decoded_str_list = self.decode_generated_ids_to_str_list(
                self.outputs.sequences)

            size = len(decoded_str_list)
            self.outputs['beam_str_list'] = decoded_str_list
            self.outputs['beam_logprob'] = self.outputs.sequences_scores
            self.outputs['generate_list'] = [
                decoded_str_list[i] for i in range(0, size, self.num_beams)]

        elif self.target_key in batch_dict:
            # trigger the seq2seq learning mode with teacher forcing
            target_list = batch_dict[self.target_key]
            _labels = self.tokenizer(
                target_list,
                add_special_tokens=True,
                padding=True,
                return_tensors='pt').input_ids
            inputs['labels'] = _labels
            inputs["decoder_input_ids"] = self.backbone_transformer.prepare_decoder_input_ids_from_labels(
                labels=_labels)

            inputs = input_to_device(inputs, self.backbone_transformer.device)

            self.outputs = self.backbone_transformer(**inputs)

        else:
            raise NotImplementedError

        return self.outputs

    def compute_pred(self):
        if 'generate_list' in self.outputs:
            return self.outputs['generate_list']
        else:
            return None

    def compute_info(self):
        # batch_info = {'loss': self.outputs.loss}
        batch_info = {}

        preds = self.compute_pred()
        if preds:
            label = self.batch_dict[self.target_key]
            correct_list = []
            for p, l in zip(preds, label):
                correct_list.append(p == l)
            batch_info['correct_vec'] = np.asarray(correct_list)
            batch_info['prediction'] = preds
            batch_info['formula'] = label

        return batch_info


# class NumSeqTagger(TaskModel):
#     def __init__(self, *args, **kwargs):
#         super(NumSeqTagger, self).__init__()


class MultiTaskTransformer(PreTrainedModel):
    supported_tasks = {
        'SequenceClassifier': SequenceClassifier,
        'SequenceGenerator': SequenceGenerator
    }

    def __init__(self, shared_encoder, task_models_dict: Dict[str, TaskModel], loss_contribution):
        super().__init__(transformers.PretrainedConfig())
        self.shared_encoder = shared_encoder
        self.task_models_dict = nn.ModuleDict(task_models_dict)
        self.loss_contribution = loss_contribution

    @classmethod
    def create(cls,
               task_params_dict={
                   'nlicls': {
                       'model_type': 'BartForSequenceClassification',
                       'pretrain_ckpt': 'facebook/bart-large-mnli',
                       'task_model': 'SequenceClassifier'
                   },
                   'formgen_e': {
                       'model_type': 'BartForConditionalGeneration',
                       'pretrain_ckpt': 'facebook/bart-large',
                       'task_model': 'SequenceGenerator',
                       'task_model_params': {
                           'target_key': 'elisp'
                       },
                       'loss_contribution': 10
                   },
                   'formgen_c': {
                       'model_type': 'BartForConditionalGeneration',
                       'pretrain_ckpt': 'facebook/bart-large',
                       'task_model': 'SequenceGenerator',
                       'task_model_params': {
                           'target_key': 'clisp'
                       },
                       'loss_contribution': 10
                   },
               }):
        # assumptions
        if 'formgen_c' in task_params_dict:
            assert 'formgen_e' in task_params_dict
        if 'formgen_e' in task_params_dict:
            assert 'formgen_c' in task_params_dict

        # initialize
        shared_encoder = None
        task_models_dict = {}
        loss_contribution = {}
        # get tokenizer
        tokenizer = BartTokenizer.from_pretrained(
            'facebook/bart-large', additional_special_tokens=additional_special_tokens)

        # create the encoders
        for task_name, task_params in task_params_dict.items():
            # prepare the backbone model
            pretrain_ckpt = task_params.get('pretrain_ckpt')
            model_type = task_params['model_type']
            model_type_class = getattr(transformers, model_type)
            # init model
            backbone_model = model_type_class.from_pretrained(pretrain_ckpt)
            backbone_model.resize_token_embeddings(len(tokenizer))

            if shared_encoder is None:
                shared_encoder = rgetattr(
                    backbone_model, cls.get_encoder_attr_name(backbone_model))
            else:
                rsetattr(
                    backbone_model, cls.get_encoder_attr_name(backbone_model), shared_encoder)

            _TaskModelClass = cls.supported_tasks[task_params['task_model']]
            task_model_params = task_params.get('task_model_params', {})
            task_models_dict[task_name] = _TaskModelClass(
                task_name, backbone_model, tokenizer, **task_model_params)

            # loss contribution
            loss_contribution[task_name] = task_params.get(
                'loss_contribution', 1)

        return cls(shared_encoder, task_models_dict, loss_contribution)

    @classmethod
    def get_encoder_attr_name(cls, model):
        """
        The encoder transformer is named differently in each model "architecture".
        This method lets us get the name of the encoder attribute
        """
        model_class_name = model.__class__.__name__
        if model_class_name.startswith("Bert"):
            return "bert"
        elif model_class_name.startswith("Roberta"):
            return "roberta"
        elif model_class_name.startswith("Albert"):
            return "albert"
        elif model_class_name.startswith("Bart"):
            return "base_model.encoder"
        else:
            raise KeyError(f"Add support for new model {model_class_name}")

    def forward(self, batch_dict, task_name=None):
        self.batch_dict = batch_dict
        if task_name:
            return self.task_models_dict[task_name](batch_dict)
        else:
            return {
                k: self.task_models_dict[k](batch_dict)
                for k in self.task_models_dict
            }

    def compute_loss(self):
        loss = 0
        self.loss = {}
        for task_name, task_model in self.task_models_dict.items():
            _loss = task_model.compute_loss()
            self.loss[task_name] = _loss.item()
            loss += _loss * self.loss_contribution[task_name]
        return loss

    def compute_info(self):
        gather_results = {}
        gather_results['labels'] = np.asarray(self.batch_dict['gold_label'])

        for task_name, task_model in self.task_models_dict.items():
            if self.loss_contribution[task_name] == 0:
                continue

            batch_dict = task_model.compute_info()
            for k, v in batch_dict.items():
                if k == 'case_id':
                    gather_results[k] = v
                else:
                    gather_results[f"{task_name}_{k}"] = v

        if 'formgen_e_prediction' in gather_results and 'formgen_e_prediction' in gather_results:
            eformlist = gather_results['formgen_e_prediction']
            cformlist = gather_results.get('formgen_c_prediction', [
                                           '!ENTAIL'] * len(eformlist))
            numdictlist = self.batch_dict['num_dict']

            symbolic_logits, symbolic_prediction = get_symbolic_predictions(
                eformlist, cformlist, numdictlist)

            gather_results['symbolic_logits'] = symbolic_logits
            gather_results['symbolic_prediction'] = symbolic_prediction
            gather_results['symbolic_correct_vec'] = symbolic_prediction == gather_results['labels']

        return gather_results


def get_symbolic_predictions(eformlist, cformlist, numdictlist):
    symbolic_logits = []
    symbolic_prediction = []
    for eform, cform, num_dict in zip(eformlist, cformlist, numdictlist):
        num_dict = {k: num_dict[k]['v'] for k in num_dict}
        predict_label = nli_label_decider(
            eform, cform, num_dict)
        if predict_label == 'undecidable':
            symbolic_logits.append([2/7, 3/7, 2/7])
            symbolic_prediction.append(1)

        elif predict_label == 'contradiction':
            symbolic_logits.append([1, 0, 0])
            symbolic_prediction.append(0)

        elif predict_label == 'neutral':
            symbolic_logits.append([0, 1, 0])
            symbolic_prediction.append(1)

        elif predict_label == 'entailment':
            symbolic_logits.append([0, 0, 1])
            symbolic_prediction.append(2)

        else:
            raise NotImplementedError

    symbolic_logits = np.asarray(symbolic_logits)
    symbolic_prediction = np.asarray(symbolic_prediction)
    return symbolic_logits, symbolic_prediction
