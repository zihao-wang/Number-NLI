"""
Each model is a sub-class of pytorch nn.Module

The model class name ends with CLS, FORM, and MIX to show it is used
with the necessary properties
    self.tokenizer
and the necessary method
    self.forward
        return the output dict, contains at least the logits of each class
    self.batch_preprocess
and some class method
    cls.compute_loss
        compute the loss function from the output dict by the forward function

"""

import logging
from abc import abstractmethod
from collections import defaultdict
import os

import numpy as np
import torch
from torch import nn
from transformers import (BartForConditionalGeneration,
                          BartForSequenceClassification, BartTokenizer,
                          RobertaConfig, RobertaForSequenceClassification,
                          RobertaTokenizer)

from .executor import nli_label_decider


tokenizer_related_params = {
    "padding": True,
    "truncation": True,
    "max_length": 512,
    "return_tensors": "pt"
}

number_tokens = ["[num:1]", "[num:2]"] + [f"[num:{k}:{i}]" for i in range(
    1, 10) for k in range(1, 2)]

lisp_special_tokens = ["(", ")", ",",
                       "+", "-", "*", "/",
                       "=", ">", "<", "!=", "<=", ">=",
                       "&", "|", "!",
                       "NA", "!ENTAIL"]


class ModelwithTokenizer(nn.Module):
    def __init__(self):
        super(ModelwithTokenizer, self).__init__()

    @abstractmethod
    def compute_loss(self, *args, **kwargs):
        pass

    @abstractmethod
    def compute_pred(self, *args, **kwargs):
        pass

    @abstractmethod
    def compute_info(self, *args, **kwargs):
        pass


class RoBERTa_CLS(ModelwithTokenizer):
    def __init__(self, pretrain_ckpt, **kwargs):
        super(RoBERTa_CLS, self).__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained(pretrain_ckpt)
        self.network = RobertaForSequenceClassification.from_pretrained(
            pretrain_ckpt, **kwargs)

        if 'num_labels' in kwargs:
            self.num_labels = kwargs['num_labels']
        else:
            self.num_labels = 3

    def batch_preprocess(self, batch_dict):
        inputs = self.tokenizer(batch_dict['s1raw'], batch_dict['s2raw'],
                                **tokenizer_related_params)

        if self.training:
            inputs['labels'] = torch.tensor(batch_dict['gold_label'])
        return inputs

    def compute_loss(self):
        return self.outputs.loss

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

    def get_cls_embedding(self):
        return self.outputs.hidden_states[:, 0, :]

    def forward(self, batch_dict):
        self.batch_dict = batch_dict

        inputs = self.batch_preprocess(batch_dict)
        device = self.network.device
        for k in inputs:
            inputs[k] = inputs[k].to(device)

        self.outputs = self.network(**inputs)


class BART_CLS(ModelwithTokenizer):
    def __init__(self, pretrain_ckpt, **kwargs):
        super(BART_CLS, self).__init__()
        self.tokenizer = BartTokenizer.from_pretrained(pretrain_ckpt)
        self.network = BartForSequenceClassification.from_pretrained(
            pretrain_ckpt)

    def batch_preprocess(self, batch_dict):
        inputs = self.tokenizer(batch_dict['s1raw'], batch_dict['s2raw'],
                                **tokenizer_related_params)
        if self.training:
            inputs['labels'] = torch.tensor(batch_dict['gold_label'])
        return inputs

    def compute_loss(self):
        return self.outputs.loss

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

    def forward(self, batch_dict):
        self.batch_dict = batch_dict

        inputs = self.batch_preprocess(batch_dict)
        device = self.network.device
        for k in inputs:
            inputs[k] = inputs[k].to(device)

        self.outputs = self.network(**inputs)


class BART_FORMS(ModelwithTokenizer):

    def __init__(self, pretrain_ckpt, num_labels=3, loss_func_type='seq2seq', beam_size=5, **kwargs):
        super(BART_FORMS, self).__init__()
        self.tokenizer = self.__get_tokenizer(pretrain_ckpt)
        self.entail_bart = BartForConditionalGeneration.from_pretrained(
            pretrain_ckpt)
        self.entail_bart.resize_token_embeddings(len(self.tokenizer))
        self.num_labels = num_labels
        if self.num_labels == 3:
            self.contradict_bart = BartForConditionalGeneration.from_pretrained(
                pretrain_ckpt)
            self.contradict_bart.resize_token_embeddings(len(self.tokenizer))

        self.beam_size = beam_size
        self.loss_func_type = []
        if 'seq2seq' in loss_func_type:
            self.loss_func_type.append('seq2seq')
        if 'grammar_rl' in loss_func_type:
            self.loss_func_type.append('grammar_rl')
        if 'label_rl' in loss_func_type:
            self.loss_func_type.append('label_rl')

    def __get_tokenizer(self,
                        init_checkpoint_path):
        number_tokens = [
            "[num:1]", "[num:2]"] + [
            f"[num:{k}:{i}]" for i in range(1, 10) for k in range(1, 3)]

        lisp_special_tokens = ["(", ")", ",",
                               "add", "sub", "mul", "div",
                               "=", ">", "<", "!=", "<=", ">=",
                               "&", "|", "!",
                               "NA", "!ENTAIL"]
        tokenizer = BartTokenizer.from_pretrained(
            init_checkpoint_path,
            additional_special_tokens=number_tokens + lisp_special_tokens)

        for k, v in zip(tokenizer.additional_special_tokens,
                        tokenizer.additional_special_tokens_ids):
            logging.info(
                f"bart tokenizer adding special token {k} with id {v}")

        # bad_word_ids = list(set(range(len(tokenizer))).difference(set(tokenizer.additional_special_tokens_ids)))
        # bad_words_ids = [[i] for i in bad_word_ids]
        # print(bad_word_ids, bad_words_ids)
        return tokenizer

    def __get_individual_inputs(self, model, lisp_list, sent1_list, sent2_list):

        inputs = self.tokenizer(sent1_list, sent2_list,
                                **tokenizer_related_params)

        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if self.training:
            _labels = self.tokenizer(
                lisp_list, add_special_tokens=True, padding=True, return_tensors='pt').input_ids
        # max_label_length = max(len(l) for l in _labels)
        # padding_side = self.tokenizer.padding_side
        # for label in _labels:
        #     remainder = [-100] * \
        #         (max_label_length - len(label))
        #     labels.append(
        #         label + remainder if padding_side == "right" else remainder + label
        #     )
        # inputs = self.tokenizer.pad(inputs, return_tensors="pt")
            inputs["labels"] = _labels
            inputs["decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(
                labels=_labels)

        return inputs

    def __batch_preprocess_form_sup(self, batch_data):

        sent1_list = batch_data['new_s1numrec']
        sent2_list = batch_data['new_s2numrec']
        entail_lisp_list = batch_data['new_elisp']
        contradict_lisp_list = batch_data['new_clisp']
        contradict_inputs = {}
        entail_inputs = self.__get_individual_inputs(
            self.entail_bart, entail_lisp_list, sent1_list, sent2_list)
        if self.num_labels == 3:
            contradict_inputs = self.__get_individual_inputs(
                self.contradict_bart, contradict_lisp_list, sent1_list, sent2_list)

        return entail_inputs, contradict_inputs

    def __batch_preprocess_inference(self, batch_data):
        inputs = self.tokenizer(batch_data['s1numrec'], batch_data['s2numrec'],
                                **tokenizer_related_params)
        return inputs

    def decode_generation_to_str_list(self, generated_ids):
        size = len(generated_ids)
        ans = []
        for j in range(size):
            s = self.tokenizer.decode(
                generated_ids[j].squeeze(), skip_special_tokens=False)
            s = s.replace("<s>", "").replace(
                "<pad>", "").replace("</s>", "").replace(" ", "")
            ans.append(s)
        return ans

    def beam_search_generation(self, entail_inputs, contradict_inputs):
        entail_outputs = {}
        contradict_outputs = {}

        def gen_routine(bart, inputs, outputs):
            beam_search_outputs = bart.generate(
                input_ids=inputs.input_ids,
                num_beams=self.beam_size,
                num_return_sequences=self.beam_size,
                output_scores=True,
                output_hidden_states=True,
                return_dict_in_generate=True)

            decoded_str_list = self.decode_generation_to_str_list(
                beam_search_outputs.sequences)
            size = len(decoded_str_list)
            outputs['beam_str_list'] = decoded_str_list
            outputs['beam_logprob'] = beam_search_outputs.sequences_scores
            outputs['formula_list'] = [
                decoded_str_list[i] for i in range(0, size, self.beam_size)]

        gen_routine(self.entail_bart, entail_inputs, entail_outputs)
        if self.num_labels == 3:
            gen_routine(self.contradict_bart,
                        contradict_inputs, contradict_outputs)

        return entail_outputs, contradict_outputs

    def compute_loss(self):
        loss = 0
        entail_outputs, contradict_outputs = self.outputs
        if 'seq2seq' in self.loss_func_type:
            loss += entail_outputs.loss + contradict_outputs['loss']

        if 'grammar_rl' in self.loss_func_type:
            entail_grammar_loss = grammar_reinforce(entail_outputs['beam_str_list'],
                                                    entail_outputs['beam_logprob'])
            contradict_grammar_loss = grammar_reinforce(contradict_outputs['beam_str_list'],
                                                        contradict_outputs['beam_logprob'])

            loss += entail_grammar_loss + contradict_grammar_loss

        if 'label_rl' in self.loss_func_type:
            preds = self.compute_pred()
            label = np.asarray(self.batch_dict['gold_label'])
            correct_vec = preds == label
            loss += correct_reinforce(
                correct_vec, entail_outputs['beam_logprob'], contradict_outputs['beam_logprob'])

        return loss

    def compute_pred(self):
        eformlist = self.outputs[0]['formula_list']
        cformlist = self.outputs[1].get(
            'formula_list', ['!ENTAIL'] * len(eformlist))
        numdictlist = self.batch_dict['num_dict']
        logit = []
        prediction_label_list = []
        for eform, cform, num_dict in zip(eformlist, cformlist, numdictlist):
            num_dict = {k: num_dict[k]['v'] for k in num_dict}
            predict_label = nli_label_decider(
                eform, cform, num_dict)
            prediction_label_list.append(predict_label)

            if predict_label == 'undecidable':
                logit.append([2/7, 3/7, 2/7])
            elif predict_label == 'contradiction':
                logit.append([1, 0, 0])
            elif predict_label == 'neutral':
                logit.append([0, 1, 0])
            elif predict_label == 'entailment':
                logit.append([0, 0, 1])
            else:
                raise NotImplementedError

        self.logits = torch.tensor(np.asarray(logit),
                                   dtype=torch.float,
                                   device=self.entail_bart.device)

        pred_label_index = []
        for l in prediction_label_list:
            if l == 'contradiction':
                idx = 0
            elif l == 'entailment' and self.num_labels == 2:
                idx = 1
            elif l == 'entailment' and self.num_labels == 3:
                idx = 2
            else:
                idx = 1
            pred_label_index.append(idx)
        return np.asarray(pred_label_index)

    def compute_info(self):
        preds = self.compute_pred()
        label = np.asarray(self.batch_dict['gold_label'])
        nli_correct_vec = preds == label
        size = len(label)

        correct_vec = []
        batch_info = {}
        batch_info['pE'] = self.outputs[0]['formula_list']
        batch_info['tE'] = self.batch_dict['elisp']
        batch_info['pC'] = self.outputs[1].get(
            'formula_list', ['!ENTAIL'] * len(label))
        batch_info['tC'] = self.batch_dict['clisp']

        for i in range(size):
            if batch_info['pE'][i] == batch_info['tE'][i] and batch_info['pC'][i] == batch_info['pC'][i]:
                correct_vec.append(1)
            else:
                correct_vec.append(0)

        batch_info['correct_vec'] = np.asarray(correct_vec)
        batch_info['nli_correct_vec'] = nli_correct_vec
        batch_info['prediction'] = preds
        batch_info['labels'] = label
        batch_info['case_id'] = self.batch_dict['case_id']
        return batch_info

    def forward(self, batch_dict):
        """
        The input is the transformer tokenizer wrapped thing
        The output is two transformer output fields when self.training is train_neural_NLI
        The output is two strings when self.training is false
        """
        self.device = self.entail_bart.device
        self.batch_dict = batch_dict

        if self.training:
            entail_inputs, contradict_inputs = self.__batch_preprocess_form_sup(
                batch_dict)

            self.entail_inputs, self.contradict_inputs = entail_inputs, contradict_inputs

            for k in entail_inputs:
                entail_inputs[k] = entail_inputs[k].to(self.device)
            entail_outputs = self.entail_bart(**entail_inputs)

            contradict_outputs = {}

            if self.num_labels == 3:
                for k in contradict_inputs:
                    contradict_inputs[k] = contradict_inputs[k].to(self.device)
                contradict_outputs = self.contradict_bart(**contradict_inputs)

            entail_beam_search_outputs, contradict_beam_search_outputs = \
                self.beam_search_generation(entail_inputs, contradict_inputs)
            for k in entail_beam_search_outputs:
                entail_outputs[k] = entail_beam_search_outputs[k]

            if self.num_labels == 3:
                for k in contradict_beam_search_outputs:
                    contradict_outputs[k] = contradict_beam_search_outputs[k]

            if 'loss' not in contradict_outputs:
                contradict_outputs['loss'] = 0

        else:
            inputs = self.__batch_preprocess_inference(batch_dict)
            for k in inputs:
                inputs[k] = inputs[k].to(self.device)

            entail_outputs, contradict_outputs = \
                self.beam_search_generation(inputs, inputs)

        self.outputs = (entail_outputs, contradict_outputs)


class RoBERTa_MIX(ModelwithTokenizer):
    def __init__(self, pretrain_ckpt, mode='mix') -> None:
        super(RoBERTa_MIX, self).__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained(pretrain_ckpt)
        self.mode = mode
        if mode == 'mix':
            self.network = RobertaForSequenceClassification.from_pretrained(
                pretrain_ckpt, num_labels=1)
        elif mode == 'regression':
            self.network = RobertaForSequenceClassification.from_pretrained(
                pretrain_ckpt, num_labels=21)

        self.loss_fct = nn.CrossEntropyLoss()

    def batch_preprocess(self, batch_dict):
        inputs = self.tokenizer(batch_dict['s1raw'], batch_dict['s2raw'],
                                **tokenizer_related_params)
        return inputs

    def forward(self, batch_dict):
        """ For the unified interface, the logits1 and logits2 are also in the batch_dict
        """
        device = self.network.device
        self.batch_dict = batch_dict
        inputs = self.batch_preprocess(batch_dict)
        for k in inputs:
            inputs[k] = inputs[k].to(device)
        self.outputs = self.network(**inputs)

    def compute_loss(self):
        device = self.network.device
        labels = torch.tensor(
            self.batch_dict['gold_label'], device=device)

        self.compute_pred()

        logits = self.outputs['final_logits']
        loss = self.loss_fct(logits, labels)
        self.outputs['loss'] = loss
        return loss

    def compute_pred(self):
        device = self.network.device
        logits1 = self.batch_dict['logits1'].to(device)
        logits2 = self.batch_dict['logits2'].to(device)
        if self.mode == 'mix':
            mix_logits = torch.sigmoid(self.outputs.logits)
            logits = logits1 * mix_logits + (1-mix_logits) + logits2
        elif self.mode == 'regression':
            mix_weights = torch.reshape(
                self.outputs.logits[:, :18], [-1, 6, 3])
            mix_bias = torch.reshape(self.outputs.logits[:, 18:], [-1, 3])
            input_logits = torch.cat([logits1, logits2], -1).reshape(-1, 6, 1)
            logits = torch.sum(
                torch.mul(input_logits, mix_weights), dim=-2) + mix_bias
            mix_logits = (mix_weights, mix_bias)
        else:
            raise NotImplementedError

        self.outputs['final_logits'] = logits
        self.outputs['mix_logits'] = mix_logits
        return logits.argmax(-1).cpu().numpy()

    def compute_info(self):
        preds = self.compute_pred()
        label = np.asarray(self.batch_dict['gold_label'])
        correct_vec = preds == label

        batch_info = {}
        # batch_info['batch_loss'] = self.compute_loss().item()
        # batch_info['correct_count'] = np.sum(correct_vec)
        batch_info['prediction'] = preds
        batch_info['labels'] = label
        batch_info['case_id'] = self.batch_dict['case_id']
        batch_info['mix_logits'] = self.outputs['mix_logits']
        batch_info['correct_vec'] = correct_vec
        batch_info['nli_correct_vec'] = correct_vec
        return batch_info

def get_model_with_tok(model_name, init_ckpt=None, pretrain_ckpt=None, device='cpu', **kwargs):
    # load with pretrain
    model_name = model_name.lower()

    if model_name == "bart_cls":
        if pretrain_ckpt is None:
            pretrain_ckpt = "ckpt_lm/bart-large-mnli"
        model = BART_CLS(pretrain_ckpt, **kwargs)
    elif model_name == "roberta_cls":
        if pretrain_ckpt is None:
            pretrain_ckpt = "ckpt_lm/roberta-large-mnli"
        model = RoBERTa_CLS(pretrain_ckpt, **kwargs)
    elif model_name == "bart_forms":
        if pretrain_ckpt is None:
            pretrain_ckpt = "ckpt_lm/bart-large"
        model = BART_FORMS(pretrain_ckpt, **kwargs)
    else:
        raise NotImplementedError("Not Implemented Models")
    # to proper device
    model.to(device)
    return model