from collections import defaultdict
import logging

from torch.utils.data import Dataset, DataLoader

from .data_helper import load_jsonl
# from .numerize import Numerizer


label2id = {
    "contradiction": 0,
    "neutral": 1,
    "entailment": 2
}

label2id_2way = {
    "contradiction": 0,
    # "neutral": 1,
    "entailment": 1
}


class MultiFileDataset(Dataset):
    def __init__(self, filename_list, num_labels=3, max_num_samples=-1):
        self.dataset = []
        _datalist = []
        for f in filename_list:
            logging.info(f"load file from {f}")
            datalist = load_jsonl(f)
            _datalist.extend(datalist)
        if max_num_samples > 0:
            _datalist = _datalist[:max_num_samples]

        for row in _datalist:
            try:
                if num_labels == 3:
                    row['gold_label'] = label2id[row['gold_label']]
                if num_labels == 2:
                    row['gold_label'] = label2id_2way[row['gold_label']]
                self.dataset.append(row)
            except:
                pass

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


def collate_dict(features):
    batch_data = defaultdict(list)
    for feature in features:
        for k in feature:
            batch_data[k].append(feature[k])
    return batch_data


def get_dataloader(filename_list, shuffle, batch_size, num_labels):
    dataset = MultiFileDataset(filename_list, num_labels)
    return DataLoader(dataset,
                      shuffle=shuffle,
                      batch_size=batch_size,
                      collate_fn=collate_dict)


"""
class NLIDataset(Dataset):
    def __init__(self, datalist):
        self.dataset = datalist

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        return {
            'sentence1': row['s1raw'],
            'sentence2': row['s2raw'],
            'labels': label2id[row['gold_label']]
        }


@dataclass
class NLICollator:
    tokenizer: PreTrainedTokenizerBase
    padding = True
    max_length = None
    pad_to_multiple_of = None
    train: bool = False

    def __call__(self, features):
        s1, s2, l = [], [], []
        for item in features:
            s1.append(item['sentence1'])
            s2.append(item['sentence2'])
            l.append(item['labels'])

        inputs = self.tokenizer(s1, s2,
                                padding=self.padding,
                                max_length=self.max_length,
                                pad_to_multiple_of=self.pad_to_multiple_of,
                                return_tensors='pt')
        labels = np.asarray(l)
        meta = {"labels": labels}
        if self.train:
            inputs['labels'] = torch.tensor(labels)
        else:
            meta['features'] = features
        return inputs, meta


class Seq2DualLISPDataset(Dataset):
    def __init__(self, datalist):
        self.dataset = datalist

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        return {
            'sentence1': row['s1numrec'],
            'sentence2': row['s2numrec'],
            'elisp': row['elisp'],
            'clisp': row['clisp'],
            'num_dict': row['num_dict'],
            'case_id': row['case_id']}


@dataclass
class Seq2DualLISPCollator:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def get_inputs(self, lisp_list, sent1_list, sent2_list):

        inputs = self.tokenizer.batch_encode_plus(
            sent1_list, sent2_list, return_attention_mask=True)

        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.

        labels = []
        _labels = self.tokenizer(
            lisp_list, add_special_tokens=True, padding=False).input_ids
        max_label_length = max(len(l) for l in _labels)
        padding_side = self.tokenizer.padding_side
        for label in _labels:
            remainder = [self.label_pad_token_id] * \
                (max_label_length - len(label))
            labels.append(
                label + remainder if padding_side == "right" else remainder + label
            )
        inputs['labels'] = labels
        inputs = self.tokenizer.pad(inputs,
                                    padding=self.padding,
                                    max_length=self.max_length,
                                    pad_to_multiple_of=self.pad_to_multiple_of,
                                    return_tensors="pt",)

        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(
                labels=inputs["labels"])
            inputs["decoder_input_ids"] = decoder_input_ids
        return inputs

    def __call__(self, features):

        sent1_list = []
        sent2_list = []
        entail_lisp_list = []
        contradict_lisp_list = []
        num_dict_list = []
        case_id_list = []
        for f in features:
            sent1_list.append(f['sentence1'])
            sent2_list.append(f['sentence2'])
            entail_lisp_list.append(f['elisp'].replace(" ", ""))
            contradict_lisp_list.append(f['clisp'])
            num_dict_list.append(f['num_dict'])
            case_id_list.append(f['case_id'])

        entail_inputs = self.get_inputs(
            entail_lisp_list, sent1_list, sent2_list)
        contradict_inputs = self.get_inputs(
            contradict_lisp_list, sent1_list, sent2_list)

        input_dict = {"entail_inputs": entail_inputs,
                      "contradict_inputs": contradict_inputs}
        meta = {"entail_lisp": entail_lisp_list,
                "contradict_lisp": contradict_lisp_list,
                "num_dict": num_dict_list,
                'case_id': case_id_list}
        return input_dict, meta


@dataclass
class NumNLICollator:
    tokenizer: PreTrainedTokenizerBase
    numerizer: Numerizer
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    train: bool = False

    def __call__(self, features):

        sent1_list, sent2_list, l = [], [], []
        for f in features:
            sent1_list.append(f['sentence1'])
            sent2_list.append(f['sentence2'])
            l.append(f['labels'])
        doc1_list = self.numerizer.pipe(sent1_list)
        doc2_list = self.numerizer.pipe(sent2_list)

        # get meta information
        batch_sentence1 = []
        batch_sentence2 = []
        inputs = defaultdict(list)
        meta = defaultdict(list)
        for doc1, doc2 in zip(doc1_list, doc2_list):
            number_dict = {}
            tokens = []
            number_counts = 0
            for t in doc1:
                if t._.is_number:
                    number_counts += 1
                    num_key = f"M{number_counts}"
                    number_dict[num_key] = t._.value
                    tokens.append(num_key)
                else:
                    tokens.append(t.text)
            batch_sentence1.append(" ".join(tokens))
            meta["sentence1_replaced"].append(" ".join(tokens))
            meta["sentence1_original"].append(doc1.text)

            tokens = []
            number_counts = 0
            for t in doc2:
                if t._.is_number:
                    number_counts += 1
                    num_key = f"N{number_counts}"
                    number_dict[num_key] = t._.value
                    tokens.append(num_key)
                else:
                    tokens.append(t.text)
            batch_sentence2.append(" ".join(tokens))
            meta["sentence2_replaced"].append(" ".join(tokens))
            meta["sentence2_original"].append(doc2.text)

            meta["num_dicts"].append(number_dict)

        inputs = self.tokenizer.batch_encode_plus(
            batch_sentence1, batch_sentence2, return_attention_mask=True)

        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.

        inputs = self.tokenizer.pad(inputs,
                                    padding=self.padding,
                                    max_length=self.max_length,
                                    pad_to_multiple_of=self.pad_to_multiple_of,
                                    return_tensors="pt",)

        labels = np.asarray(l)
        if self.train:
            inputs['labels'] = torch.tensor(labels)

        return inputs, labels, meta


def get_dataset_class(dataset_type):
    if dataset_type == 'NLI':
        return NLIDataset
    elif dataset_type == 'Seq2DualLISP':
        return Seq2DualLISPDataset


def get_collator_class(collator_type):
    if collator_type == 'NLI':
        return NLICollator
    elif collator_type == 'Seq2DualLISP':
        return Seq2DualLISPCollator

def get_dataloader(shuffle, tok, train, file, batch_size,
                   dataset_type, collator_type):
    from torch.utils.data import DataLoader
    datalist = load_jsonl(file)
    dataset_class = get_dataset_class(dataset_type)
    collator_class = get_collator_class(collator_type)

    dataset = dataset_class(datalist)

    return DataLoader(dataset,
                      shuffle=shuffle,
                      batch_size=batch_size,
                      collate_fn=collator_class(tok, train))
"""
