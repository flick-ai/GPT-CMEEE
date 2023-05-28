from torch.utils.data import Dataset
from os.path import join
import json
from itertools import repeat
import torch
import numpy as np
from transformers import BartConfig, BartPretrainedModel, BartModel

NER_PAD, NO_ENT = '[PAD]', 'O'
LABEL  = ['dep', 'equ', 'mic', 'ite', 'dru', 'pro', 'sym', 'dis', 'bod']
_LABEL_RANK = {L: i for i, L in enumerate(['dep', 'equ', 'mic', 'ite', 'dru', 'pro', 'sym', 'dis', 'bod'])}
LABEL2TYPE = {'bod':'身体', 'dru':'药物', 'dis':'疾病', 'dep':'科室', 'equ':'医疗设备', 'mic':'微生物类', 'ite':'医学检验项目', 'pro':'医疗程序', 'sym':'临床表现'}
EE_id2label  = [NER_PAD, NO_ENT] + [f"{P}-{L}" for L in LABEL  for P in ("B", "I")]
EE_label2id  = {b: a for a, b in enumerate(EE_id2label)}
EE_NUM_LABELS  = len(EE_id2label)

def get_id_start_end(labels_or_preds: np.ndarray,start: int, id2label:dict) -> int:
    id = start + 1
    label = id2label[labels_or_preds[start]]
    while id < len(labels_or_preds) and label.startswith('I') :
        id += 1
        label = id2label[labels_or_preds[id]]
    return id

def get_type(entity: np.ndarray, id2label: dict) -> str:
    types = np.array(list(map(lambda x: id2label[x].split('-')[-1], entity)))
    items, counts = np.unique(types, return_counts=True)
    t = np.nonzero(counts == counts.max())[0]
    if len(t) == 1:
        return items[t[0]]
    else :
        max_freq = -1
        max_t = ''
        for c in items[t]:
            freq = _LABEL_RANK[c]
            if freq > max_freq:
                max_freq = freq
                max_t = c

def extract_entities(batch_labels_or_preds: np.ndarray):

    batch_labels_or_preds[batch_labels_or_preds == -100] = EE_label2id[NER_PAD]
    id2label = EE_id2label
    batch_entities = []

    for labels_or_preds in batch_labels_or_preds:
        id = 0
        entities = []
        while id < len(labels_or_preds):
            if labels_or_preds[id] == 1:
                id += 1
                continue
            if labels_or_preds[id] == 0:
                break

            label = EE_id2label[labels_or_preds[id]]

            if label.startswith('B'):
                start = id
                end = get_id_start_end(labels_or_preds, start, id2label)
                entity = labels_or_preds[start:end]
                # get the type
                etype = get_type(entity, id2label)
                entities.append((start, end - 1, etype))
                id = end
                continue

            else:
                id += 1
        batch_entities.append(entities)
    return batch_entities

def str2label(text, entities):
    label2id = EE_label2id
    max_length = 512
    label = [NO_ENT] * len(text)

    def _write_label(_label: list, _type: str, _start: int, _end: int):
        for i in range(_start, _end + 1):
            if i == _start:
                _label[i] = f"B-{_type}"
            else:
                _label[i] = f"I-{_type}"

    for entity in entities:
        start_idx = entity["start_idx"]
        end_idx = entity["end_idx"]
        entity_type = entity["type"]
        assert entity["entity"] == text[start_idx: end_idx + 1], f"{entity} mismatch: `{text}`"

    _write_label(label, entity_type, start_idx, end_idx)
    import ipdb
    ipdb.set_trace()

def f1_score(prediction, target):
    pass
class BartForScore(BartPretrainedModel):
    config_class = BartConfig
    base_model_prefix = "model"

    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.config = config
        self.model = BartModel(config)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            labels2=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            no_decode=False,
    ):
        sequence_output = self.model(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )[0]
        sequence_output = torch.mean(sequence_output, dim=1).squeeze()
        return sequence_output

class InputExample:
    def __init__(self, sentence_id: str, text: str, entities = None):
        self.sentence_id = sentence_id
        self.text_origin = text
        self.entities = entities
        self.text = text

    def to_score_task(self, for_nested_ner: bool = False):
        if self.entities is None:
            return self.sentence_id, self.text
        else:

            label = [NO_ENT] * len(self.text)

            def _write_label(_label: list, _type: str, _start: int, _end: int):
                for i in range(_start, _end + 1):
                    if i == _start:
                        _label[i] = f"B-{_type}"
                    else:
                        _label[i] = f"I-{_type}"

            for entity in self.entities:
                start_idx = entity["start_idx"]
                end_idx = entity["end_idx"]
                entity_type = entity["type"]
                assert entity["entity"] == self.text[start_idx: end_idx + 1], f"{entity} mismatch: `{self.text}`"

            _write_label(label, entity_type, start_idx, end_idx)
            return self.sentence_id, self.text, label

    def to_gpt_task(self):
        base = "|实体|类别|起始位置|终止位置|\n"
        line = "|-----|-----|-----|-----|\n"
        base = base + line

        for entity in self.entities:
            entity_text = entity['entity']
            start_idx = entity["start_idx"]
            end_idx = entity["end_idx"]
            entity_type = entity["type"]
            target = '|'+entity_text+'|'+LABEL2TYPE[entity_type]+'|'+str(start_idx)+'|'+str(end_idx)+'|\n'
            base = base + target

        return self.text, base



class EEDataloader:
    def __init__(self, cblue_root: str="./"):
        self.cblue_root = cblue_root
        self.data_root = join(cblue_root, "CMeEE")

    @staticmethod
    def _load_json(filename: str):
        with open(filename, encoding="utf8") as f:
            return json.load(f)

    @staticmethod
    def _parse(cmeee_data):
        return [InputExample(sentence_id=str(i), **data) for i, data in enumerate(cmeee_data)]

    def get_data(self, mode: str='select_dev', path=None):
        if path is None:
            return self._parse(self._load_json(join(self.data_root, f"CMeEE_{mode}.json")))
        else:
            return self._parse(self._load_json(path))

class EEDataset(Dataset):
    def __init__(self, cblue_root: str, mode: str, tokenizer):
        self.cblue_root = cblue_root
        self.data_root = join(cblue_root, "CMeEE")

        # This flag is used in CRF
        self.no_decode = mode.lower() == "train"
        self.max_length  = 512

        self.examples = EEDataloader(cblue_root).get_data(mode) # get original data
        print(len(self.examples))
        self.data, self.idx = self._preprocess(self.examples, tokenizer) # preprocess

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def _preprocess(self, examples, tokenizer):
        is_test = examples[0].entities is None
        data = []
        idx = []

        label2id = EE_label2id
        for example in examples:
            text, entity = example.text, example.entities
            idx.append({'text':text, 'entities':entity})
            if is_test:
                _sentence_id, text = example.to_score_task()
                label = repeat(None, len(text))
            else:
                _sentence_id, text, label = example.to_score_task()

            tokens = []
            label_ids = None if is_test else []

            for word, L in zip(text, label):
                token = tokenizer.tokenize(word)
                if not token:
                    token = [tokenizer.unk_token]
                tokens.extend(token)

                if not is_test:
                    label_ids.extend([label2id[L]] + [tokenizer.pad_token_id] * (len(token) - 1))


            tokens = [tokenizer.cls_token] + tokens[: self.max_length - 2] + [tokenizer.sep_token]
            token_ids = tokenizer.convert_tokens_to_ids(tokens)

            if not is_test:
                label_ids = [label2id[NO_ENT]] + label_ids[: self.max_length - 2] + [label2id[NO_ENT]]

                data.append((token_ids, label_ids))
            else:
                data.append((token_ids,))

        return data, idx

class CollateFnForEE:
    def __init__(self, pad_token_id: int, label_pad_token_id: int = EE_label2id[NER_PAD], for_nested_ner: bool = False):
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = label_pad_token_id
        self.for_nested_ner = for_nested_ner

    def __call__(self, inputs) -> dict:
        input_ids = [x[0]  for x in inputs]
        labels    = [x[1]  for x in inputs] if len(inputs[0]) > 1 else None

        if self.for_nested_ner:
            label2 = [x[2]  for x in inputs] if len(inputs[0]) > 1 else None

        max_len = max(map(len, input_ids))
        attention_mask = torch.zeros((len(inputs), max_len), dtype=torch.long)

        for i, _ids in enumerate(input_ids):
            attention_mask[i][:len(_ids)] = 1
            _delta_len = max_len - len(_ids)
            input_ids[i] += [self.pad_token_id] * _delta_len
            if labels is not None:
                labels[i] += [self.label_pad_token_id] * _delta_len

        inputs = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": attention_mask,
            "labels": torch.tensor(labels, dtype=torch.long) if labels is not None else None,
        }
        return inputs

if __name__ == "__main__":
    EEDataset("/home/luhaotian/AI3612/cmeee/data/CBLUEDatasets", 'select_dev')