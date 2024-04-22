from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import torch
import transformers
from torch.utils.data import Dataset
from transformers import Seq2SeqTrainer
import torch.nn.functional as F

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

class ScoreDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data):
        super(ScoreDataset, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return dict(input_ids=self.data[i])

def _single_tokenize(text, tokenizer):
    toked = tokenizer(
            text,
            return_tensors="pt"
        )

    return toked['input_ids'][0]

@dataclass
@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    beta: float = field(default=0.15)
    length_penalty: float = field(default=1)
    gap: float = field(default=0)
    seed: int = field(
        default=1
    )

    temperature: float = field(
        default=1
    )

@dataclass
class DataCollatorForSupervisedDatasetMix(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        idxs = []
        all_scores = []
        input_ids = []
        labels = []
        for idx, ins in enumerate(instances):
            ins = ins['input_ids']
            query = ins['query']
            response = ins['response']
            score = ins['score']
            all_scores.append(score)
            idxs.append(idx)

            query_input_ids = _single_tokenize(query, self.tokenizer)
            query_target = torch.LongTensor([IGNORE_INDEX] * query_input_ids.shape[0])

            res_input_ids = _single_tokenize(response + self.tokenizer.eos_token, self.tokenizer)  # eos here
            input_ids.append(torch.cat((query_input_ids, res_input_ids), dim=0))
            labels.append(torch.cat((query_target, res_input_ids), dim=0))

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        return dict(
            input_ids=input_ids,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            labels=labels,
            idxs=torch.LongTensor(idxs),
            scores=torch.FloatTensor(all_scores),
        )

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        idxs = []
        all_scores = []
        input_ids = []
        labels = []
        for idx, ins in enumerate(instances):
            ins = ins['input_ids']
            query = ins['query']
            responses = ins['responses']
            scores = ins['scores']
            all_scores.append(scores)
            idxs.append([idx] * len(scores))

            query_input_ids = _single_tokenize(query, self.tokenizer)
            query_target = torch.LongTensor([IGNORE_INDEX] * query_input_ids.shape[0])
            for res in responses:
                res_input_ids = _single_tokenize(res + self.tokenizer.eos_token, self.tokenizer)  # eos here
                input_ids.append(torch.cat((query_input_ids, res_input_ids), dim=0))
                labels.append(torch.cat((query_target, res_input_ids), dim=0))

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        return dict(
            input_ids=input_ids,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            labels=labels,
            idxs=torch.LongTensor(idxs),
            scores=torch.FloatTensor(all_scores),
        )

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, train_data, test_data, mix=False) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = ScoreDataset(data=train_data)
    test_dataset = ScoreDataset(data=test_data)
    if mix:
        data_collator = DataCollatorForSupervisedDatasetMix(tokenizer=tokenizer)
    else:
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    return dict(train_dataset=train_dataset, eval_dataset=test_dataset, data_collator=data_collator)


local_rank = None
class RankTrainer(Seq2SeqTrainer):
    def gather_logits_labels(self, logits, labels):
        mask = (labels != -100).float()
        new_logits = logits.clone()  # B x S x hidsize
        new_labels = labels.clone()
        new_labels[labels == -100] = 0  # B x S
        output = torch.gather(new_logits, dim=-1, index=new_labels.unsqueeze(-1)).squeeze(-1)
        output = output * mask  # B * L
        return output

    def get_score(self, logit_label, labels):
        mask = (labels != -100).float()
        length = mask.sum(-1)
        scores = logit_label.sum(-1) / (length ** self.args.length_penalty)
        return scores

    def aft_rank_loss(self, scores, idxs, rw_scores):
        tmp = 0
        sorted_indexs = rw_scores.argsort(dim=-1, descending=True)
        sorted_rw_scores = rw_scores[sorted_indexs]
        sorted_scores = scores[sorted_indexs]

        positive_lower_boundary = 10000
        for i in range(0, len(sorted_rw_scores) - 1):
            pos_scores = sorted_scores[i]
            pos_rw_score = sorted_rw_scores[i].item()
            neg_start = None

            positive_lower_boundary = min(pos_scores.item(), positive_lower_boundary)
            for kk in range(i + 1, len(sorted_rw_scores)):
                if sorted_rw_scores[kk].item() < pos_rw_score:
                    neg_start = kk
                    break
            if neg_start is None:
                continue

            for neg_score in sorted_scores[neg_start:]:
                tmp += torch.exp((neg_score - pos_scores) / self.args.temperature)
                tmp += torch.exp(2 * positive_lower_boundary - 2 * self.args.beta - pos_scores.item() - neg_score)

        if tmp != 0:
            loss = torch.log(1 + tmp)
        else:
            loss = 0
        return loss

    def compute_loss(self, model, inputs, return_outputs=False):
        logits = model(input_ids=inputs.get('input_ids'), attention_mask=inputs.get('attention_mask'))[0]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs.get("labels")[..., 1:].contiguous()
        logits = F.log_softmax(shift_logits, dim=-1)
        logit_label = self.gather_logits_labels(logits, shift_labels)
        scores = self.get_score(logit_label, shift_labels)

        alignment_loss = self.aft_rank_loss(scores, inputs.get("idxs"), inputs.get("scores").squeeze())
        loss = alignment_loss

        return (loss, scores) if return_outputs else loss