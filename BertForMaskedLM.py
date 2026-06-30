import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import MaskedLMOutput
from torch.nn import CrossEntropyLoss


class BertForMaskedLM(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = nn.Linear(config.hidden_size, config.vocab_size)
        self.post_init()

    def get_output_embeddings(self):
        return self.cls

    def set_output_embeddings(self, new_embeddings):
        self.cls = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        return_dict=True,
        **kwargs
    ):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        last_hidden_state = outputs.last_hidden_state  # [bs, seq_len, hidden_size]

        # MLM classifier
        logits = self.cls(last_hidden_state)  # [bs, seq_len, vocab_size]

        mlm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            mlm_loss = loss_fct(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )

        if not return_dict:
            return (mlm_loss, logits, last_hidden_state) if mlm_loss is not None else (logits, last_hidden_state)

        return MaskedLMOutput(
            loss=mlm_loss,
            logits=logits,
            hidden_states=last_hidden_state,  #  last_hidden_state  hidden_states
        )

    def mlm_from_hidden(self, hidden_states, labels=None):

        logits = self.cls(hidden_states)
        mlm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            mlm_loss = loss_fct(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
        return mlm_loss, logits
