import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import SequenceClassifierOutput


class TFEncoderMMP(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model=256,
        nhead=4,
        dim_feedforward=512,
        dropout=0.1,
    ):
        super(TFEncoderMMP, self).__init__()
        self.d_model = d_model
        self.ffn = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2,
        )

    def forward(self, inputs, attention_mask=None):
        inputs = self.ffn(inputs)

        output = self.transformer_encoder(
            inputs,
            src_key_padding_mask=None if torch.all(attention_mask == 1) else attention_mask,
        )

        int_attn_mask = attention_mask.int()
        output[attention_mask != True] = torch.zeros(self.d_model).to(output.device)
        mmp = output.sum(dim=1) / (int_attn_mask.sum(dim=-1).reshape(-1, 1))
        return mmp


class MGTDetectionModel(nn.Module):
    def __init__(self, input_dim=3, d_model=256, last_dim=768, num_labels=2):
        super(MGTDetectionModel, self).__init__()
        self.num_labels = num_labels
        self.model_1 = TFEncoderMMP(input_dim=input_dim, d_model=d_model)
        self.model_2 = TFEncoderMMP(input_dim=input_dim, d_model=d_model)
        self.model_3 = TFEncoderMMP(input_dim=input_dim, d_model=d_model)
        self.last_model = TFEncoderMMP(input_dim=last_dim, d_model=last_dim)
        self.output_layer = nn.Linear(last_dim, 2)

    def forward(
        self,
        model_1_input,
        model_1_attention_mask,
        model_2_input,
        model_2_attention_mask,
        model_3_input,
        model_3_attention_mask,
        labels=None,
    ):
        model_1_output = self.model_1(model_1_input, model_1_attention_mask)
        model_2_output = self.model_2(model_2_input, model_2_attention_mask)
        model_3_output = self.model_3(model_3_input, model_3_attention_mask)
        concat_output = torch.concat([model_1_output, model_2_output, model_3_output], dim=1)
        logits = self.output_layer(concat_output)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )
