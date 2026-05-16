"""
A bare-bones GPT-2 style transformer.
"""

import math
from typing import Dict

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from jaxtyping import Float, Int
from torch.nn.functional import softmax
from dataclasses import dataclass
from einops import rearrange
from transformers import GPT2LMHeadModel
import huggingface_hub
import numpy as np
from utils import state_dict_converter


# TODO: Add in attention mask to the entire assignment
# TODO: Maybe add KV caching


@dataclass
class ModelConfig:
    d_model: int
    n_heads: int
    n_layers: int
    context_length: int
    vocab_size: int


class CausalAttention(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        # Using attention dim from attention is all you need
        assert config.d_model % config.n_heads == 0
        self.d_attention = int(config.d_model / config.n_heads)

        #self.c_attn = nn.Linear(config.d_model, 3 * config.d_model)

        self.W_k = nn.Linear(config.d_model, self.d_attention * config.n_heads)
        self.W_q = nn.Linear(config.d_model, self.d_attention * config.n_heads)
        self.W_v = nn.Linear(config.d_model, self.d_attention * config.n_heads)

        self.W_o = nn.Linear(self.d_attention * config.n_heads, config.d_model)
        self.head_num = config.n_heads
        self.d_model = config.d_model
        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.context_length, config.context_length)).view(
                1, 1, config.context_length, config.context_length
            ),
            persistent=False
        )

    def forward(
        self, x: Float[Tensor, "batch seq_len d_model"]
    ) -> Float[Tensor, "batch seq_len d_model"]:

        # TODO, complete 
        K = self.W_k(x)
        Q = self.W_q(x)
        V = self.W_v(x)
        K = K.reshape(x.shape[0], x.shape[1], self.head_num, self.d_attention)
        K = K.transpose(1,2)
        Q = Q.reshape(x.shape[0], x.shape[1], self.head_num, self.d_attention)
        Q = Q.transpose(1,2)
        V = V.reshape(x.shape[0], x.shape[1], self.head_num, self.d_attention)
        V = V.transpose(1,2)
        score = torch.matmul(Q, K.transpose(3,2))/torch.sqrt(torch.tensor(self.d_attention))
        real_mask = self.causal_mask[:,:,:x.shape[1],:x.shape[1]]
        real_mask = torch.where(real_mask>0, 0.0, float('-inf'))
        score = score + real_mask 
        score = torch.softmax(score, dim = -1) # batch, h, t, t
        out = torch.matmul(score, V) # batch, h, t, d/h
        out = out.transpose(1,2).reshape(x.shape[0], x.shape[1], -1)
        out = self.W_o(out)
        return out



class GELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))  # fmt: skip

class MLP(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.fc1 = nn.Linear(config.d_model, 4 * config.d_model)
        self.fc2 = nn.Linear(4 * config.d_model, config.d_model)
        self.gelu = GELU()

    def forward(
        self, x: Float[Tensor, "batch seq_len d_model"]
    ) -> Float[Tensor, "batch seq_len d_model"]:

        # TODO, complete
        x = self.gelu(self.fc1(x))
        x = self.fc2(x)
        return x
        

class DecoderBlock(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.mlp = MLP(config)
        self.attention = CausalAttention(config)
        self.pre_layer_norm = nn.LayerNorm(config.d_model)
        self.post_layer_norm = nn.LayerNorm(config.d_model)

    def forward(
        self, x: Float[Tensor, "batch seq_len d_model"]
    ) -> Float[Tensor, "batch seq_len d_model"]:

        # TODO complete
        after_attention = self.attention(self.pre_layer_norm(x)) + x
        after_mlp = self.mlp(self.post_layer_norm(after_attention)) + after_attention
        return after_mlp


class Transformer(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config
        self.embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embeddings = nn.Embedding(config.context_length, config.d_model)
        self.backbone = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layers)])
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):

        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.ones_(module.weight)

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layers)
                )

    def forward(
        self, x: Int[Tensor, "batch_size seq_len"]
    ) -> Float[Tensor, "batch seq_len vocab_size"]:

        # TODO, complete
        embed_tokens = self.embeddings(x)
        # pe của gpt 2 không hoạt động theo kiểu cosine mà có thể học được, với đầu vào không phải là token id cụ thể mà là index của các từ trong câu từ 0 đến n
        idx_matrix = torch.arange(0, x.shape[1]).expand(x.shape[0], x.shape[1])
        pe = self.position_embeddings(idx_matrix)
        full_embed = embed_tokens + pe
        for decoder in self.backbone:
            full_embed = decoder(full_embed)
        final_embed = self.final_layer_norm(full_embed)  # batch, seq len, d model
        output = self.lm_head(final_embed)  # batch, seq len, vocab size
        return output
    @torch.no_grad()
    def generate(
        self,
        x: Int[Tensor, "batch_size seq_len"],
        num_new_tokens: int,
    ) -> Int[Tensor, "batch_size seq_len+num_new_tokens"]:

        # TODO, complete
        for _ in range(num_new_tokens):
            output = self.forward(x)
            output = output[:,-1,:]
            output = output.argmax(dim = 1, keepdim = True)
            x = torch.concat((x, output), dim = 1)
        return x


    def get_loss_on_batch(
        self,
        input_ids: Int[Tensor, "batch_size seq_len"], 
    ) -> Float[Tensor, ""]:
        
        # TODO, complete
        criterion = nn.CrossEntropyLoss()
        target_ids = input_ids[:,1:]       # batch, seq_len
        inputs_ids = input_ids[:,:-1]
        pred = self.forward(inputs_ids)    # batch, seq_len, vocab_size
        pred = pred.reshape(-1, pred.shape[2])
        target_ids = target_ids.reshape(-1)
        loss = criterion(pred, target_ids)    
        return loss


    @classmethod
    def from_pretrained(cls):
        """
        We simply always load up the GPT-2 model
        """

        # Config for GPT-2
        config = ModelConfig(
            d_model=768,
            n_heads=12,
            n_layers=12,
            context_length=1024,
            vocab_size=50257,
        )

        model = cls(config)

        # Load weights from HuggingFace
        model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
        converted_state_dict: Dict[str, Tensor] = state_dict_converter(model_hf.state_dict())

        model.load_state_dict(converted_state_dict)

        return model


if __name__ == "__main__":

    # Uncomment this if you are not logged in
    # huggingface_hub.login()
    
    model = Transformer.from_pretrained()
