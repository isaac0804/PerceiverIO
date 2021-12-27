import torch
import torch.nn as nn
import torch.nn.functional as F

from perceiver_config import PerceiverConfig


class MLP(nn.Module):
    def __init__(self, features, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        self.act_layer = act_layer()
        self.drop = nn.Dropout(drop)
        self.linear_layers = nn.ModuleList(
            [nn.Linear(features[i], features[i + 1]) for i in range(len(features) - 1)]
        )

    def forward(self, x):
        for idx, linear in enumerate(self.linear_layers):
            x = linear(x)
            x = self.act_layer(x)
            if idx != len(self.linear_layers) - 1:
                x = self.drop(x)
        return x


class PerceiverLayer(nn.Module):
    """Building Block of Perceiver Encoder"""

    def __init__(
        self, config, is_cross_attention=False, q_dim=None, kv_dim=None
    ) -> None:
        super().__init__()
        self.config = config
        self.is_cross_attention = is_cross_attention
        self.q_dim = q_dim if q_dim else config.d_latents
        self.kv_dim = kv_dim if kv_dim else self.q_dim
        self.attention = nn.MultiheadAttention(
            self.q_dim,
            config.num_heads,
            kdim=self.kv_dim,
            vdim=self.kv_dim,
            batch_first=True,
        )
        self.layernorm_q = nn.LayerNorm(self.q_dim)
        self.layernorm_premlp = nn.LayerNorm(self.q_dim)
        if is_cross_attention:
            self.layernorm_kv = nn.LayerNorm(self.kv_dim)
        self.mlp = MLP(
            [self.q_dim, self.q_dim * config.widening_factor, self.q_dim],
            drop=config.dropout,
        )

    def forward(self, q, kv=None):
        print(q.shape)
        q = self.layernorm_q(q)
        print(q.shape)
        if not self.is_cross_attention:
            x, attn = self.attention(q, q, q)
        else:
            kv = self.layernorm_kv(kv)
            x, attn = self.attention(q, kv, kv)
        x = self.layernorm_premlp(q + x)
        x = x + self.mlp(x)
        return x, attn


class PerceiverModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        # Learnable Parameters
        self.latents = nn.Parameter(
            torch.randn(1, config.num_latents, config.d_latents)
        )
        self.output_queries = nn.Parameter(
            torch.randn(1, config.num_outputs, config.d_outputs)
        )

        # Input
        if config.ignore_first_cross_attention:
            self.first_cross_attention = PerceiverLayer(
                config,
                is_cross_attention=True,
                q_dim=config.d_latents,
                kv_dim=config.d_inputs,
            )
        self.cross_attention = PerceiverLayer(
            config,
            is_cross_attention=True,
            q_dim=config.d_latents,
            kv_dim=config.d_inputs,
        )

        # Self Attention Tower (shared weights)
        self.self_attention_tower = nn.ModuleList(
            [PerceiverLayer(config) for _ in range(config.num_self_attentions)]
        )

        # Output
        self.output_cross_attention = PerceiverLayer(
            config,
            is_cross_attention=True,
            q_dim=config.d_outputs,
            kv_dim=config.d_latents,
        )

    def forward(self, inputs):

        # First Cross Attention
        if self.config.ignore_first_cross_attention:
            x, _ = self.first_cross_attention(self.latents, inputs)
        else:
            x, _ = self.cross_attention(self.latents, inputs)
        # First Self Attention
        for layer in self.self_attention_tower:
            x, _ = layer(x)

        # Following Cross Attention and Self Attention
        for _ in range(self.config.num_blocks - 1):
            x, _ = self.cross_attention(x, inputs)
            for layer in self.self_attention_tower:
                x, _ = layer(x)

        # Output Cross Attention
        x, _ = self.output_cross_attention(self.output_queries, x)

        return x


if __name__ == "__main__":
    config = PerceiverConfig()
    layer = PerceiverLayer(config)
    model = PerceiverModel(config)
    cross_layer_in = PerceiverLayer(
        config, is_cross_attention=True, q_dim=config.d_latents, kv_dim=config.d_inputs
    )
    cross_layer_out = PerceiverLayer(
        config, is_cross_attention=True, q_dim=config.d_outputs, kv_dim=config.d_latents
    )
    inputs = torch.randn((1, 128, 64))
    latents = torch.randn((1, 32, 32))
    output_query = torch.randn((1, 128, 64))

    outputs, attn = layer(latents)
    print(outputs)
    print(outputs.shape)

    # outputs, attn = cross_layer_in(inputs, latents)
    outputs, attn = cross_layer_in(latents, inputs)
    print(outputs)
    print(outputs.shape)

    outputs, attn = cross_layer_out(output_query, latents)
    print(outputs)
    print(outputs.shape)

    outputs = model(inputs)
    print(outputs)
    print(outputs.shape)
