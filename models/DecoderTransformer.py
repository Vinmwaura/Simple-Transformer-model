import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import (
    LinearBlock,
    TransformerBlock,
    PositionalEncoding)

"""
Transformer Architecture.
Decoder-only Transformer models.
"""
class DecoderTransformer(nn.Module):
    def __init__(
            self,
            padding_idx,
            num_embeddings,
            hidden_dim,
            embedding_dim,
            num_heads=8,
            num_blocks=6,
            out_classes=10,
            activation_type="gelu"):
        super().__init__()

        # Learnable Embedding and Positional Encoding.
        self.emb_layers = nn.Sequential(
            nn.Embedding(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                padding_idx=padding_idx),
            PositionalEncoding())

        # Decoder Blocks.
        self.decoder_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.decoder_blocks.append(
                TransformerBlock(
                    heads=num_heads,
                    hidden_dim=hidden_dim,
                    embedding_dim=embedding_dim,
                    use_self_attn=True,
                    use_cross_attn=False,
                    use_masked_attn=True,
                    activation_type=activation_type
                )
            )

        self.classifier = nn.Sequential(
            LinearBlock(
                in_dim=embedding_dim,
                out_dim=hidden_dim,
                use_activation=True),
            LinearBlock(
                in_dim=hidden_dim,
                out_dim=out_classes,
                use_activation=False))

    def custom_load_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                print(f"No Layer found: {name}, skipping")
                continue
            # Skip loading mismatched weights, in cases of weight changes.
            if (own_state[name].shape != param.data.shape):
                print(f"Skipped: {name}")
                continue
            if isinstance(param, torch.nn.parameter.Parameter):
                # Backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)

    def forward(self, x):
        # Decoder Blocks.
        x_dec = self.emb_layers(x)
        for decoder_block in self.decoder_blocks:
            x_dec = decoder_block(x_dec)

        # Classifer Out.
        x_class = self.classifier(x_dec)
        return x_class
