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
            embedding_dim,
            num_embeddings,
            num_heads=8,
            out_classes=10,
            num_blocks=6,
            activation_type="gelu"):
        super().__init__()

        # Learnable Embedding and Positional Encoding.
        self.emb_layers = nn.Sequential(
            nn.Embedding(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                padding_idx=padding_idx),
            PositionalEncoding()
        )

        # Decoder Blocks.
        self.decoder_blocks = nn.ModuleList()
        self.decoder_blocks.append(
            TransformerBlock(
                heads=num_heads,
                dim=embedding_dim,
                use_self_attn=True,
                use_cross_attn=False,
                use_masked_attn=True,
                activation_type=activation_type
            )
        )

        self.classifier = nn.Sequential(
            LinearBlock(
                in_dim=embedding_dim,
                out_dim=512,
                use_activation=True),
            LinearBlock(
                in_dim=512,
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
        # Learnable Embedding + Positional Encodings.
        x = self.emb_layers(x)

        # Decoder Blocks.
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x)

        # Classifer Out.
        x = self.classifier(x)
        return x
