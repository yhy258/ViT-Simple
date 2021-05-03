import copy
import torch
import torch.nn as nn
import Backbone


class VisionTransformer(nn.Module):
    def __init__(self, preprocessing, transformer, num_classes, hid_dim):
        super(VisionTransformer, self).__init__()
        self.embedding = preprocessing
        self.transformer = transformer
        self.mlp_head = nn.Linear(hid_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)  # [N+1, BS, D]
        x = x.permute(1, 0, 2)  # [BS, N+1, D]
        x = self.mlp_head(x[:, 0])  # [BS, D]이렇게 들어감 -> [BS, num_classes]
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, layer, num_layers):
        super().__init__()
        self.layers = clone_layers_(layer, num_layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TransformerEncoderLayer(nn.Module):  # [batch_size, patch_num, P*P, D]
    def __init__(self, hid_dim, ff_dim, n_heads):  # 16, 14, etc.
        super().__init__()

        self.norm1 = nn.LayerNorm(hid_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(hid_dim, eps=1e-6)

        self.mhsa = nn.MultiheadAttention(hid_dim, n_heads, dropout=0.1)
        self.dropout1 = nn.Dropout(0.1)

        self.mlp = MLP(hid_dim, ff_dim)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, x):
        # [N+1, BS, D]
        shortcut = x

        q = k = v = self.norm1(x)

        x = self.mhsa(query=q, key=k, value=v)[0] + shortcut

        shortcut = x

        x = self.norm2(x)

        x = self.dropout1(self.mlp(x))

        x = x + shortcut

        return x


class Preprocessing(nn.Module):
    def __init__(self, image_size, hid_dim, patch_size, is_hybrid=False, device=None):
        super().__init__()
        h, w = image_size
        if is_hybrid:
            self.backbone = Backbone.ResNetFeatures()
            h, w = h // 16, w // 16
            in_channel = self.backbone.out_channels
        else:
            in_channel = 3

        self.is_hybrid = is_hybrid
        self.image_embedding = nn.Conv2d(in_channel, hid_dim, kernel_size=patch_size,
                                         stride=patch_size)  # BS, D, H/patch_size, W/patch_size

        n_patches = (h // patch_size) * (w // patch_size)

        self.pos_embedding = nn.Parameter(torch.zeros(1, n_patches + 1, hid_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hid_dim))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        BS = x.shape[0]
        if self.is_hybrid:
            x = self.backbone(x)

        x = self.image_embedding(x)
        x = x.flatten(2)  # [BS, D, N]
        x = x.transpose(-1, -2)  # [BS, N, D]

        cls_token = self.cls_token.expand(BS, -1, -1)  # [BS, 1, D]
        x = torch.cat([x, cls_token], dim=-2)  # [BS, N+1, D]
        x = self.dropout(x + self.pos_embedding)

        return x.permute(1, 0, 2)  # [N+1, BS, D]


class MLP(nn.Module):
    def __init__(self, hid_dim, ff_dim):
        super().__init__()
        self.linear1 = nn.Linear(hid_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, hid_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.normal_(self.linear1.bias, std=1e-6)
        nn.init.normal_(self.linear2.bias, std=1e-6)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)

        return x


def clone_layers_(layer, num_layers):
    return [copy.deepcopy(layer) for _ in range(num_layers)]