import torch
from torchvision.models.vision_transformer import VisionTransformer
import torch.nn as nn


class LambdaLayer(nn.Module):
    def __init__(
        self,
    ):
        super(LambdaLayer, self).__init__()

    def forward(self, x):
        return x[:, 0]


class ViT_B_16(VisionTransformer):
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        num_classes=1000,
    ):
        super(ViT_B_16, self).__init__(
            image_size=image_size,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            num_classes=num_classes,
        )
        self.feature_size = hidden_dim
        self.no_flatten = True

    def get_layer_list(self):
        layers = [
            lambda x: self.get_backbone(x),
            lambda x: x + self.encoder.pos_embedding,
            lambda x: self.encoder.dropout(x),
        ]

        for layer in self.encoder.layers:
            layers.append(layer)
        layers.append(self.encoder.ln)
        layers.append(LambdaLayer())
        layers.append(self.heads)
        return layers

    def get_backbone(self, x):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        return x

    def forward(self, x, return_feature=False, return_feature_list=False):
        x = self.get_backbone(x)
        feat_list = []
        if return_feature_list:
            for layer in self.get_layer_list()[1:-1]:
                x = layer(x)
                feat_list.append(x)
            feat_list.append(x)
        else:
            x = self.encoder(x)
            x = x[:, 0]
        # Apply heads to the class token
        logits = self.heads(x)
        # Return based on flags
        if return_feature:
            return logits, x
        elif return_feature_list:
            return logits, feat_list
        else:
            return logits

    def forward_threshold(self, x, threshold):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        feature = x.clip(max=threshold)
        logits_cls = self.heads(feature)

        return logits_cls

    def get_fc(self):
        fc = self.heads[0]
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()

    def get_fc_layer(self):
        return self.heads[0]
