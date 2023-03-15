from copy import deepcopy
import torch.nn as nn
import torch
from torchvision.models.resnet import ResNet
from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import Bottleneck
from pretrainedmodels.models.torchvision_models import pretrained_settings

from ..base._base import EncoderMixin

class Encoder_channelatt_img(ResNet, EncoderMixin):
    def __init__(self, out_channels, classes_num=12, patch_size=10, depth=5, att_depth=1, **kwargs):
        super().__init__(**kwargs)
        self._depth = depth
        self._attention_on_depth=att_depth

        self._out_channels = out_channels

        self._in_channels = 3

        self.patch_size = patch_size

        self.conv_img=nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(7,7),padding=3),

            nn.Conv2d(64, classes_num, kernel_size=(3,3), padding=1)
        )

        self.conv_feamap=nn.Sequential(
            nn.Conv2d(self._out_channels[self._attention_on_depth], classes_num, kernel_size=(1, 1), stride=1)
        )

        self.unfold = nn.Unfold(kernel_size=(self.patch_size, self.patch_size), stride=(self.patch_size, self.patch_size))

        self.resolution_trans=nn.Sequential(
            nn.Linear(self.patch_size * self.patch_size, 2*self.patch_size * self.patch_size, bias=False),
            nn.Linear(2*self.patch_size * self.patch_size, self.patch_size * self.patch_size, bias=False),
            nn.ReLU()
        )

        del self.fc
        del self.avgpool

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.conv1, self.bn1, self.relu),
            nn.Sequential(self.maxpool, self.layer1),
            self.layer2,
            self.layer3,
            self.layer4,
        ]

    def forward(self, x):
        stages = self.get_stages()
        features = []
        attentions=[]

        x = stages[0](x)
        features.append(x)

        ini_img=self.conv_img(x)

        x = stages[1](x)
        features.append(x)

        if self._attention_on_depth == 1:
            feamap = self.conv_feamap(x) / (2 ** self._attention_on_depth * 2 ** self._attention_on_depth)

            for i in range(feamap.size()[1]):

                unfold_img = self.unfold(ini_img[:, i:i + 1, :, :]).transpose(-1, -2)
                unfold_img = self.resolution_trans(unfold_img)

                unfold_feamap = self.unfold(feamap[:, i:i + 1, :, :])
                unfold_feamap = self.resolution_trans(unfold_feamap.transpose(-1, -2)).transpose(-1, -2)

                att = torch.matmul(unfold_img, unfold_feamap) / (self.patch_size * self.patch_size)

                att=torch.unsqueeze(att,1)

                attentions.append(att)

            attentions = torch.cat((attentions), dim=1)

        x = stages[2](x)
        features.append(x)

        if self._attention_on_depth == 2:
            feamap = self.conv_feamap(x) / (2 ** self._attention_on_depth * 2 ** self._attention_on_depth)

            for i in range(feamap.size()[1]):
                unfold_img = self.unfold(ini_img[:, i:i + 1, :, :]).transpose(-1, -2)
                unfold_img = self.resolution_trans(unfold_img)

                unfold_feamap = self.unfold(feamap[:, i:i + 1, :, :])
                unfold_feamap = self.resolution_trans(unfold_feamap.transpose(-1, -2)).transpose(-1, -2)

                att = torch.matmul(unfold_img, unfold_feamap) / (self.patch_size * self.patch_size)

                att=torch.unsqueeze(att,1)

                attentions.append(att)

            attentions = torch.cat((attentions), dim=1)


        x = stages[3](x)
        features.append(x)

        if self._attention_on_depth == 3:
            feamap = self.conv_feamap(x) / (2 ** self._attention_on_depth * 2 ** self._attention_on_depth)

            for i in range(feamap.size()[1]):
                unfold_img = self.unfold(ini_img[:, i:i + 1, :, :]).transpose(-1, -2)
                unfold_img = self.resolution_trans(unfold_img)

                unfold_feamap = self.unfold(feamap[:, i:i + 1, :, :])
                unfold_feamap = self.resolution_trans(unfold_feamap.transpose(-1, -2)).transpose(-1, -2)

                att = torch.matmul(unfold_img, unfold_feamap) / (self.patch_size * self.patch_size)

                att = torch.unsqueeze(att, 1)

                attentions.append(att)

            attentions = torch.cat(attentions, dim=1)

        x = stages[4](x)
        features.append(x)

        if self._attention_on_depth == 4:
            feamap = self.conv_feamap(x) / (2 ** self._attention_on_depth * 2 ** self._attention_on_depth)

            for i in range(feamap.size()[1]):

                unfold_img = self.unfold(ini_img[:, i:i + 1, :, :]).transpose(-1, -2)
                unfold_img = self.resolution_trans(unfold_img)

                unfold_feamap = self.unfold(feamap[:, i:i + 1, :, :])
                unfold_feamap = self.resolution_trans(unfold_feamap.transpose(-1, -2)).transpose(-1, -2)

                att = torch.matmul(unfold_img, unfold_feamap) / (self.patch_size * self.patch_size)

                att = torch.unsqueeze(att, 1)

                attentions.append(att)

            attentions = torch.cat(attentions, dim=1)

        x = stages[5](x)
        features.append(x)

        if self._attention_on_depth == 5:
            feamap = self.conv_feamap(x) / (2 ** self._attention_on_depth * 2 ** self._attention_on_depth)

            for i in range(feamap.size()[1]):
                unfold_img = self.unfold(ini_img[:, i:i + 1, :, :]).transpose(-1, -2)
                unfold_img = self.resolution_trans(unfold_img)

                unfold_feamap = self.unfold(feamap[:, i:i + 1, :, :])
                unfold_feamap = self.resolution_trans(unfold_feamap.transpose(-1, -2)).transpose(-1, -2)

                att = torch.matmul(unfold_img, unfold_feamap) / (self.patch_size * self.patch_size)

                att = torch.unsqueeze(att, 1)

                attentions.append(att)

            attentions = torch.cat((attentions), dim=1)

        return features, attentions


    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("fc.bias")
        state_dict.pop("fc.weight")
        #state_dict.pop("conv_and_unfold_list")

        super().load_state_dict(state_dict, strict=False, **kwargs)


new_settings = {
    "resnet18": {
        "ssl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnet18-d92f0530.pth",
        "swsl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnet18-118f1556.pth"
    },
    "resnet50": {
        "ssl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnet50-08389792.pth",
        "swsl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnet50-16a12f1b.pth"
    },
    "resnext50_32x4d": {
        "imagenet": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
        "ssl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnext50_32x4-ddb3e555.pth",
        "swsl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnext50_32x4-72679e44.pth",
    },
    "resnext101_32x4d": {
        "ssl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnext101_32x4-dc43570a.pth",
        "swsl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnext101_32x4-3f87e46b.pth"
    },
    "resnext101_32x8d": {
        "imagenet": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
        "instagram": "https://download.pytorch.org/models/ig_resnext101_32x8-c38310e5.pth",
        "ssl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnext101_32x8-2cfe2f8b.pth",
        "swsl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnext101_32x8-b4712904.pth",
    },
    "resnext101_32x16d": {
        "instagram": "https://download.pytorch.org/models/ig_resnext101_32x16-c6f796b0.pth",
        "ssl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnext101_32x16-15fffa57.pth",
        "swsl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnext101_32x16-f3559a9c.pth",
    },
    "resnext101_32x32d": {
        "instagram": "https://download.pytorch.org/models/ig_resnext101_32x32-e4b90b00.pth",
    },
    "resnext101_32x48d": {
        "instagram": "https://download.pytorch.org/models/ig_resnext101_32x48-3e41cc8a.pth",
    }
}

pretrained_settings = deepcopy(pretrained_settings)
for model_name, sources in new_settings.items():
    if model_name not in pretrained_settings:
        pretrained_settings[model_name] = {}

    for source_name, source_url in sources.items():
        pretrained_settings[model_name][source_name] = {
            "url": source_url,
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }


encoders_channelatt_img = {
    "resnet18": {
        "encoder": Encoder_channelatt_img,
        "pretrained_settings": pretrained_settings["resnet18"],
        "params": {
            "out_channels": (3, 64, 64, 128, 256, 512),
            "block": BasicBlock,
            "layers": [2, 2, 2, 2],
        },
    },
    "resnet34": {
        "encoder": Encoder_channelatt_img,
        "pretrained_settings": pretrained_settings["resnet34"],
        "params": {
            "out_channels": (3, 64, 64, 128, 256, 512),
            "block": BasicBlock,
            "layers": [3, 4, 6, 3],
        },
    },
    "resnet50": {
        "encoder": Encoder_channelatt_img,
        "pretrained_settings": pretrained_settings["resnet50"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 6, 3],
        },
    },
    "resnet101": {
        "encoder": Encoder_channelatt_img,
        "pretrained_settings": pretrained_settings["resnet101"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 23, 3],
        },
    },
    "resnet152": {
        "encoder": Encoder_channelatt_img,
        "pretrained_settings": pretrained_settings["resnet152"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 8, 36, 3],
        },
    },
    "resnext50_32x4d": {
        "encoder": Encoder_channelatt_img,
        "pretrained_settings": pretrained_settings["resnext50_32x4d"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 6, 3],
            "groups": 32,
            "width_per_group": 4,
        },
    },
    "resnext101_32x4d": {
        "encoder": Encoder_channelatt_img,
        "pretrained_settings": pretrained_settings["resnext101_32x4d"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 23, 3],
            "groups": 32,
            "width_per_group": 4,
        },
    },
    "resnext101_32x8d": {
        "encoder": Encoder_channelatt_img,
        "pretrained_settings": pretrained_settings["resnext101_32x8d"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 23, 3],
            "groups": 32,
            "width_per_group": 8,
        },
    },
    "resnext101_32x16d": {
        "encoder": Encoder_channelatt_img,
        "pretrained_settings": pretrained_settings["resnext101_32x16d"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 23, 3],
            "groups": 32,
            "width_per_group": 16,
        },
    },
    "resnext101_32x32d": {
        "encoder": Encoder_channelatt_img,
        "pretrained_settings": pretrained_settings["resnext101_32x32d"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 23, 3],
            "groups": 32,
            "width_per_group": 32,
        },
    },
    "resnext101_32x48d": {
        "encoder": Encoder_channelatt_img,
        "pretrained_settings": pretrained_settings["resnext101_32x48d"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 23, 3],
            "groups": 32,
            "width_per_group": 48,
        },
    },
}

