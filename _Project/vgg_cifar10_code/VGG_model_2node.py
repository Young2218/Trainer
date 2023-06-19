from torch import nn


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, m_name, num_classes):
        super(VGG, self).__init__()
        # self.features = self._make_layers(cfg[vgg_name])


        self.features = self._make_layers(cfg[m_name], batch_norm=True)
        self.fc_layer = nn.Linear(512, 2)
        self.classifier = nn.Linear(2, num_classes)


    def forward(self, x, target=None):

        out = self.features(x)
        # print("out1 :", out.shape)
        out = out.view(out.size(0), -1)
        # print("out2 :", out.shape)

        fc_features = self.fc_layer(out)
        out = self.classifier(fc_features)

        # print("out3 :", out.shape)
        return fc_features, out


    def _make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, x, kernel_size=3, padding=1)

                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(x), nn.ReLU(inplace=True)]

                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)