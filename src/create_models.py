from importsfolder import *
from config_file import config

class TimmEfficientNet_b0(nn.Module):
    def __init__(self):
        super(TimmEfficientNet_b0, self).__init__()
        self.model = timm.create_model("swin_large_patch4_window12_384", pretrained=True, in_chans=3)
        self.model.head = nn.Linear(self.model.head.in_features, 128)
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(128 + 12, 1)
        
    def forward(self, image ,tabular_data_inputs):
        x = self.model(image)
        x = self.dropout(x)
        x = torch.cat([x, tabular_data_inputs], dim=1)
        x = self.out(x)
        
        return x

if __name__ == '__main__':
    model = TimmEfficientNet_b0()
    model = model.to(config.DEVICE)