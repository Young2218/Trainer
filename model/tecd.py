import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchsummary import summary
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

from model.blocks.attentionBlock import AttentionBlock
from model.blocks.doubleConvBlock import DoubleConvBlock
from model.blocks.upConvBlock import UpConvBlock

class TECD(nn.Module):
    def __init__(self) -> None:
        super(TECD, self).__init__()

        segformerModel = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b4-finetuned-ade-512-512")
        self.encoder = segformerModel.segformer 
            
        self.Up5 = UpConvBlock(ch_in=512,ch_out=320)
        self.Att5 = AttentionBlock(F_g=320,F_l=320,F_int=160)
        self.Up_conv5 = DoubleConvBlock(ch_in=640, ch_out=320)

        self.Up4 = UpConvBlock(ch_in=320,ch_out=128)
        self.Att4 = AttentionBlock(F_g=128,F_l=128,F_int=64)
        self.Up_conv4 = DoubleConvBlock(ch_in=256, ch_out=128)
        
        self.Up3 = UpConvBlock(ch_in=128,ch_out=64)
        self.Att3 = AttentionBlock(F_g=64,F_l=64,F_int=32)
        self.Up_conv3 = DoubleConvBlock(ch_in=128, ch_out=64)
        
        self.Up2 = UpConvBlock(ch_in=64,ch_out=32)
        self.Up_conv2 = DoubleConvBlock(ch_in=32, ch_out=32)
        
        self.Up1 = UpConvBlock(ch_in=32,ch_out=16)
        self.Up_conv1 = DoubleConvBlock(ch_in=16, ch_out=16)

        self.Conv_1x1 = nn.Conv2d(16,1,kernel_size=1,stride=1,padding=0)

        
        
    def forward(self, images):
        outputs = self.encoder(
            images,
            output_attentions=None,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=None,
        )
        outputs = outputs['hidden_states']
        # for i in outputs:
        #     print(i.shape)
        
        # output[3] 512,16,16
        d5 = self.Up5(outputs[3]) # 320, 32, 32
        x4 = self.Att5(g=d5,x=outputs[2]) # 320, 32, 32
        d5 = torch.cat((x4,d5),dim=1)  # 640, 32, 32
        d5 = self.Up_conv5(d5) 
        
        # output[2] 320,16,16
        d4 = self.Up4(d5) # output[3] 160, 32,32
        x3 = self.Att4(g=d4,x=outputs[1])
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        # output[1] 128,16,16
        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=outputs[0])
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3) # 64, 128, 128

        
        d2 = self.Up2(d3)
        d2 = self.Up_conv2(d2)
        
        d1 = self.Up1(d2)
        d1 = self.Up_conv1(d1)

        outputs = self.Conv_1x1(d1)

        return outputs
    
if __name__ == "__main__":
    model = TECD().to("cuda")
    sample = torch.rand((1,3,512,512)).to('cuda')
    print(model(sample).shape)
    