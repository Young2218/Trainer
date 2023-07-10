import torch.nn as nn
import torch
import torch.nn.functional as F


class ViT(nn.Module):
    """
    Vit는 3가지로 구성되어야 함
    1. Patch Embedding
    2. Transformer Encoder
    3. MLP layers
    """
    
    def __init__(self, in_channels, patch_size, hidden_dim, num_encoder, num_classes) -> None:
        super().__init__()
        
        self.patch_emb = PatchEmbedding(in_channels, patch_size, hidden_dim)
        self.class_token = nn.Parameter(torch.zeros(1,1,hidden_dim))
        
        
        encoders = []
        for i in range(num_encoder):
            encoders.append(TransformerEncoder(hidden_dim, 4, 8, 0.5))
        self.encoders = nn.Sequential(*encoders)
        
        self.mlp_head = MLPhead(hidden_dim, num_classes)
        
    def forward(self, x: torch.Tensor):
        x = self.patch_emb(x)
        n = x.shape[0]
        
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1) # n, 256, 768 -> n, 257, 768
        
        x = self.encoders(x)
        
        x = self.mlp_head(x)
        
        return x
        


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels:int = 3, patch_size:int=16, hidden_dim:int=768) -> None:
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        
        self.emb_block = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)
        
    
        
    def forward(self, x: torch.Tensor):
        n, c, w, h = x.shape
        pw, ph = w // self.patch_size, h // self.patch_size
        x = self.emb_block(x) # n, 3, 256, 256 -> n, 768, 16, 16 
        
        
        x = x.reshape(n, self.hidden_dim, pw*ph) #  n, 768, 16, 16 -> n, 768, 16*16 
        x = x.permute(0, 2, 1) # n, 768, 16*16 -> n, 16*16, 768
            
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 768, expansion:int = 4, num_head: int = 8, drop_out: float = 0.5) -> None:
        super().__init__()
        
        self.normMSA = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            MSA(hidden_dim, num_head, drop_out)
        )
        
        self.normMLP = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim*expansion),
            nn.GELU(),
            nn.Dropout(drop_out),
            nn.Linear(expansion*hidden_dim, hidden_dim)
        )
        

    def forward(self, x):
        norm_MSA = self.normMSA(x)
        x += norm_MSA
        norm_MLP = self.normMLP(x)
        x += norm_MLP
        
        return x
        
    

class MSA(nn.Module):
    def __init__(self, hidden_dim: int = 768, num_head: int = 8, drop_out: float = 0.5) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_head = num_head
        self.drop_out = drop_out
        
        self.q_emb = nn.Linear(hidden_dim, hidden_dim)
        self.k_emb = nn.Linear(hidden_dim, hidden_dim)
        self.v_emb = nn.Linear(hidden_dim, hidden_dim)
        
        self.att_drop_out = nn.Dropout(p=drop_out)
        self.projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        n, s, _ = x.shape
        d = self.hidden_dim // self.num_head
        
        q = self.q_emb(x)
        q = q.reshape(n,s, self.num_head, d)
        q = q.permute(0, 2, 1, 3)
        
        k = self.k_emb(x)
        k = k.reshape(n,s, self.num_head, d)
        k = k.permute(0, 2, 1, 3)
        
        v = self.v_emb(x)
        v = v.reshape(n,s, self.num_head, d)
        v = v.permute(0, 2, 1, 3)
        
        energy = torch.einsum('bhqd, bhkd -> bhqk', q, k)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.masked_fill(~mask, fill_value)
        
        scaling = self.hidden_dim ** 0.5
        att = F.softmax(energy, dim=1) / scaling
        att = self.att_drop_out(att)
        
        out = torch.einsum('bhal, bhlv -> bhav', att, v) # -> batch, 8, 257, 96
        out = out.permute(0,2,1,3)
        out = out.reshape(n, s, self.hidden_dim)
        
        out = self.projection(out)
        
        return out

class MLPhead(nn.Module):
    def __init__(self, hideen_size, n_classes) -> None:
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(hideen_size),
            nn.Linear(hideen_size, n_classes)
        )
    
    def forward(self, x:torch.Tensor):
        # x = x.permute(0,2,1)
        x = torch.mean(x, dim=1)
        x = self.classifier(x)
        
        return x
        

#====================================================================================================
if __name__ == "__main__":
    net = ViT(in_channels=3, patch_size= 16, hidden_dim=768, num_encoder=6, num_classes=10)
    img = torch.zeros((1,3,256,256))
    
    net(img)