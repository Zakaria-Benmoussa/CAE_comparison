import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse

# utilisation du GPU ou CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class autoencodeur simple
class AutoEncoder(nn.Module):
    def __init__(self,input_channels,img_size):
        # N, 784 
        super().__init__()
        
        input_dim = input_channels*img_size**2
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim/4),  
            nn.ReLU(),
            nn.Linear(input_dim/4, input_dim/16),     
            nn.ReLU(),
            nn.Linear(input_dim/16, input_dim/64),      
            nn.ReLU(),
            nn.Linear(input_dim/64, input_dim/128)        
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(input_dim/128, input_dim/64),       
            nn.ReLU(),
            nn.Linear(input_dim/64, input_dim/16),
            nn.ReLU(),
            nn.Linear(input_dim/16, input_dim/4),
            nn.ReLU(),
            nn.Linear(input_dim/4, input_dim),  
            nn.Sigmoid()
        )
    
    def forward(self, x):
        #print(self.input_channels, self.img_size, self.input_channels*self.img_size**2)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

# class autoencodeur convolutif
class ConvAutoEncoder(nn.Module):
    def __init__(self, input_channels, img_size, depth, coeff_BN):
        super().__init__()
        
        self.in_chnl = input_channels        # Nombre de chaines dans l'image
        self.img_size = img_size                    # Taille de l'image
        self.depth = depth                          # Profondeur du réseau
        self.coeff_BN = coeff_BN                    # Facteur de réduction de l'espace latent
        
        self.last_conv_out = 32 * (2 ** (self.depth-1) )
        fc_size = self.last_conv_out * (self.img_size // (2 ** self.depth)) ** 2
        
        # Encodeur
        self.conv = nn.ModuleList()
        for i in range(self.depth):
            if i == 0:
                self.conv.append(nn.Conv2d(self.in_chnl, 32, 3, padding=1))
            else:
                self.conv.append(nn.Conv2d(self.img_size*2**(i-1), 32*2**i, 3, padding=1))
        
        # Decodeur
        self.tconv = nn.ModuleList()
        for i in range(self.depth-1, 0, -1):
            self.tconv.append(nn.ConvTranspose2d(32*2**i, 32*2**(i-1), 3, padding=1))
        self.tconv.append(nn.ConvTranspose2d(32, self.in_chnl, 3, padding=1))
        
        # Fully conected
        self.fc1 = nn.Linear(fc_size, fc_size//self.coeff_BN)
        self.fc2 = nn.Linear(fc_size//self.coeff_BN, fc_size)
        
        # Scaling 
        self.pool = nn.MaxPool2d(2, 2)
        self.unpool = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.min, self.max = 0,0
        
        
    def forward(self, x):
        # Encodeur
        for i in range(self.depth):
            x = F.relu(self.conv[i](x))
            x = self.pool(x)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        
        x = self.fc2(x)
        x = x.view(-1, self.last_conv_out, self.img_size // (2 ** self.depth), self.img_size // (2 ** self.depth))
        
        # Decodeur
        for i in range(self.depth):
            x = self.unpool(x)
            x = F.relu(self.tconv[i](x))
            
        return x

# Modèle ondelettes
class WDAED(nn.Module):
    def __init__(self,input_channels):
        super().__init__()
        
        self.input_channels = input_channels
        
        # Encodeur
        self.conv = nn.Conv2d(input_channels, input_channels, 3, padding=1)
        self.conv_strd = nn.Conv2d(input_channels, input_channels, 3, stride=2, padding=1)
        
        self.encodeur = nn.Sequential(
            self.conv,
            nn.ReLU(),
            self.conv_strd,
            nn.ReLU(),
            self.conv,
            nn.ReLU()
            )
        
        # Decodeur
        self.decodeur = nn.ModuleList()
        for i in range(9):
            self.decodeur.append(self.conv)
            self.decodeur.append(nn.ReLU())
        self.decodeur.append(self.conv)
        
        # Interpolation
        class Interpolate(nn.Module):
            def __init__(self, size, mode):
                super(Interpolate, self).__init__()
                self.interp = F.interpolate
                self.size = size
                self.mode = mode
                
            def forward(self, x):
                x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
                return x
        self.bi_interp = Interpolate(16, 'bilinear')
        
        # Decomposition/Recomposition par ondelettes
        self.xfm = DWTForward(J=1, wave='haar', mode='zero').to(device)
        self.ifm = DWTInverse(mode='zero', wave='haar').to(device)
        
        # Couches pour chaque décomposition
        self.CAE = nn.Sequential(
                self.encodeur,
                self.bi_interp,
                *self.decodeur
            )
    
    def forward(self, x):
        cA, cD = self.xfm(x)
        
        cH, cV, cD = torch.unbind(cD[0],2)
        
        cA = self.CAE(cA)
        cH = self.CAE(cH)
        cV = self.CAE(cV)
        cD = self.CAE(cD)
        
        cH = cH.unsqueeze(0) 
        cV = cV.unsqueeze(0) 
        cD = cD.unsqueeze(0) 
        
        cD = list(torch.stack((cH, cV, cD),3))
        
        x = self.ifm((cA, cD))
        
        return x 

"""
tensor = torch.rand(1,1,32,32).to(device)
model = WDAED(1).to(device)

out = model(tensor)
#print(out.shape)

"""
