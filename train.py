import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from skimage.metrics import structural_similarity as ssim
import models as m

# utilisation du GPU ou CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Normalise un tenseur dans l'intervalle [0,1]
def normalize(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    return (tensor - min_val) / (max_val - min_val)

# Entrainement du modèle
def entrainer(model, dataset, input_channels, nb_epoch, pth_name, lr=0.001):
    train_dataset = Subset(dataset, range(20000))
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Paramètres de l'entrainement 
    criterion = nn.MSELoss().to(device)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Boucle d'entrainement
    start_time = time.time()
    train_losses = []
    print("Start of the training")
    for epoch in range(nb_epoch):
        # Train
        model.train()
        running_loss = 0.0
        for images, _ in train_loader:
            images = images.to(device)
            output = model(images)
            loss = criterion(output,images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        print(f"Epoch: {epoch+1} | Train loss: {train_loss:.4f}")
        
    
    # Affichage du temps d'entrainement
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Time: {training_time:.2f}s")
    
    # Enregistre les poids
    torch.save(model.state_dict(), pth_name)
    
    # Mesure du SSIM
    img = dataset[0][0].unsqueeze(0)
    out = model(img.to(device))
    out = normalize(out.squeeze(0).squeeze(0))
    
    if out.shape[0] == 3:
        img, out = img.cpu().detach().numpy().squeeze(0), out.cpu().detach().numpy()
        s = 0
        for i in range(3):
            s_chnl = ssim(img[i], out[i], data_range=1.0)
            s += s_chnl
        s = s/3
    else:
        img, out = img.cpu().detach().numpy().squeeze(0).squeeze(0), out.cpu().detach().numpy()
        s = ssim(img, out, data_range=1.0)
    print(f"SSIM score: {s:.4f}")
    
    
def execute(model_name='CAE', depth=3, coeff_NB=4, database="MNIST", nb_epoch=10, pth_name="model.pth"):
    
    # Importe la base de donnees dans un dataloader et construction du modèle
    if database == "MNIST":
        dataset = datasets.MNIST("./MNIST-data", transform=transforms.Compose([
                                                                                transforms.ToTensor(),
                                                                                transforms.Resize((32,32),antialias=True)
                                                                              ]))
        input_channels = 1
        img_size = dataset[0][0].shape[1]
    elif database == "CIFAR10":
        dataset = datasets.CIFAR10("./CIFAR10-data", transform=transforms.Compose([transforms.ToTensor()]))
        input_channels = 3
        img_size = dataset[0][0].shape[1]
    
    if model_name == 'CAE':
        model = m.ConvAutoEncoder(input_channels,img_size,depth,coeff_NB)
    elif model_name == "WDAED":
        model = m.WDAED(input_channels)
    
    # Entraine et test le modèle
    entrainer(model,dataset,input_channels,nb_epoch,pth_name)

    # Vide la mémoire GPU
    if device == torch.device("cuda"):
        torch.cuda.empty_cache()
    
    