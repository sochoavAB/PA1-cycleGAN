import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset

print("Python version")
print(sys.version)
print("PyTorch version:", torch.__version__)

# Define the Resnet Block for the generator
class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        return x + self.conv_block(x)

# Define the generator model
class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=9):
        super(Generator, self).__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True)
        ]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(True)
            ]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(int(ngf * mult / 2)),
                nn.ReLU(True)
            ]

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

# Define the discriminator model
class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64):
        super(Discriminator, self).__init__()
        model = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]

        model += [
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, True)
        ]

        model += [
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, True)
        ]

        model += [
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, True)
        ]

        model += [nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

# Define the dataset class for CycleGAN
class CycleGANDataset(Dataset):
    def __init__(self, npz_file, transform=None, fraction=0.8):
        data = np.load(npz_file)
        total_samples = len(data['arr_0'])
        num_samples = int(total_samples * fraction)
        
        self.A = data['arr_0'][:num_samples]
        self.B = data['arr_1'][:num_samples]
        self.transform = transform

    def __len__(self):
        return len(self.A)

    def __getitem__(self, idx):
        A = self.A[idx].astype(np.float32) / 255.0
        B = self.B[idx].astype(np.float32) / 255.0
        
        A = torch.from_numpy(A).permute(2, 0, 1)
        B = torch.from_numpy(B).permute(2, 0, 1)
        
        if self.transform:
            A = self.transform(A)
            B = self.transform(B)
        
        return A, B

# Define the GAN loss
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

# Función para guardar los modelos
def save_models(netG_A2B, netG_B2A, epoch):
    torch.save(netG_A2B.state_dict(), f'netG_A2B_epoch_{epoch+1}.pth')
    torch.save(netG_B2A.state_dict(), f'netG_B2A_epoch_{epoch+1}.pth')

# Función para guardar métricas en un archivo
def save_metrics(losses, valid_losses, epoch):
    with open('training_metrics.txt', 'a') as f:
        f.write(f"\nEpoch {epoch+1}\n")
        f.write("Training Losses:\n")
        f.write(f"Generator Loss: {losses['G'][-1]:.4f}\n")
        f.write(f"Discriminator A Loss: {losses['D_A'][-1]:.4f}\n")
        f.write(f"Discriminator B Loss: {losses['D_B'][-1]:.4f}\n")
        f.write("\nValidation Losses:\n")
        f.write(f"Generator Loss: {valid_losses['G'][-1]:.4f}\n")
        f.write(f"Discriminator A Loss: {valid_losses['D_A'][-1]:.4f}\n")
        f.write(f"Discriminator B Loss: {valid_losses['D_B'][-1]:.4f}\n")
        f.write("-" * 50 + "\n")

# Función para graficar las pérdidas
def plot_losses(losses, valid_losses):
    plt.figure(figsize=(15, 5))
    
    # Training losses
    plt.subplot(1, 2, 1)
    plt.plot(losses['G'], label='Generator')
    plt.plot(losses['D_A'], label='Discriminator A')
    plt.plot(losses['D_B'], label='Discriminator B')
    plt.title('Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Validation losses
    plt.subplot(1, 2, 2)
    plt.plot(valid_losses['G'], label='Generator')
    plt.plot(valid_losses['D_A'], label='Discriminator A')
    plt.plot(valid_losses['D_B'], label='Discriminator B')
    plt.title('Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('losses_plot.png')
    plt.close()

def train(netG_A2B, netG_B2A, netD_A, netD_B, train_loader, valid_loader, num_epochs, device):
    criterionGAN = GANLoss().to(device)
    criterionCycle = nn.L1Loss()
    criterionIdt = nn.L1Loss()

    optimizer_G = optim.Adam(list(netG_A2B.parameters()) + list(netG_B2A.parameters()), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D_A = optim.Adam(netD_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D_B = optim.Adam(netD_B.parameters(), lr=0.0002, betas=(0.5, 0.999))

    scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=30, gamma=0.1)
    scheduler_D_A = optim.lr_scheduler.StepLR(optimizer_D_A, step_size=30, gamma=0.1)
    scheduler_D_B = optim.lr_scheduler.StepLR(optimizer_D_B, step_size=30, gamma=0.1)

    losses = {'G': [], 'D_A': [], 'D_B': []}
    valid_losses = {'G': [], 'D_A': [], 'D_B': []}

    # Clean metrics file at the start of training
    with open('training_metrics.txt', 'w') as f:
        f.write("Training Started\n")
        f.write("=" * 50 + "\n")

    for epoch in range(num_epochs):
        # Training
        netG_A2B.train()
        netG_B2A.train()
        netD_A.train()
        netD_B.train()
        
        epoch_losses = {'G': [], 'D_A': [], 'D_B': []}
        
        for i, (real_A, real_B) in enumerate(train_loader):
            real_A = real_A.to(device)
            real_B = real_B.to(device)

            # Entrenar Generadores
            optimizer_G.zero_grad()

            fake_B = netG_A2B(real_A)
            fake_A = netG_B2A(real_B)

            loss_idt_A = criterionIdt(netG_A2B(real_B), real_B) * 5.0
            loss_idt_B = criterionIdt(netG_B2A(real_A), real_A) * 5.0

            loss_GAN_A2B = criterionGAN(netD_B(fake_B), True)
            loss_GAN_B2A = criterionGAN(netD_A(fake_A), True)

            recovered_A = netG_B2A(fake_B)
            loss_cycle_ABA = criterionCycle(recovered_A, real_A) * 10.0

            recovered_B = netG_A2B(fake_A)
            loss_cycle_BAB = criterionCycle(recovered_B, real_B) * 10.0

            loss_G = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB + loss_idt_A + loss_idt_B
            loss_G.backward()
            optimizer_G.step()

            # Entrenar Discriminador A
            optimizer_D_A.zero_grad()
            loss_D_A = criterionGAN(netD_A(real_A), True) + criterionGAN(netD_A(fake_A.detach()), False)
            loss_D_A.backward()
            optimizer_D_A.step()

            # Entrenar Discriminador B
            optimizer_D_B.zero_grad()
            loss_D_B = criterionGAN(netD_B(real_B), True) + criterionGAN(netD_B(fake_B.detach()), False)
            loss_D_B.backward()
            optimizer_D_B.step()

            epoch_losses['G'].append(loss_G.item())
            epoch_losses['D_A'].append(loss_D_A.item())
            epoch_losses['D_B'].append(loss_D_B.item())

            if i % 125 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], '
                      f'D_A_loss: {loss_D_A.item():.4f}, D_B_loss: {loss_D_B.item():.4f}, '
                      f'G_loss: {loss_G.item():.4f}')

        # Calculate average training losses for the epoch
        for key in losses.keys():
            losses[key].append(np.mean(epoch_losses[key]))

        # Validation
        netG_A2B.eval()
        netG_B2A.eval()
        netD_A.eval()
        netD_B.eval()
        
        epoch_valid_losses = {'G': [], 'D_A': [], 'D_B': []}
        
        with torch.no_grad():
            for real_A, real_B in valid_loader:
                real_A = real_A.to(device)
                real_B = real_B.to(device)

                # Generator losses
                fake_B = netG_A2B(real_A)
                fake_A = netG_B2A(real_B)

                loss_idt_A = criterionIdt(netG_A2B(real_B), real_B) * 5.0
                loss_idt_B = criterionIdt(netG_B2A(real_A), real_A) * 5.0

                loss_GAN_A2B = criterionGAN(netD_B(fake_B), True)
                loss_GAN_B2A = criterionGAN(netD_A(fake_A), True)

                recovered_A = netG_B2A(fake_B)
                loss_cycle_ABA = criterionCycle(recovered_A, real_A) * 10.0

                recovered_B = netG_A2B(fake_A)
                loss_cycle_BAB = criterionCycle(recovered_B, real_B) * 10.0

                loss_G = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB + loss_idt_A + loss_idt_B

                # Discriminator losses
                loss_D_A = criterionGAN(netD_A(real_A), True) + criterionGAN(netD_A(fake_A), False)
                loss_D_B = criterionGAN(netD_B(real_B), True) + criterionGAN(netD_B(fake_B), False)

                epoch_valid_losses['G'].append(loss_G.item())
                epoch_valid_losses['D_A'].append(loss_D_A.item())
                epoch_valid_losses['D_B'].append(loss_D_B.item())

        # Calculate average validation losses for the epoch
        for key in valid_losses.keys():
            valid_losses[key].append(np.mean(epoch_valid_losses[key]))

        print(f'Epoch [{epoch+1}/{num_epochs}] - '
              f'Train G_loss: {losses["G"][-1]:.4f}, '
              f'Train D_A_loss: {losses["D_A"][-1]:.4f}, '
              f'Train D_B_loss: {losses["D_B"][-1]:.4f}, '
              f'Valid G_loss: {valid_losses["G"][-1]:.4f}, '
              f'Valid D_A_loss: {valid_losses["D_A"][-1]:.4f}, '
              f'Valid D_B_loss: {valid_losses["D_B"][-1]:.4f}')

        # Save metrics every epoch
        save_metrics(losses, valid_losses, epoch)

        # Plot losses every 10 epochs
        if (epoch + 1) % 10 == 0:
            plot_losses(losses, valid_losses)

        # Save models every 50 epochs
        if (epoch + 1) % 25 == 0:
            save_models(netG_A2B, netG_B2A, epoch)

        scheduler_G.step()
        scheduler_D_A.step()
        scheduler_D_B.step()

    return losses, valid_losses

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize models
    netG_A2B = Generator(3, 3).to(device)
    netG_B2A = Generator(3, 3).to(device)
    netD_A = Discriminator(3).to(device)
    netD_B = Discriminator(3).to(device)

    # Normalization transform
    transform = transforms.Compose([
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load datasets
    train_dataset = CycleGANDataset('./data/confocal_exper_altogether_trainR_256.npz', transform=transform)
    valid_dataset = CycleGANDataset('./data/confocal_exper_non_sat_filt_validR_256.npz', transform=transform)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False)

    # Train the model
    num_epochs = 125
    losses, valid_losses = train(netG_A2B, netG_B2A, netD_A, netD_B, train_loader, valid_loader, num_epochs, device)
    
    # Plot final losses
    plot_losses(losses, valid_losses)