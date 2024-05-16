import torch
import math
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=2e-2, img_size=64, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule(schedule_type='linear').to(self.device)
        self.alpha = 1 - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0).to(self.device)

    def prepare_noise_schedule(self, schedule_type='linear'):
        if schedule_type == 'linear':
            self.beta = torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
        elif schedule_type == 'cosine':
            self.beta = torch.cos(torch.linspace(0, 1, self.noise_steps) * math.pi / 2) * (self.beta_end - self.beta_start) + self.beta_start
        else:
            raise ValueError('Unknown schedule type')
        
        return self.beta.to(self.device)

    def noise_images(self, x, t):
        sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod[t])[:, None, None, None]
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.alpha_cumprod[t])[:, None, None, None]
        noise = torch.randn_like(x, device=self.device)

        return sqrt_alpha_cumprod * x + sqrt_one_minus_alpha_cumprod * noise, noise
    
    def sample_timesteps(self, n):
        return torch.randint(low=0, high=self.noise_steps, size=(n,), device=self.device)
    

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, residual=False):
        super(ConvBlock, self).__init__()

        self.residual = residual
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, out_channels)
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.conv(x))
        else:
            return self.conv(x)

        
class DownConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, timestep_dim=32):
        super(DownConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

        self.embedding = nn.Sequential(
            nn.SiLU(),
            nn.Linear(timestep_dim, out_channels)
        )

    def forward(self, x, t):
        x = self.conv(x)
        t = self.embedding(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])

        return x + t
    

class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, timestep_dim=32):
        super(UpConvBlock, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            ConvBlock(in_channels, in_channels, residual=True),
            ConvBlock(in_channels, out_channels)
        )

        self.embedding = nn.Sequential(
            nn.SiLU(),
            nn.Linear(timestep_dim, out_channels)
        )

    def forward(self, x, t, skip_x):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        t = self.embedding(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])

        return x + t
    

class SelfAttentionBlock(nn.Module):
    def __init__(self, in_channels, size):
        super(SelfAttentionBlock, self).__init__()

        self.channels = in_channels
        self.size = size

        self.multihead_attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm([in_channels])
        self.feedforward = nn.Sequential(
            nn.LayerNorm([in_channels]),
            nn.Linear(in_channels, in_channels),
            nn.GELU(),
            nn.Linear(in_channels, in_channels)
        )
    
    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        out_x = self.norm(x)
        out_x, _ = self.multihead_attn(out_x, out_x, out_x)
        out_x = out_x + x
        out_x = self.feedforward(out_x) + out_x
        out_x = out_x.swapaxes(1, 2).view(-1, self.channels, self.size, self.size)

        return out_x
        
    

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, timestep_dim=32, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(UNet, self).__init__()

        self.device = device
        self.timestep_dim = timestep_dim

        self.inc = ConvBlock(in_channels, 64//4).to(device)
        self.down1 = DownConvBlock(64//4, 128//4, timestep_dim=self.timestep_dim).to(device)
        self.attention1 = SelfAttentionBlock(128//4, 32).to(device)
        self.down2 = DownConvBlock(128//4, 256//4, timestep_dim=self.timestep_dim).to(device)
        self.attention2 = SelfAttentionBlock(256//4, 16).to(device)
        self.down3 = DownConvBlock(256//4, 256//4, timestep_dim=self.timestep_dim).to(device)
        self.attention3 = SelfAttentionBlock(256//4, 8).to(device)

        self.bottleneck1 = ConvBlock(256//4, 512//4).to(device)
        self.bottleneck2 = ConvBlock(512//4, 512//4).to(device)
        self.bottleneck3 = ConvBlock(512//4, 256//4).to(device)

        self.up1 = UpConvBlock(512//4, 128//4, timestep_dim=self.timestep_dim).to(device)
        self.attention4 = SelfAttentionBlock(128//4, 16).to(device)
        self.up2 = UpConvBlock(256//4, 64//4, timestep_dim=self.timestep_dim).to(device)
        self.attention5 = SelfAttentionBlock(64//4, 32).to(device)
        self.up3 = UpConvBlock(128//4, 64//4, timestep_dim=self.timestep_dim).to(device)
        self.attention6 = SelfAttentionBlock(64//4, 64).to(device)

        self.outc = nn.Conv2d(64//4, out_channels, kernel_size=1).to(device)

    def pos_encoding(self, t, channels):
        inv_freq = 1. / (10000 ** (torch.arange(0, channels, 2).float().to(self.device) / channels))
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq).to(self.device)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq).to(self.device)
        pos_enc = torch.cat((pos_enc_a, pos_enc_b), dim=-1)

        return pos_enc.to(self.device)

    def forward(self, x, t):
        t = t.unsqueeze(1).type(torch.float)
        t = self.pos_encoding(t, self.timestep_dim).to(self.device)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.attention1(x2)
        x3 = self.down2(x2, t)
        x3 = self.attention2(x3)
        x4 = self.down3(x3, t)
        x4 = self.attention3(x4)

        x4 = self.bottleneck1(x4)
        x4 = self.bottleneck2(x4)
        x4 = self.bottleneck3(x4)

        x = self.up1(x4, t, x3)
        x = self.attention4(x)
        x = self.up2(x, t, x2)
        x = self.attention5(x)
        x = self.up3(x, t, x1)
        x = self.attention6(x)

        output = self.outc(x)

        return output
        

class Trainer:
    def __init__(self, model, diffusion, optimizer, criterion, trainloader, save_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.diffusion = diffusion
        self.optimizer = optimizer
        self.criterion = criterion
        self.trainloader = trainloader
        self.save_path = save_path
        self.device = device

    def train_model(self, epochs):
        for epoch in range(epochs):
            self.model.train()
            progress_bar = tqdm(self.trainloader, desc=f'Epoch {epoch + 1}/{epochs}')
            for i, (images, _) in enumerate(progress_bar):
                images = images.to(self.device)
                t = self.diffusion.sample_timesteps(images.size(0)).to(self.device)
                x_t, noise = self.diffusion.noise_images(images, t)
                predicted_noise = self.model(x_t, t)
                loss = self.criterion(predicted_noise, noise)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Update the progress bar with the current loss
                progress_bar.set_description(f"Epoch {epoch + 1}/{epochs} - Loss: {loss.item():.4f}")
            
            generated_images = self.visualize_performance(n=images.size(0))

            n = len(generated_images)
            fig, axs = plt.subplots(1, n, figsize=(n * 3, 3))
            for i, ax in enumerate(axs.flat):
                img = generated_images[i].cpu().permute(1, 2, 0)
                ax.imshow(img)
                ax.axis('off')
            plt.tight_layout()
            plt.show()
            
        print('Training complete')
        torch.save(self.model.state_dict(), self.save_path)

    def visualize_performance(self, n=4):
        self.model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.diffusion.img_size, self.diffusion.img_size)).to(self.device)
            for i in tqdm(reversed(range(self.diffusion.noise_steps)), desc='Generating images'):
                t = torch.full((n,), i, dtype=torch.long).to(self.device)
                predicted_noise = self.diffusion.noise_images(x, t)
                alpha = self.diffusion.alpha[t][:, None, None, None]
                alpha_cumprod = self.diffusion.alpha_cumprod[t][:, None, None, None]
                beta = self.diffusion.beta[t][:, None, None, None]

                noise = torch.randn_like(x, device=self.device) if i > 0 else torch.zeros_like(x, device=self.device)

                x = (1 / torch.sqrt(alpha)) * (x - predicted_noise * ((1 - alpha) / torch.sqrt(1 - alpha_cumprod))) + torch.sqrt(beta) * noise

            x = (x.clamp(-1, 1) + 1) / 2
            x = (x * 255).type(torch.uint8)
        
        self.model.train()