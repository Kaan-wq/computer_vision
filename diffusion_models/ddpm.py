import torch
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class Diffusion:
    def __init__(self, noise_steps=1000, 
                 img_size=64, 
                 schedule_type='linear', 
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.noise_steps = noise_steps
        self.img_size = img_size
        self.device = device
        self.schedule_type = schedule_type

        self.beta, self.alpha, self.alpha_cumprod = self.prepare_noise_schedule(schedule_type=schedule_type)
        self.alpha = 1 - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0).to(self.device)

    def prepare_noise_schedule(self, schedule_type='linear'):
        if schedule_type == 'linear':
            scale = 1000 / self.noise_steps
            beta_start = scale * 1e-4
            beta_end = scale * 2e-2

            self.beta = torch.linspace(beta_start, beta_end, self.noise_steps)
            self.alpha = (1 - self.beta)
            self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)

        elif schedule_type == 'cosine':
            def f(t):
                return torch.cos((t / self.noise_steps + 0.008) / (1 + 0.008) * 0.5 * torch.pi) ** 2
            
            x = torch.linspace(0, self.noise_steps, self.noise_steps + 1)

            self.alpha_cumprod = f(x) / f(torch.tensor([0]))
            self.beta = 1 - self.alpha_cumprod[1:] / self.alpha_cumprod[:-1]
            self.beta = torch.clip(self.beta, 0.0001, 0.999)
            self.alpha = 1 - self.beta
        else:
            raise NotImplementedError(f"unknown beta schedule: {schedule_type}")
        
        return self.beta.to(self.device), self.alpha.to(self.device), self.alpha_cumprod.to(self.device)
        

    def noise_images(self, x, t):
        sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod[t])[:, None, None, None]
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.alpha_cumprod[t])[:, None, None, None]
        noise = torch.randn_like(x, device=self.device)

        return sqrt_alpha_cumprod * x + sqrt_one_minus_alpha_cumprod * noise, noise
    
    def sample_timesteps(self, n):
        return torch.randint(low=0, high=self.noise_steps, size=(n,), device=self.device)
    
    def generate_images(self, model, n=4):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(self.noise_steps)), desc='Generating images'):
                t = torch.full((n,), i, dtype=torch.long).to(self.device)
                predicted_noise, _ = self.noise_images(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_cumprod = self.alpha_cumprod[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]

                noise = torch.randn_like(x, device=self.device) if i > 0 else torch.zeros_like(x, device=self.device)

                x = (1 / torch.sqrt(alpha)) * (x - predicted_noise * ((1 - alpha) / torch.sqrt(1 - alpha_cumprod))) + torch.sqrt(beta) * noise

        model.train()
        return x
    

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

        self.inc = ConvBlock(in_channels, 64).to(device)
        self.down1 = DownConvBlock(64, 128, timestep_dim=self.timestep_dim).to(device)
        self.attention1 = SelfAttentionBlock(128, 32).to(device)
        self.down2 = DownConvBlock(128, 256, timestep_dim=self.timestep_dim).to(device)
        self.attention2 = SelfAttentionBlock(256, 16).to(device)
        self.down3 = DownConvBlock(256, 256, timestep_dim=self.timestep_dim).to(device)
        self.attention3 = SelfAttentionBlock(256, 8).to(device)

        self.bottleneck1 = ConvBlock(256, 512).to(device)
        self.bottleneck2 = ConvBlock(512, 512).to(device)
        self.bottleneck3 = ConvBlock(512, 256).to(device)

        self.up1 = UpConvBlock(512, 128, timestep_dim=self.timestep_dim).to(device)
        self.attention4 = SelfAttentionBlock(128, 16).to(device)
        self.up2 = UpConvBlock(256, 64, timestep_dim=self.timestep_dim).to(device)
        self.attention5 = SelfAttentionBlock(64, 32).to(device)
        self.up3 = UpConvBlock(128, 64, timestep_dim=self.timestep_dim).to(device)
        self.attention6 = SelfAttentionBlock(64, 64).to(device)

        self.outc = nn.Conv2d(64, out_channels, kernel_size=1).to(device) 

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

        x = self.outc(x)

        return x
    

class EMA:
    def __init__(self, model, decay=0.99):
        self.model = model
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    def update(self):
        with torch.no_grad():
            for name, param in self.model.state_dict().items():
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
        

class Trainer:
    def __init__(self, model, diffusion, optimizer, criterion, trainloader, testloader, save_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.diffusion = diffusion
        self.optimizer = optimizer
        self.criterion = criterion
        self.trainloader = trainloader
        self.testloader = testloader
        self.save_path = save_path
        self.device = device
        self.ema = EMA(self.model)

    def load_ema_weights(self):
        ema_weights = torch.load(self.save_path.replace('.pth', '_ema.pth'))
        self.ema.shadow.load_state_dict(ema_weights)    

    def train_model(self, epochs):
        for epoch in range(epochs):
            self.model.train()
            progress_bar = tqdm(self.trainloader, desc=f'Epoch {epoch + 1}/{epochs}')
            for i, (images, _) in enumerate(progress_bar):
                images = images.to(self.device)
                t = self.diffusion.sample_timesteps(images.size(0)).to(self.device)
                x_t, noise = self.diffusion.noise_images(images, t)
                prediction = self.model(x_t, t)

                objective = (1 / self.diffusion.alpha[t]) * (x_t - ((self.diffusion.beta[t] / torch.sqrt(1 - self.diffusion.alpha_cumprod[t])) * noise))
                loss = self.criterion(prediction, objective)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.ema.update()

                # Update the progress bar with the current loss
                progress_bar.set_description(f"Epoch {epoch + 1}/{epochs} - Loss: {loss.item():.4f}")

            if (epoch + 1) % 10 == 0:

                generated_images = self.diffusion.generate_images(self.model, n=images.size(0))
                generated_images = np.clip(generated_images.cpu(), 0, 1)

                fig, axs = plt.subplots(1, 4, figsize=(4 * 3, 3))
                for i, ax in enumerate(axs.flat):
                    img = generated_images[i].cpu().permute(1, 2, 0)
                    ax.imshow(img)
                    ax.axis('off')
                plt.tight_layout()
                plt.show()
            
        print('Training complete')
        torch.save(self.model.state_dict(), self.save_path)
        torch.save(self.ema.shadow, self.save_path.replace('.pth', '_ema.pth'))
        print('Model saved')

    def eval_model(self):
        self.model.eval()
        losses = [[] for _ in range(self.diffusion.noise_steps)]

        with torch.no_grad():
            progress_bar = tqdm(self.testloader, desc='Evaluating model')
            for i, (images, _) in enumerate(progress_bar):
                images = images.to(self.device)
                t = self.diffusion.sample_timesteps(images.size(0)).to(self.device)
                x_t, noise = self.diffusion.noise_images(images, t)
                predicted_noise = self.model(x_t, t)

                for n, pred, time in zip(noise, predicted_noise, t):
                    loss = self.criterion(pred, n)
                    losses[time].append(loss.item())

            average_losses = [np.mean(lst) if lst else None for lst in losses]

            print('Evaluation complete')
            self.model.train()
        
        return average_losses
    
    def eval_ema_model(self):
        self.ema.model.eval()
        losses = [[] for _ in range(self.diffusion.noise_steps)]

        with torch.no_grad():
            progress_bar = tqdm(self.testloader, desc='Evaluating EMA model')
            for i, (images, _) in enumerate(progress_bar):
                images = images.to(self.device)
                t = self.diffusion.sample_timesteps(images.size(0)).to(self.device)
                x_t, noise = self.diffusion.noise_images(images, t)
                predicted_noise = self.ema.model(x_t, t)

                for n, pred, time in zip(noise, predicted_noise, t):
                    loss = self.criterion(pred, n)
                    losses[time].append(loss.item())

            average_losses = [np.mean(lst) if lst else None for lst in losses]

            print('Evaluation complete')
            self.ema.model.train()
        
        return average_losses