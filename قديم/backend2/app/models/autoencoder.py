# app/models/autoencoder.py
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, frames, points, channels, latent_dim=128):
        super().__init__()
        self.frames = frames
        self.points = points
        self.channels = channels
        
        # Flatten input: (batch, frames, points, channels) -> (batch, frames * points * channels)
        input_dim = frames * points * channels
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, latent_dim)
        )
        
    def forward(self, x):
        # x shape: (batch, frames, points, channels)
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)  # Flatten all spatial and temporal dimensions
        return self.net(x_flat)

class Decoder(nn.Module):
    def __init__(self, frames, points, channels, latent_dim=128):
        super().__init__()
        self.frames = frames
        self.points = points
        self.channels = channels
        output_dim = frames * points * channels
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, output_dim),
            nn.Sigmoid()  # Assuming input is normalized to [0,1]
        )
        
    def forward(self, z):
        batch_size = z.shape[0]
        x_flat = self.net(z)
        # Reshape back to original dimensions
        x = x_flat.reshape(batch_size, self.frames, self.points, self.channels)
        return x

class AE(nn.Module):
    def __init__(self, frames, points, channels, latent_dim=128):
        super().__init__()
        self.encoder = Encoder(frames, points, channels, latent_dim)
        self.decoder = Decoder(frames, points, channels, latent_dim)
        
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z