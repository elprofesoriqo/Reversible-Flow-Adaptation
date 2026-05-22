# pyrefly: ignore [missing-import]
import jax
# pyrefly: ignore [missing-import]
import jax.numpy as jnp
# pyrefly: ignore [missing-import]
import flax.linen as nn

class MlpBlock(nn.Module):
    """Transformer MLP block."""
    mlp_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = nn.Dense(self.mlp_dim)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.out_dim)(x)
        return x

class EncoderBlock(nn.Module):
    """Transformer Encoder Block (Self-Attention + MLP)."""
    num_heads: int
    mlp_dim: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        y = nn.LayerNorm()(x)
        y = nn.SelfAttention(num_heads=self.num_heads)(y)
        x = x + y
        
        y = nn.LayerNorm()(x)
        y = MlpBlock(mlp_dim=self.mlp_dim, out_dim=x.shape[-1])(y)
        x = x + y
        return x

class VisionEncoder(nn.Module):
    """Vision Transformer (ViT) for depth maps."""
    feature_dim: int = 256
    patch_size: int = 8
    num_heads: int = 4
    num_layers: int = 2
    
    @nn.compact
    def __call__(self, vision_obs: jax.Array) -> jax.Array:
        # vision_obs shape: (Batch, H, W, C)
        b, h, w, c = vision_obs.shape
        
        # Patch Embedding (Conv2D with stride = patch_size)
        x = nn.Conv(features=128, kernel_size=(self.patch_size, self.patch_size), 
                    strides=(self.patch_size, self.patch_size), padding='VALID')(vision_obs)
        
        # Flatten patches: (Batch, H', W', Features) -> (Batch, Num_Patches, Features)
        num_patches = (h // self.patch_size) * (w // self.patch_size)
        x = x.reshape((b, num_patches, -1))
        
        # Add [CLS] Token
        cls_token = self.param('cls_token', nn.initializers.zeros, (1, 1, x.shape[-1]))
        cls_token = jnp.broadcast_to(cls_token, (b, 1, x.shape[-1]))
        x = jnp.concatenate([cls_token, x], axis=1)
        
        # Add Positional Encodings
        pos_embedding = self.param('pos_embedding', nn.initializers.normal(stddev=0.02), 
                                   (1, num_patches + 1, x.shape[-1]))
        x = x + pos_embedding
        
        # Transformer Encoder
        for _ in range(self.num_layers):
            x = EncoderBlock(num_heads=self.num_heads, mlp_dim=256)(x)
            
        x = nn.LayerNorm()(x)
        
        # Extract [CLS] Token output and project to feature_dim
        cls_out = x[:, 0]
        out = nn.Dense(self.feature_dim)(cls_out)
        
        return out

class UNet1D(nn.Module):
    """1D Denoising Network over Action Sequences."""
    action_dim: int = 12
    chunk_size: int = 16
    
    @nn.compact
    def __call__(self, noisy_actions: jax.Array, time_steps: jax.Array, cond_features: jax.Array) -> jax.Array:
        # Time Embedding
        t_emb = nn.Dense(128)(time_steps)
        t_emb = jnp.broadcast_to(t_emb[:, None, :], (noisy_actions.shape[0], self.chunk_size, 128))
        
        # Condition Embedding (Proprio + Vision)
        cond_emb = nn.Dense(128)(cond_features)
        cond_emb = jnp.broadcast_to(cond_emb[:, None, :], (noisy_actions.shape[0], self.chunk_size, 128))
        
        # Fusion
        x = jnp.concatenate([noisy_actions, t_emb, cond_emb], axis=-1)
        
        # 1D Convolutions mapping sequence -> sequence
        x = nn.Conv(features=256, kernel_size=(3,), padding='SAME')(x)
        x = nn.gelu(x)
        x = nn.Conv(features=256, kernel_size=(3,), padding='SAME')(x)
        x = nn.gelu(x)
        
        # Predict Noise (Epsilon)
        noise_pred = nn.Conv(features=self.action_dim, kernel_size=(3,), padding='SAME')(x)
        return noise_pred
