import torch
import torch.nn as nn
from typing import List

class TabTransformer(nn.Module):
    """
    TabTransformer Architecture for Clinical Tabular Data.
    
    Reference:
    "Classification of Health Indicators for Diabetes Mellitus Prediction Using a TabTransformer Model on Clinical Tabular Data"
    (Al Khaidar & Sri Kurnia) and Huang et al. / Arik & Pfister (2021).
    
    Architecture Flow:
    1. Categorical Features -> Individual Learnable Embedding Tables (Dimension = 32)
    2. Stacked Embeddings [Batch, Num_Categories, 32] -> Multi-Head Self-Attention Transformer Blocks (4 layers, 8 heads, FF=128, GELU, Dropout=0.1)
    3. Contextual Categorical Representation -> Flattened to [Batch, Num_Categories * 32]
    4. Concatenation with Normalized Numerical Features -> [Batch, (Num_Categories * 32) + Num_Numerical]
    5. MLP Classifier:
       - Linear((Num_Categories * 32) + Num_Numerical, 64) -> ReLU -> Dropout(0.2)
       - Linear(64, 32) -> ReLU -> Dropout(0.2)
       - Linear(32, num_outputs) -> Logits
    """
    def __init__(
        self,
        cat_cardinalities: List[int],
        num_numerical: int,
        embed_dim: int = 32,
        num_heads: int = 8,
        num_layers: int = 4,
        ff_dim: int = 128,
        transformer_dropout: float = 0.1,
        mlp_hidden_dims: List[int] = [64, 32],
        mlp_dropout: float = 0.2,
        num_classes: int = 1
    ):
        super().__init__()
        self.num_categorical = len(cat_cardinalities)
        self.num_numerical = num_numerical
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        # 1. Categorical Feature Embeddings
        # Each categorical attribute receives its own dedicated embedding dictionary of size (cardinality x 32)
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=cardinality, embedding_dim=embed_dim)
            for cardinality in cat_cardinalities
        ])
        
        # 2. Transformer Encoder Blocks
        # Captures non-linear latent interactions and contextual dependencies across categorical attributes
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=transformer_dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
        # 3. Layer Normalization for Numerical Inputs (optional stabilizer)
        self.num_norm = nn.Identity()
        
        # 4. Multi-Layer Perceptron (MLP) Classifier
        # Input dimension = flattened contextual embeddings + numerical features
        in_mlp_dim = (self.num_categorical * embed_dim) + num_numerical
        
        mlp_layers: List[nn.Module] = []
        current_dim = in_mlp_dim
        for hidden_dim in mlp_hidden_dims:
            mlp_layers.append(nn.Linear(current_dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(mlp_dropout))
            current_dim = hidden_dim
            
        # Final prediction layer (output logits)
        mlp_layers.append(nn.Linear(current_dim, num_classes))
        self.mlp = nn.Sequential(*mlp_layers)
        
    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for TabTransformer.
        
        Args:
            x_num (torch.Tensor): Continuous numerical tensor of shape [batch_size, num_numerical]
            x_cat (torch.Tensor): Categorical index tensor of shape [batch_size, num_categorical]
            
        Returns:
            torch.Tensor: Prediction logits of shape [batch_size] (if binary) or [batch_size, num_classes]
        """
        # Step A: Project each categorical feature into 32-dimensional embedding
        # List of [batch_size, embed_dim] -> Stack into [batch_size, num_cat, embed_dim]
        cat_embeds = [
            emb_layer(x_cat[:, i])
            for i, emb_layer in enumerate(self.cat_embeddings)
        ]
        cat_stack = torch.stack(cat_embeds, dim=1) # [B, num_cat, 32]
        
        # Step B: Pass through Multi-Head Self-Attention Transformer blocks
        contextual_cat = self.transformer_encoder(cat_stack) # [B, num_cat, 32]
        
        # Step C: Flatten contextual categorical representations
        cat_flat = contextual_cat.flatten(start_dim=1) # [B, num_cat * 32]
        
        # Step D: Concatenate with normalized numerical features
        num_features = self.num_norm(x_num) # [B, num_numerical]
        combined = torch.cat([cat_flat, num_features], dim=1) # [B, (num_cat * 32) + num_numerical]
        
        # Step E: Classify via 2-layer MLP to obtain output logits
        logits = self.mlp(combined) # [B, num_classes]
        
        if self.num_classes == 1:
            return logits.squeeze(-1) # Return [B] for binary BCEWithLogitsLoss
        return logits
