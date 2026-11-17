import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)
        # Layers: Embedding(vocabulary_size, 16) -> Linear(16, 1) -> Sigmoid
        pass
        self.model = nn.Sequential(
            nn.Embedding(vocabulary_size,16),
            nn.Linear(16,1),
            nn.Sigmoid())


    def forward(self, x: TensorType[int]) -> TensorType[float]:
        # Hint: The embedding layer outputs a B, T, embed_dim tensor
        # but you should average it into a B, embed_dim tensor before using the Linear layer

        # Return a B, 1 tensor and round to 4 decimal places
        pass
        # Embedding: (B, T) -> (B, T, 16)
        embeddings = self.model[0](x)

        # Average across sequence length: (B, T, 16) -> (B, 16)
        embeddings = embeddings.mean(dim=1)

        # Linear + Sigmoid: (B, 16) -> (B, 1)
        output = self.model[1:](embeddings)

        # Round to 4 decimal places
        return torch.round(output * 10000) / 10000
