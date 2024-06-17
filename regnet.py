import torch
import torchvision.models as models

# Load the pre-trained RegNetY-400MF model
model = models.regnet_y_400mf(weights=None)

# Create a random input tensor with the shape (batch_size, channels, height, width)
# RegNet models expect 3 channels (RGB images), height and width should be 224
input_tensor = torch.randn(1, 3, 224, 224)

# Forward pass through the model
output = model(input_tensor)

# Print the output shape
print(output.shape)
