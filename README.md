# CycleGAN for Face-to-Sketch Conversion

This repository contains a PyTorch implementation of CycleGAN for converting face photographs to sketches and vice versa. The model is trained on a dataset of paired face photos and sketches to learn the bidirectional mapping between these two domains.

## Overview

CycleGAN is a type of Generative Adversarial Network (GAN) that learns to translate between two domains without paired training examples. In this implementation, we use it to convert between:
- Domain X: Real face photographs
- Domain Y: Artistic face sketches

The model consists of four main networks:
- `G_X2Y`: Generator for translating photos (X) to sketches (Y)
- `G_Y2X`: Generator for translating sketches (Y) to photos (X)
- `D_X`: Discriminator for photos domain
- `D_Y`: Discriminator for sketches domain

## Architecture

### Generator
The generator architecture follows the structure:
1. **Encoder**: Three convolutional blocks that downsample the input image
2. **Transformer**: Nine residual blocks that process feature maps
3. **Decoder**: Three transposed convolutional blocks that upsample to the output image

Each generator uses:
- Instance normalization
- ReLU activation functions
- Skip connections in residual blocks
- Tanh activation in the output layer

### Discriminator
The discriminator uses a series of convolutional blocks with:
- LeakyReLU activations
- Instance normalization
- A final convolutional layer for binary classification (real/fake)

## Data Processing

The data pipeline includes:
- Custom `GANdataGenerator` dataset class for loading and preprocessing images
- Image transformations (resize, center crop, normalization)
- Rescaling images to the range [-1, 1] to match tanh output range

The model is trained on:
- Face photographs from `/train/photos`
- Face sketches from `/train/sketches`

## Training Process

The training involves several key components:

### Loss Functions
1. **Adversarial Loss**: MSE loss for real/fake classification 
2. **Cycle Consistency Loss**: L1 loss between original and reconstructed images
3. **Identity Loss**: Optional L1 loss to ensure generators preserve same-domain inputs

### Training Loop
For each training iteration:
1. Train discriminators on real and fake images
2. Train generators with:
   - Adversarial loss from discriminators
   - Cycle consistency loss for reconstruction
   - Identity loss for domain preservation

### Visualization and Checkpoints
The training includes:
- Periodic visualization of generated images
- Regular checkpoint saving
- Sample generation for monitoring progress

## Usage

### Requirements
- PyTorch
- torchvision
- OpenCV
- PIL
- NumPy
- Matplotlib

### Training
```python
# Create model
G_X2Y, G_Y2X, D_X, D_Y = create_model(padding_mode='replicate')

# Define optimizers
g_optimizer = optim.Adam(g_params, lr=0.0002, betas=[0.5, 0.999])
d_x_optimizer = optim.Adam(D_X.parameters(), lr=0.0002, betas=[0.5, 0.999])
d_y_optimizer = optim.Adam(D_Y.parameters(), lr=0.0002, betas=[0.5, 0.999])

# Train the model
losses = training_loop(X_dataloader, Y_dataloader, X_dataloader, Y_dataloader, n_epochs=1000)
```

### Inference
```python
# Load a pretrained model
G_X2Y = CycleGenerator(conv_dim=64, n_resblocks=9, padding_mode='reflect')
G_X2Y.load_state_dict(torch.load('G_X2Y.pth'))
G_X2Y.eval()

# Convert an image
image_tensor = load_image('input.jpg')
with torch.no_grad():
    sketch_tensor = G_X2Y(image_tensor)
    
# Save or visualize the result
save_image(sketch_tensor, 'output_sketch.jpg')
```

## Results

The model demonstrates the ability to:
- Convert photographs to realistic sketches
- Maintain facial structure and identifying features
- Reconstruct original images through cycle consistency

## Checkpoints

The training process automatically saves model checkpoints every 10 epochs to:
- `G_X2Y.pth`: Photo-to-sketch generator
- `Y2X.pth`: Sketch-to-photo generator
- `D_X.pth`: Photo discriminator
- `D_Y.pth`: Sketch discriminator

## Implementation Details

### Key Features
- Residual blocks to help gradients flow through deep generator networks
- Instance normalization for style transfer tasks
- Reflection padding to avoid boundary artifacts
- Cycle consistency to enforce bijective mapping between domains
- Identity mapping to preserve color and content when appropriate

### Hyperparameters
- Learning rate: 0.0002
- Adam optimizer betas: [0.5, 0.999]
- Cycle consistency lambda: 10
- Identity loss lambda: 0.1
- Batch size: 16
- Image size: 128×128

## Acknowledgments

This implementation is based on the CycleGAN paper:
- [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) by Zhu et al.

## License

This project is available under the MIT License.
