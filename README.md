# CycleGAN: Monet Style Transfer (Kaggle: GAN-Getting-Started)
This repository contains my solution for the Kaggle competition, GAN-Getting-Started, which challenged participants to implement a Generative Adversarial Network (GAN) to perform artistic style transfer, converting real-world photographs into the style of Claude Monet.

## Project Summary
The core of this project is a Cycle-Consistent Generative Adversarial Network (CycleGAN) trained on Google's Tensor Processing Units (TPU) for speed. The training focused on optimizing the balance between applying aggressive style and preserving the original photo's content.

| Metric | Result | Interpretation |
| :--- | :--- | :--- |
| **MiFID Score** | **192.42** | Solid result, significantly below the 1000 threshold for a student project. |
| **Model Stability** | Highly Stable | Confirmed by balanced adversarial loss curves (G $\approx 3.0$, D $\approx 0.65$). |
| **Optimal Hyperparameter** | $\mathbf{\lambda = 10.0}$ | Confirmed the standard weight for Cycle Consistency provides the best content/style balance. |

## Methodology and Architecture

The problem requires Unpaired Image-to-Image Translation, which means the model learns the style mapping between two collections of images (Photos and Monets) without having matched pairs. The CycleGAN solves this by enforcing a Cycle Consistency Loss (Photo $\rightarrow$ Monet $\rightarrow$ Photo $\approx$ Original Photo).

### Model components

* Generator ($G_{A \to B}$): Based on a ResNet-modified U-Net architecture. The U-Net structure with skip connections is crucial for translating style while maintaining high-resolution detail.
* Discriminator ($D_B$): Uses a PatchGAN architecture, which judges the realism of $70 \times 70$ patches of the output image instead of the whole image. This forces the Generator to learn realistic textures and local details (e.g., Monet brushstrokes).
* Custom Fix: The project uses a custom-defined InstanceNormalization layer to replace the deprecated tensorflow-addons library. Instance Normalization is vital for stable style transfer as it normalizes features based on the content of a single image instance, rather than the entire batch.

### Distributed Pipeline
The entire process utilizes a TPUStrategy and relies on optimized TFRecord loading. This required custom tf.data pipelines wrapped in strategy.distribute_datasets_from_function to ensure parallel data loading, which resolves device conflicts and keeps the TPU cores fully saturated.

## Key Results and Analysis

### Adversarial Loss History
The loss history chart confirms the stability of the adversarial training. The Generator (G) and Discriminator (D) quickly found a stable equilibrium (oscillating steadily) rather than collapsing, proving the success of the optimization setup. Throughout the experimentation the baseline lambda prevailed. 

| Lambda (λ) | Final Avg G Loss | Interpretation |
| :--- | :--- | :--- |
| **10.00000** | **2.0507** | **Optimal:** Best balance; lowest loss due to stable content preservation. |
| 5.000000 | 2.1793 | High Style Aggression; higher loss due to content distortion. |
| 20.000000 | 2.4061 | Content Priority; highest loss because weak style fails the adversarial test. |

### How to Run the Project
1. Clone the Repository.
2. Ensure Kaggle/Colab Environment: This notebook requires a TPU or high-end GPU accelerator.
3. Install Dependencies: Install all required libraries (TensorFlow, Scikit-learn, etc.)
4. Run Notebook: Execute the CycleGAN_Monet_Notebook.ipynb sequentially to train the model and generate the output images.
