# SeedVision AI: Deep Learning Applications (DLA) Scientific Methodology

## 1. Deep Learning Architectures

### 1.1 Model A: SeedNet-Baseline (Custom CNN)
Designed as a lightweight, domain-optimized convolutional network tailored for rapid CPU training and inference:
- **Feature Extractor**: Four convolutional blocks structured with:
  - $\text{Conv2D}(C_{\text{in}}, C_{\text{out}}, k=3, s=1, p=1)$
  - $\text{BatchNorm2d}(C_{\text{out}})$
  - $\text{LeakyReLU}(\text{negative\_slope}=0.1)$
  - $\text{MaxPool2D}(k=2, s=2)$
- Channel sequence: $3 \rightarrow 32 \rightarrow 64 \rightarrow 128 \rightarrow 256$.
- **Spatial Aggregation**: $\text{AdaptiveAvgPool2D}((1, 1))$, reducing activation tensors to a $256$-dimensional feature vector regardless of input image dimensions.
- **Classification Head**:
  - Linear layer: $256 \rightarrow 256$ units.
  - $\text{BatchNorm1D}(256)$ and $\text{ReLU}$.
  - $\text{Dropout}(p=0.35)$ for regularization.
  - Final projection layer: $256 \rightarrow 4$ quality classes.

### 1.2 Model B: MobileNetV2 (Transfer Learning)
Designed for resource-constrained edge inference and mobile deployment:
- Uses inverted residual bottleneck blocks with depthwise separable convolutions:
  - $1 \times 1$ expansion convolution.
  - $3 \times 3$ depthwise convolution.
  - $1 \times 1$ linear projection (bottleneck).
- Pre-trained on ImageNet-1k, utilizing early visual representations (edges, textures, gradients).
- Classification head replaced by a customized linear projection with dropout ($p=0.30$) and fine-tuning with a reduced learning rate ($3 \times 10^{-4}$).

## 2. Training Protocol & Regularization
- **Loss Function**: Categorical Cross-Entropy with mild label smoothing ($\epsilon = 0.05$):
  $$L_{\text{smooth}}(y, \hat{y}) = -\sum_{k=1}^K \left[ (1 - \epsilon)y_k + \frac{\epsilon}{K} \right] \log \hat{y}_k$$
- **Optimization**: Adam / AdamW with cosine annealing learning rate scheduler:
  $$\eta_t = \eta_{\text{min}} + \frac{1}{2}(\eta_{\text{max}} - \eta_{\text{min}})\left(1 + \cos\left(\frac{t}{T_{\text{max}}}\pi\right)\right)$$
- **Gradient Clipping**: Norm capped at $1.0$ to ensure numerical stability.
- **Early Stopping**: Validation accuracy and loss monitored with a patience of 6 epochs.

## 3. Explainability with Grad-CAM
Gradient-weighted Class Activation Mapping (Grad-CAM) produces visual explanations without architectural modifications. The gradient of class score $y^c$ with respect to feature activation $A^k$ at layer $l$:
$$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{ij}^k}$$

Heatmap synthesis:
$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$

The resulting map is normalized and blended as an alpha overlay:
$$\text{Overlay} = \alpha \cdot \text{Heatmap}_{\text{RGB}} + (1 - \alpha) \cdot \text{Image}_{\text{RGB}}$$
