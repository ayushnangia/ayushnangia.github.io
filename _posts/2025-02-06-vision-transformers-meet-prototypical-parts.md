---
layout: post
kind: technical
title: Vision Transformers Meet Prototypical Parts
date: '2025-02-06'
permalink: /writing/vision-transformers-meet-prototypical-parts/
math: true
tags:
- Interpretability
- Vision Transformers
- Part Prototype Networks
- Intrinsic interpretability
- Machine Learning
- Deep Learning
---

## Introduction

**Vision Transformers have revolutionized computer vision.** But like many deep learning models, they often lack interpretability. ProtoViT bridges this gap by combining:
- **Vision Transformer's power** (self-attention and patch-based processing)
- **Prototype Network's interpretability** (learning from recognizable examples)

👉 **For background on prototype networks** → [**Discussing ProtoPNet**](https://ayushnangia.github.io/writing/discussing-protopnet/)

---

## A Quick Refresh 

Before diving into ProtoViT, let's revisit the core ideas behind **Vision Transformers (ViT)**:  

<div style="text-align: center; margin: 20px;">
    <img src="/images/PVIT/Vision_Transformer.gif" 
         alt="Vision Transformer Architecture" 
         title="Vision Transformer Architecture"
         style="width: 600px; margin: 0 auto;">
    <div style="font-size: 14px; color: #555; margin-top: 10px; text-align: center;">
        Animation of ViT
    </div>
</div>


 1. **Patch-based Processing**  
   Images are split into fixed-size patches (e.g., 16x16 pixels). These patches are flattened and linearly embedded, turning visual data into a sequence of "tokens".  

 2. **Position Embeddings**  
   Unlike CNNs, ViTs have no inherent spatial awareness. Position embeddings are added to preserve spatial relationships between patches.  

 3. **Self-Attention Layers**  
   Multiple transformer layers learn **global relationships** between patches. Each layer weighs the relevance of one patch to another, enabling dynamic feature learning.  

 4. **MLP Head for Classification**  
   After processing through transformer blocks, a final MLP (Multilayer Perceptron) head generates class predictions.  

> **Takeaway**: ViTs replace convolutional layers with a pure transformer architecture, treating images as sequences and leveraging attention for long-range dependencies. 


---

## ProtoViT: The Best of Both Worlds


<div class="post-content-image-container" style="
    width: calc(100% + 48px);
    margin-left: -24px;
    margin-right: -24px;
    margin-top: 24px;
    margin-bottom: 24px;
    text-align: center;
">
    <img 
        src="/images/PVIT/ProtoVit_arch.png" 
        alt="ProtoViT Architecture Overview" 
        title="ProtoViT Architecture Overview" 
        style="
            width: 100%;
            max-width: 1000px;
            height: auto;
            display: block;
            padding: 0 24px;
        "
    >
    <div class="post-content-image-caption" style="
        font-size: 14px;
        color: #555;
        margin-top: 10px;
        text-align: center;
        padding: 0 24px;
    ">
        ProtoViT Architecture Overview
    </div>
</div>

ProtoViT combines Vision Transformers with prototypical parts learning. Key innovations include:
1. **Transformer-based feature extraction**
2. **Greedy matching and Sigmoid-based slots**
3. **Interpretable decision process**

---

## Architecture Deep Dive

### 1. **Vision Transformer Backbone**

<div class="post-content-image-container" style="
    width: calc(100% + 48px);
    margin-left: -24px;
    margin-right: -24px;
    margin-top: 24px;
    margin-bottom: 24px;
    text-align: center;
">
    <img 
        src="/images/PVIT/ProtoVIT_encoder.png" 
        alt="ProtoViT Encoder" 
        title="ProtoViT Encoder" 
        style="
            width: 100%;
            max-width: 1000px;
            height: auto;
            display: block;
            padding: 0 24px;
        "
    >
    <div class="post-content-image-caption" style="
        font-size: 14px;
        color: #555;
        margin-top: 10px;
        text-align: center;
        padding: 0 24px;
    ">
        ProtoViT Encoding Process
    </div>
</div>

**Input Processing:**
- Images are split into patches (14×14 pixels)
- Each patch is flattened and linearly projected into a d-dimensional embedding space (d=768)
- Position embeddings are added to maintain spatial information

**ViT Backbone:**
- Standard transformer layers process the sequence of patch embeddings
- A learnable [CLS] token is prepended to patch embeddings
- Multi-head self-attention layers (12 heads) capture global relationships

**Salient Feature Extraction:**
```python
# Highlighting distinctive local features
patch_features = patch_tokens - cls_token.expand_as(patch_tokens)
```

This can be written mathematically as:


<div style="text-align: center;">

$z_f = [z^1_f, z^2_f, \cdots, z^N_f]$, where $z^i_f = z^i_{patch} - z_{class}$, and $z^i_f \in \mathbb{R}^d$
</div>

### 2. **Greedy matching and Prototype Layers**

<div style="text-align: center; margin: 20px;">
    <div class="video-container" style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%;">
        <iframe 
            style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"
            src="https://www.youtube.com/embed/rmVhsIVvkK4"
            title="ProtoVIT: Greedy Matching and Prototype Learning"
            frameborder="0"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
            allowfullscreen>
        </iframe>
    </div>
    <div style="font-size: 14px; color: #555; margin-top: 10px; text-align: center;">
        Greedy Matching Visualization
    </div>
</div>

This layer is the heart of ProtoViT's interpretability. It enables the model to make explainable predictions by comparing input images with learned prototypes - key visual patterns that represent meaningful parts of objects.

At its core, ProtoViT learns a set of `m` prototypes, each composed of `K` sub-prototypes that represent different aspects of a visual concept. For example, a bird's beak prototype might have sub-prototypes for its tip, middle, and base. Mathematically, we represent this as:

<div style="text-align: center; margin: 30px 0;">

$P = \{p_1, p_2, ..., p_m\}$

where each prototype $p_j = [p^1_j, p^2_j, ..., p^K_j]$
</div>

The layer works by matching these sub-prototypes to the most similar patches in the input image. This matching process uses cosine similarity to measure how well each image patch aligns with a sub-prototype:

<div style="text-align: center; margin: 30px 0;">

$cos(z^i_f, p^k_j) = \frac{z^i_f \cdot p^k_j}{||z^i_f|| \cdot ||p^k_j||}$
</div>

> **Note:** Unlike ProtoPNet which uses L2 distance, ProtoViT opts for cosine similarity. This choice makes the matching process more robust to variations in feature magnitudes, focusing instead on the directional alignment of feature vectors.

To ensure the matches make visual sense, the layer employs two key mechanisms:

1. **Adjacency Masking** ensures that matched patches are spatially close to each other. This prevents the model from matching disconnected parts of the image to the same prototype. For instance, when identifying a bird's beak, all matched patches should be near each other.

2. **Adaptive Slots** dynamically determine which sub-prototypes are important for the current image. Each sub-prototype gets an importance score that determines whether it should be included:

<div style="text-align: center; margin: 30px 0;">

$\tilde{1}_{\{\text{include }p^k_j\}} = \text{Sigmoid}(v^k_j, \tau)$
</div>


The adaptive slots mechanism dynamically determines which sub-prototypes to include when computing prototype similarities. Let's break down its components:

- The indicator function $\tilde{1}_{\{\text{include }p^k_j\}} $ determines whether to include the $k$-th sub-prototype of prototype $j$.
- It uses a learnable parameter $\mathbf{v}_{j}^{k}$ that represents the importance of each sub-prototype.
- The temperature parameter $\tau$ controls the sharpness of the decision boundary through the sigmoid function.

For example, given a bird's beak prototype with three sub-prototypes (tip, middle, base), the mechanism might learn:
- High $\mathbf{v}_{j}^{k}$ for the tip ($\text{Sigmoid}(5.0, \tau) \approx 1$)
- Moderate for the middle ($\text{Sigmoid}(2.0, \tau) \approx 1$)
- Low for the base ($\text{Sigmoid}(-1.0, \tau) \approx 0$)

This allows the model to focus on the most relevant parts (tip and middle) while ignoring less important ones (base), making the similarity computation more precise and interpretable.

The final similarity score combines all these elements, weighing the contribution of each matched sub-prototype by its importance:

<div style="text-align: center; margin: 20px;">
    <img src="/images/PVIT/compute similarity formula.png" 
         alt="Compute Similarity Formula" 
         title="Compute Similarity Formula"
         style="width: 600px; margin: 0 auto;">
    <div style="font-size: 14px; color: #555; margin-top: 10px; text-align: center;">
        Compute Similarity Formula
    </div>
</div>

Let's understand the intuition behind this formula:

- **Rescaling Factor** ($\frac{K}{\sum \tilde{1}}$): This ensures prototypes with fewer included sub-prototypes aren't unfairly penalized. For instance, if only 2 out of 3 sub-prototypes of a bird's beak are visible in a side view, the similarity score is scaled by $\frac{3}{2}$ to maintain fair comparison.

This mechanism allows ProtoViT to break down its decisions into interpretable steps. When classifying a bird image, it might tell us: <br>"I see a bird because I found a beak-like pattern here (showing the matched patches), wing features there, and tail features over there."

### 3. **Evidence layer**

The Evidence Layer is the final component that converts prototype similarity scores into class predictions. It takes the similarity scores $g_{p_j}^{greedy}(z_f)$ from the Greedy matching and Prototype Layers and processes them through a fully connected layer making the final prediction which is interpretable through the prototype activations.


## Training Process:

> 👉 For background on prototype network training, see [**Discussing ProtoPNet: Training Stages**](https://ayushnangia.github.io/writing/discussing-protopnet/#training-stages)

### 1. **Optimization of layers before last layer**
   
The first phase optimizes all layers except the last one using SGD, with the goal of learning a latent space where feature patches naturally cluster near semantically relevant prototypes of their class. During initialization, all slot indicators $ \tilde{1}_{\{\text{include }p^k_j\}} $ are set to 1, and the last layer weights are initialized to connect each prototype to its corresponding class.

The training uses four specialized loss functions:

<div style="text-align: center; margin: 30px 0;">

$L_{total} = L_{CE} + \lambda_1L_{Clst} + \lambda_2L_{Sep} + \lambda_3L_{Coh} + \lambda_4L_{Orth}$
</div>

- **Cross-Entropy Loss** ($L_{CE}$): Standard classification loss that measures how well the model predicts the correct class:

- **Cluster Loss** ($L_{Clst}$): Ensures each training image has features close to at least one sub-prototype of its class:

<div style="text-align: center; margin: 30px 0;">  
  $L_{Clst} = -\frac{1}{n}\sum_{i=1}^n \max_{p_j \in P_{y_i}} \max_{p^k_j \in p_j} \max_{z^i_f \in z_{A^k_j}} cos(z^i_f, p^k_j)$
</div>

where:
- $P_{y_i}$ is the set of prototypes belonging to the true class of image $i$
- $p^k_j$ is the $k$-th sub-prototype of prototype $j$
- $z^i_f$ is a feature token from image $i$
- $z_{A^k_j}$ is the set of feature tokens in the adjacency region of the $k$-th sub-prototype

- **Separation Loss** ($L_{Sep}$): Pushes features away from prototypes of incorrect classes:

<div style="text-align: center; margin: 30px 0;">  
  $L_{Sep} = \frac{1}{n}\sum_{i=1}^n \max_{p_j \notin P_{y_i}} \max_{p^k_j \in p_j} \max_{z^i_f \in z_{A^k_j}} cos(z^i_f, p^k_j)$
</div>

where:
- $p_j \notin P_{y_i}$ indicates prototypes not belonging to the true class of image $i$

- **Coherence Loss** ($L_{Coh}$): Makes sub-prototypes within the same prototype semantically similar:

<div style="text-align: center; margin: 30px 0;">  
  $L_{Coh} = \frac{1}{m}\sum_{j=1}^m \max_{p^k_j, p^s_j \in p_j} (1-cos(p^k_j, p^s_j)) \cdot \tilde{1}_{\{\text{include }p^k_j\}} \tilde{1}_{\{\text{include }p^s_j\}}$
</div>

where:
- $m$ is the number of prototypes
- $p^k_j, p^s_j$ are pairs of sub-prototypes within prototype $j$ *not the same prototype*
- $\tilde{1}$ is the indicator function determining sub-prototype inclusion

- **Orthogonality Loss** ($L_{Orth}$): Promotes diversity and learns distinctive features among prototypes of the same class:

<div style="text-align: center; margin: 30px 0;">     
  $L_{Orth} = \sum_{l=1}^C \|P^{(l)}P^{(l)T} - I_\rho\|_F^2$
</div>

where:
- $C$ is the number of classes
- $P^{(l)}$ represents all prototypes belonging to class $l$, where each prototype is flattened into a single row vector of dimension $Kd$ (K sub-prototypes × d-dimensional features)
- $I_\rho$ is the identity matrix
- $\|\cdot\|_F$ denotes the Frobenius norm, which measures how far $P^{(l)}P^{(l)T}$ deviates from the identity matrix, effectively quantifying how orthogonal (dissimilar) the prototypes are to each other

### 2. **Slots Pruning**
   
The second phase focuses on removing sub-prototypes that don't align semantically with others in the same prototype. This pruning process ensures each prototype maintains a coherent visual concept.

**Key Aspects:**
- All model parameters are frozen except for slot indicators $\mathbf{v}^k_j$
- Uses a simplified loss function:

<div style="text-align: center; margin: 30px 0;">
$L_{prune} = L_{CE} + \lambda_5L_{Coh}$
</div>

where $\lambda_5$ is intentionally reduced to prevent over-aggressive pruning.

The slot indicators are approximated using a sigmoid function with high temperature $\tau$, pushing values closer to binary decisions (0 or 1). After training completes, these values are rounded to create a definitive binary mask for each sub-prototype.

### 3. **Prototype Projection**

This phase anchors prototypes to real image patches, making them directly interpretable. The process involves:

**Steps:**
- Slot indicators are frozen with $\tilde{1}_{\{\text{include }p^k_j\}} = 0$ permanently excluding those sub-prototypes
- Each remaining prototype $p_j$ is projected onto the closest training image patch in the latent space using cosine similarity 
- Thanks to ViT's patch-based architecture, these prototypes can be visualized directly without any upsampling

> **Theoretical Guarantee**: If prototypes are well-trained (showing minimal movement during projection), the model's performance remains stable after projection. This guarantee comes from ProtoPNet's theoretical foundations.

### 4. **Optimization of the Last Layer**

This final phase optimizes the classification layer to enhance sparsity and interpretability while maintaining performance.

**Procedure:**
- Only the last layer weights $h$ are trained through convex optimization
- All other parameters remain frozen
- Uses a specialized loss function combining cross-entropy with L1 regularization:

<div style="text-align: center; margin: 30px 0;">
$L_h = L_{CE} + \lambda_6\sum_{b=1}^C\sum_{l=1,l\neq b}^C \|W_{b,l}\|_1$
</div>

where:
- $W_{b,l}$ represents weights connecting prototypes of class $l$ to output class $b$ (where $b \neq l$)
- $\|W_{b,l}\|_1$ is the L1 norm promoting sparsity
- $\lambda_6$ controls the strength of the sparsity penalty

The L1 penalty encourages the model to use only the most relevant prototypes for each class prediction, enhancing interpretability.

This staged training approach carefully balances model performance with interpretability, resulting in prototypes that are both discriminative and semantically meaningful.

> **Hardware Note:** While ProtoPNet typically requires multiple high-end GPUs (2×A100 or 3-4×V100), ProtoViT can train on a single high-memory GPU:
> - 1× NVIDIA Quadro RTX 6000 (24GB) or
> - 1× NVIDIA GeForce RTX 4090 (24GB) or
> - 1× NVIDIA RTX A6000 (48GB)

---

## Comparison with Baseline


<div class="post-content-image-container" style="
    width: calc(100% + 48px);
    margin-left: -24px;
    margin-right: -24px;
    margin-top: 24px;
    margin-bottom: 24px;
    text-align: center;
">
    <img 
        src="/images/PVIT/Results.png" 
        alt="Comparison with Baseline Models" 
        title="Comparison with Baseline Models"
        style="
            width: 100%;
            max-width: 800px;
            height: auto;
            display: block;
            padding: 0 24px;
        "
    >
    <div class="post-content-image-caption" style="
        font-size: 14px;
        color: #555;
        margin-top: 10px;
        text-align: center;
        padding: 0 24px;
    ">
        Comparison with Baseline Models
    </div>
</div>


---

## Advantages Over Traditional Approaches

For a technical comparison with other prototype-based approaches:

| Model                      | Support ViT Backbone? | Deformable Prototypes? | Coherent Prototypes? | Adaptive Sizes? | Inherently Interpretable? |
|----------------------------|-----------------------|------------------------|----------------------|-----------------|----------------------------|
| ProtoPNet                  | Yes                   | No                     | Maybe                | No              | Yes                        |
| Deformable ProtoPNet       | No                    | Yes                    | No                   | No              | Yes                        |
| ProtoPformer               | Yes                   | No                     | Maybe                | No              | No                         |
| **ProtoViT**               | Yes                   | Yes                    | Yes                  | Yes             | Yes                        |

---
## Comparison of Different Model Backbones


<div class="post-content-image-container" style="
    width: calc(100% + 48px);
    margin-left: -24px;
    margin-right: -24px;
    margin-top: 24px;
    margin-bottom: 24px;
    text-align: center;
">
    <img 
        src="/images/PVIT/Backbone_comp.png" 
        alt="Backbone Architecture Comparison" 
        title="Backbone Architecture Comparison" 
        style="
            width: 100%;
            max-width: 800px;
            height: auto;
            display: block;
            padding: 0 24px;
        "
    >
    <div class="post-content-image-caption" style="
        font-size: 14px;
        color: #555;
        margin-top: 10px;
        text-align: center;
        padding: 0 24px;
    ">
        Backbone Architecture Comparison
    </div>
</div>

---

## Results and Visualizations

### Analysis: ProtoPNet vs ProtoVit

<div class="post-content-image-container" style="
    width: calc(100% + 48px);
    margin-left: -24px;
    margin-right: -24px;
    margin-top: 24px;
    margin-bottom: 24px;
    text-align: center;
">
    <img 
        src="/images/PVIT/ProtoPNet_analysis.png" 
        alt="ProtoPNet Analysis" 
        title="ProtoPNet Analysis"
        style="
            width: 100%;
            max-width: 600px;
            height: auto;
            display: block;
            padding: 0 24px;
        "
    >
    <div class="post-content-image-caption" style="
        font-size: 14px;
        color: #555;
        margin-top: 10px;
        text-align: center;
        padding: 0 24px;
    ">
        ProtoPNet Analysis Visualization
    </div>
</div>

<div class="post-content-image-container" style="
    width: calc(100% + 48px);
    margin-left: -24px;
    margin-right: -24px;
    margin-top: 24px;
    margin-bottom: 24px;
    text-align: center;
">
    <img 
        src="/images/PVIT/ProtoVit_analysis.png" 
        alt="ProtoVit Analysis" 
        title="ProtoVit Analysis"
        style="
            width: 100%;
            max-width: 600px;
            height: auto;
            display: block;
            padding: 0 24px;
        "
    >
    <div class="post-content-image-caption" style="
        font-size: 14px;
        color: #555;
        margin-top: 10px;
        text-align: center;
        padding: 0 24px;
    ">
        ProtoVit Analysis Visualization
    </div>
</div>

### Classification Examples

<div class="post-content-image-container" style="
    width: calc(100% + 48px);
    margin-left: -24px;
    margin-right: -24px;
    margin-top: 24px;
    margin-bottom: 24px;
    text-align: center;
">
    <img 
        src="/images/PVIT/jeff bezos0_2040.jpg_analysis.png" 
        alt="Jeff Bezos Classification Analysis" 
        title="Jeff Bezos Classification Analysis"
        style="
            width: 100%;
            max-width: 1600px;
            height: auto;
            display: block;
            padding: 0 24px;
        "
    >
    <div class="post-content-image-caption" style="
        font-size: 16px;
        color: #555;
        margin-top: 15px;
        text-align: center;
        padding: 0 24px;
    ">
        Example of Strong Classification: The model identifies key facial features through well-matched prototypes
    </div>
</div>

<div class="post-content-image-container" style="
    width: calc(100% + 48px);
    margin-left: -24px;
    margin-right: -24px;
    margin-top: 24px;
    margin-bottom: 24px;
    text-align: center;
">
    <img 
        src="/images/PVIT/Emma Watson104_1846.jpg_analysis.png" 
        alt="Emma Watson Classification Analysis" 
        title="Emma Watson Classification Analysis"
        style="
            width: 100%;
            max-width: 1600px;
            height: auto;
            display: block;
            padding: 0 24px;
        "
    >
    <div class="post-content-image-caption" style="
        font-size: 16px;
        color: #555;
        margin-top: 15px;
        text-align: center;
        padding: 0 24px;
    ">
        Example of Moderate Classification: While the model correctly identifies the person, some prototype matches are less precise
    </div>
</div>

<div class="post-content-image-container" style="
    width: calc(100% + 48px);
    margin-left: -24px;
    margin-right: -24px;
    margin-top: 24px;
    margin-bottom: 24px;
    text-align: center;
">
    <img 
        src="/images/PVIT/Mark Zuckerberg107_1993.jpg_analysis_2.png" 
        alt="Mark Zuckerberg Classification Analysis" 
        title="Mark Zuckerberg Classification Analysis"
        style="
            width: 100%;
            max-width: 1600px;
            height: auto;
            display: block;
            padding: 0 24px;
        "
    >
    <div class="post-content-image-caption" style="
        font-size: 16px;
        color: #555;
        margin-top: 15px;
        text-align: center;
        padding: 0 24px;
    ">
        Example of Misclassification: Despite finding some relevant facial features, the model fails to make the correct identification
    </div>
</div>

---

## Limitations 

1. **Explainability Gaps**
   - Lacks textual explanations for its reasoning
   - Requires domain-specific vocabularies for specialized fields
   - Visual explanations may not align with human-interpretable concepts without post-hoc analysis

2. **Technical Challenges**
   - Location misalignment in deeper layers
   - Resolution constraints (optimal at 224×224 pixels)
   - Performance degradation with domain shifts

3. **Data Dependencies**
   - Training bias from specific datasets
   - Limited generalization to dissimilar domains
   - Prototype interpretability varies with data quality

---

## Implementation Resources

### Official Implementation
The original implementation is available on GitHub:
<br>
[**ProtoViT Official Repository**](https://github.com/Henrymachiyu/ProtoViT)

### Community Implementation
An improved fork with dockerfile and preprocessing scripts:
<br>
[**ProtoViT Enhanced Fork**](https://github.com/ayushnangia/ProtoViT)

### Pre-trained Models

**CUB-200-2011 Dataset**
- Model: DeiT-Small 
- 🤗 [**Download Weights**](https://huggingface.co/Ayushnangia/protovit-deit_small_patch16_224-cub)

**Pinterest Face Recognition Dataset**
- Model: DeiT-Small 
- 🤗 [**Download Weights**](https://huggingface.co/Ayushnangia/protovit-deit_small_patch16_224-pins)

---

## Citation
Cited as:

<blockquote>
Nangia, Ayush. (Feb 2025). "Vision Transformers Meet Prototypical Parts". Ayush Nangia. https://ayushnangia.github.io/writing/vision-transformers-meet-prototypical-parts/
</blockquote>

Or

<pre><code>@article{vit2025protovit,
  title   = "Vision Transformers Meet Prototypical Parts",
  author  = "Nangia, Ayush",
  journal = "Ayush Nangia",
  year    = "2025",
  month   = "Feb",
  url     = "https://ayushnangia.github.io/writing/vision-transformers-meet-prototypical-parts/"
}</code></pre>

## References

<ol class="references">
    <li>Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. <i>arXiv preprint arXiv:2010.11929</i>. <a href="https://arxiv.org/abs/2010.11929">https://arxiv.org/abs/2010.11929</a></li>
    <li>Chen, C., Li, O., Tao, C., Barnett, A. J., Su, J., & Rudin, C. (2019). This Looks Like That: Deep Learning for Interpretable Image Recognition. <i>arXiv preprint arXiv:1806.10574</i>. <a href="https://arxiv.org/abs/1806.10574">https://arxiv.org/abs/1806.10574</a></li>
    <li>Ma, C., Donnelly, J., Liu, W., Vosoughi, S., Rudin, C., & Chen, C. (2024). Interpretable Image Classification with Adaptive Prototype-based Vision Transformers. <i>arXiv preprint arXiv:2410.20722</i>. <a href="https://arxiv.org/abs/2410.20722">https://arxiv.org/abs/2410.20722</a></li>
</ol>
