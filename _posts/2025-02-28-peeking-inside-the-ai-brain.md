---
layout: post
title: Peeking Inside the AI Brain
date: '2025-02-28'
permalink: /writing/peeking-inside-the-ai-brain/
tags:
- AI
- LLMs
- Interpretability
- Machine Learning
- Deep Learning
- Structural Analysis
---

## **Introduction: Why Look Inside AI Models?**

Imagine trying to fix a complex machine without being able to open it up and look inside. That's the challenge we face with modern AI systems, they're incredibly powerful, but often operate as "black boxes." Structural analysis is our toolkit for opening these boxes and understanding what's happening inside.

> This blog post is highly inspired by Stanford University's Natural Language Understanding course lectures available at [Stanford XCS224U: NLU](https://www.youtube.com/watch?v=K_Dh0Sxujuc&list=PLoROMvodv4rOwvldxftJTmoR3kRcWkJBp).

While traditional evaluation methods focus on what AI models output (behavioral analysis), structural analysis examines how they actually work internally. It's like the difference between judging a car by its speed versus understanding its engine mechanics.

<div class="post-content-image-container" style="
    width: calc(100% + 48px);
    margin-left: -24px;
    margin-right: -24px;
    margin-top: 24px;
    margin-bottom: 24px;
    text-align: center;
">
    <img 
        src="/images/Struc_analysis_images/Limitation_of_Behavioral_analysis.png" 
        alt="Limitations of Behavioral Analysis" 
        title="Limitations of Behavioral Analysis" 
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
        Limitations of Behavioral Analysis
    </div>
</div>

### **Why This Matters**

1. **Beyond the Black Box**: Traditional evaluation only shows us inputs and outputs, missing the crucial "how" and "why" of AI decisions.

2. **Finding Root Causes**: When AI systems make mistakes or show bias, structural analysis helps identify the underlying reasons rather than just the symptoms.

3. **Targeted Improvements**: Understanding internal mechanisms enables precise model improvements rather than trial-and-error fixes.

4. **Safety & Trust**: As AI systems become more powerful, understanding their inner workings becomes crucial for ensuring reliability and alignment with human values.

## **The Three Pillars of Structural Analysis**

### **1. Probing: The AI X-Ray**

Think of probing as taking X-rays of different parts of a neural network. Just as medical X-rays reveal bone structure, probes reveal what information is encoded in different parts of the model.

#### **How It Works**

1. **Linear Probes**: Simple classifiers that test if specific information (like syntax or semantics) can be extracted from layer activations.
   - Like basic X-rays showing bone structure
   - Quick and interpretable results
   - May miss complex patterns

2. **Non-linear Probes**: More sophisticated classifiers that can detect complex patterns.
   - Like advanced imaging (CT/MRI) showing detailed tissue structure
   - Can find subtle patterns
   - More computationally intensive

<div class="post-content-image-container" style="
    width: calc(100% + 48px);
    margin-left: -24px;
    margin-right: -24px;
    margin-top: 24px;
    margin-bottom: 24px;
    text-align: center;
">
    <img 
        src="/images/Struc_analysis_images/Probing.png" 
        alt="Neural Network Probing Methods" 
        title="Neural Network Probing Methods" 
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
        Probing different language models to analyze their internal representations
    </div>
</div>

### **2. Feature Attribution: Following the Decision Trail**

Feature attribution is like tracing a detective's investigation - it reveals which parts of the input were most important for the final decision.

#### **Key Methods**

1. **Saliency Maps**: Highlight which input elements (words, pixels) most influenced the output.
   - Like highlighting key evidence in a case
   - Visual and intuitive
   - Can sometimes be noisy

2. **Attention Analysis**: Shows how the model weighs different parts of the input.
   - Reveals what the model "focuses" on
   - Particularly useful for transformer models
   - Helps understand information flow

<div class="post-content-image-container" style="
    width: calc(100% + 48px);
    margin-left: -24px;
    margin-right: -24px;
    margin-top: 24px;
    margin-bottom: 24px;
    text-align: center;
">
    <img 
        src="/images/Struc_analysis_images/Feature_attr.png" 
        alt="Feature Attribution Methods Overview" 
        title="Feature Attribution Methods Overview" 
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
        Three main approaches to feature attribution: gradient-based, surrogate models, and perturbation methods
    </div>
</div>

### **3. Interventions: AI Surgery**

Like a neurosurgeon studying brain function by carefully modifying specific regions, interventions involve precisely changing parts of the network to understand their role.

#### **Common Techniques**

1. **Ablation Studies**: Temporarily disabling specific components to see their impact.
   - Like studying brain function by temporarily deactivating regions
   - Clear cause-effect relationships
   - Can reveal redundancy and critical components

2. **Activation Editing**: Modifying specific activations to change model behavior.
   - Precise control over model internals
   - Can test hypotheses about learned representations
   - Useful for understanding and fixing biases

<div class="post-content-image-container" style="
    width: calc(100% + 48px);
    margin-left: -24px;
    margin-right: -24px;
    margin-top: 24px;
    margin-bottom: 24px;
    text-align: center;
">
    <img 
        src="/images/Struc_analysis_images/intervention.png" 
        alt="Neural Network Intervention Methods" 
        title="Neural Network Intervention Methods" 
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
        Example of a REFT intervention showing how model behavior changes with targeted modifications
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
        src="/images/Struc_analysis_images/Framework_eval.png" 
        alt="Comparison of Structural Analysis Methods" 
        title="Comparison of Structural Analysis Methods" 
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
        Comparing how each method performs for different research goals
    </div>
</div>

## **The Challenges: Why This Is Hard**

Understanding AI internals faces several fundamental challenges:

### **1. Scale and Complexity**
- Modern models have billions of parameters
- Attention mechanisms scale quadratically
- Analyzing all interactions is computationally intractable

### **2. Distributed Knowledge**
- Information is spread across many neurons
- No single "grandmother cell" for concepts
- Features combine in complex ways

### **3. Non-linear Behavior**
- Multiple layers create complex transformations
- Small changes can have large effects
- Traditional linear analysis tools often fail

### **4. Causal Confusion**
- Hard to isolate individual effects
- Networks can compensate for changes
- Multiple paths to the same output

## **Looking Forward: The Future of AI Interpretability**

As AI systems become more powerful and ubiquitous, understanding their inner workings becomes increasingly crucial. The field of structural analysis is evolving rapidly, with new tools and techniques emerging regularly.

### **Key Developments to Watch**

1. **Automated Analysis Tools**: Making interpretability more accessible
2. **Real-time Monitoring**: Understanding AI decisions as they happen
3. **Standardized Methods**: Creating common frameworks for analysis


## **Citation**

<blockquote>
Nangia, Ayush. (Feb 2025). "Peeking Inside the AI Brain". Ayush Nangia. https://ayushnangia.github.io/writing/peeking-inside-the-ai-brain/
</blockquote>

Or

<pre><code>@article{vit2025llmstructure,
  title   = "Peeking Inside the AI Brain",
  author  = "Nangia, Ayush",
  journal = "Ayush Nangia",
  year    = "2025",
  month   = "Feb",
  url     = "https://ayushnangia.github.io/writing/peeking-inside-the-ai-brain/"
}</code></pre>

## **References**

<ol class="references">
    <li>Elhage, N., Hume, T., Olsson, C., Schiefer, N., Henighan, T., Kravec, S., Hatfield-Dodds, Z., Lasenby, R., Drain, D., Chen, C., Grosse, R., McCandlish, S., Kaplan, J., Amodei, D., Wattenberg, M., & Olah, C. (2022). Toy Models of Superposition. <i>arXiv preprint arXiv:2209.10652</i>.</li>
    <li>Varma, G. (2021). Feature Attribution in Explainable AI. <i>Geek Culture</i>. <a href="https://medium.com/geekculture/feature-attribution-in-explainable-ai-626f0a1d95e2">https://medium.com/geekculture/feature-attribution-in-explainable-ai-626f0a1d95e2</a></li>
    <li>Stanford University. (n.d.). XCS224U: Natural Language Understanding. [Video playlist]. YouTube. <a href="https://www.youtube.com/watch?v=K_Dh0Sxujuc&list=PLoROMvodv4rOwvldxftJTmoR3kRcWkJBp">https://www.youtube.com/watch?v=K_Dh0Sxujuc&list=PLoROMvodv4rOwvldxftJTmoR3kRcWkJBp</a></li>
</ol> 