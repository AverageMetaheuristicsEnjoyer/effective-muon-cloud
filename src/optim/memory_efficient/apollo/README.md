# 🚀 APOLLO: Approximated Gradient Scaling for Memory-Efficient LLM Optimization 

This repository contains the pre-release version of the **APOLLO** algorithm, proposed in our paper:  
[**APOLLO: SGD-like Memory, AdamW-level Performance**](https://arxiv.org/abs/2412.05270).

APOLLO achieves memory efficiency comparable to SGD while delivering performance on par with AdamW, making it an ideal optimizer for large language models (LLMs).

🌐 **Explore more**: Check out the [Project Page](https://zhuhanqing.github.io/APOLLO/) for additional demos showcasing APOLLO's memory-saving and throughput benefits.

## 📦 Installation

### Install APOLLO via pip
You can install the APOLLO optimizer directly from pip:
```bash
pip install apollo-torch
```

### Install APOLLO from source
To install APOLLO from the source code:

```bash
git clone git@github.com:zhuhanqing/APOLLO.git
cd APOLLO
pip install -e .
```

### Install experiment dependencies

```bash
pip install -r exp_requirements.txt
```