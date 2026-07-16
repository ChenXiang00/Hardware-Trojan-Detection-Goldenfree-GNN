# Hardware-Trojan-Detection-Goldenfree-GNN
Example code of "Functional Hardware Trojan Detection in RTL Designs Using Data Flow Graph-Based Contrastive Graph Neural Networks".

### Introduction 
With the globalization of Integrated Circuit (IC) design and fabrication, the integration of third-party Intellectual Property (IP) and Electronic Design Automation (EDA) tools has significantly heightened the risk of **Hardware Trojans (HTs)**. These malicious modifications can lead to sensitive information leakage, functional disruption, and Denial-of-Service (DoS) attacks. Furthermore, the rise of generative AI has further increased the diversity and stealthiness of modern Trojan circuits. 

To address these critical security challenges, this repository implements the official framework proposed in our paper: **An inference-stage golden-reference-free RTL HT detection framework** that integrates Data Flow Graph (DFG) based graph representation learning and graph-level classification into a unified security analysis pipeline.

Our framework adopts a decoupled learning strategy that divides the pipeline into a two-stage process. In the first stage, contrastive pre-training utilizes Contrastive Learning (CL) to extract highly discriminative and robust graph representations from the RTL-converted Data Flow Graphs (DFGs). In the second stage, a separate downstream classifier is trained on these pre-trained representations to precisely distinguish between Trojan-free and Trojan-infected circuits without modifying the frozen encoder.

### Environment & Installation
We recommend using Python 3.10+ for this project. You can install all the required packages directly using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```
All experiments were conducted and verified under the following environment:
**OS**: Ubuntu 22.04.5 LTS (64-bit)
**Hardware**: Intel i7-13700 CPU | NVIDIA GeForce RTX 2080 Ti GPU (22 GB)
**CUDA**: 12.4

### Quick Start

Before running the code, please ensure that your pre-trained encoder model file (e.g., .pth) is already placed inside the runs directory at the root of this repository, and then execute the following command to start training the downstream classifier:

```bash
python 01_ClassifierTraining.py
```

