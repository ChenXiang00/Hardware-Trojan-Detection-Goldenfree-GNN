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

For demonstration and quick testing, we provide a small sample dataset in the DataSet.zip file. Users can extract this file to access example circuit data and use it to verify the training and inference workflow.

### Quick Start
Before training the downstream classifier, first the GNN encoder using contrastive learning (CL). During this stage, augmented views of each circuit Data Flow Graph (DFG) are generated and processed by a shared GNN encoder to learn robust and discriminative graph representations. The resulting encoder is then used as the initialization for downstream Hardware Trojan classification.

For demonstration and quick evaluation, we also provide encoder checkpoint in this repository, allowing users to directly proceed to downstream classifier training without repeating this stage.

```bash
python 00_EncoderTraining.py
```
The training script provides configurable options for dataset paths, output directory, learning rate, batch size, number of epochs, GNN hidden dimension, number of layers, dropout rate, training device, graph augmentation settings, contrastive-learning temperature, prototype-loss parameters, validation ratio, random seed, and graph balancing. All parameters are assigned default values, so the encoder can be trained directly without additional arguments.

For example, users can override the default training configuration by specifying command-line arguments. The following command trains the encoder for 200 epochs with a batch size of 8:

```bash
python 00_EncoderTraining.py --epochs 200 --batch-size 8
```

Before running the code, please ensure that your pre-trained encoder model file (e.g., .pth) is already placed inside the runs directory at the root of this repository, and then execute the following command to start training the downstream classifier:

```bash
python 01_ClassifierTraining.py
```

The classifier training script provides configurable options for dataset paths, the pre-trained encoder checkpoint, output directory, batch size, number of epochs, learning rate, validation ratio, random seed, and training device. Model-related settings such as the hidden dimension, number of GNN layers, and dropout rate are also configurable and should generally remain consistent with the encoder configuration described above. All parameters are assigned default values, so the classifier can be trained directly once the required datasets and pre-trained encoder checkpoint are available.

```bash
python 01_ClassifierTraining.py --epochs 150 --batch-size 16
```
