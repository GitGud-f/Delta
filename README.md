# Delta: A Lightweight Monocular Depth Estimation Model via Knowledge Distillation

<p align="center">
  <a href="https://doi.org/10.5281/zenodo.23856641">
    <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.23856641.svg" alt="DOI">
  </a>
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/ONNX_Runtime-005CED?logo=onnx&logoColor=white" alt="ONNX Runtime">
  <img src="https://img.shields.io/badge/Flutter-02569B?logo=flutter&logoColor=white" alt="Flutter">
  <img src="https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg" alt="License">
</p>

**Delta** is an end-to-end Edge AI framework for monocular depth estimation on resource-constrained devices. By leveraging **knowledge distillation**, Delta compresses a large, high-capacity Vision Transformer teacher (**Depth Anything V2**) into an ultra-compact student model featuring a hybrid **MobileViT-XS** backbone and a lightweight **MiniDPT** decoder.

The resulting model contains only **3.4M parameters** (~7× smaller than the teacher) and achieves real-time inference (**186 ms**) on an iPhone 13 via an accompanying cross-platform **Flutter** application with ONNX Runtime acceleration.

---

## 📄 Publication / Technical Report

If you use this work or codebase, please cite our technical report:

> **Mayas Al Khateeb, Nadim Arubai, and Omar Hamdoun.**  
> *"Delta: A Lightweight Monocular Depth Estimation Model via Knowledge Distillation."*  
> Technical Report, Higher Institute for Applied Sciences and Technology (HIAST), 2026.  
> **DOI:** [10.5281/zenodo.23856641](https://doi.org/10.5281/zenodo.23856641)

```bibtex
@article{alkhateeb2026delta,
  author    = {Al Khateeb, Mayas and Arubai, Nadim and Hamdoun, Omar},
  title     = {Delta: A Lightweight Monocular Depth Estimation Model via Knowledge Distillation},
  journal   = {Zenodo},
  year      = {2026},
  doi       = {10.5281/zenodo.23856641},
  url       = {https://doi.org/10.5281/zenodo.23856641}
}
```

---

## 🖼️ Teaser & Demo

![Teaser](docs/Teaser.png)

---

## ✨ Key Features

- **Teacher-Student Distillation:** Compresses the state-of-the-art **Depth Anything V2 (Small)** model into a lightweight student.
- **Hybrid MobileViT-XS + MiniDPT Architecture:** Combines CNN inductive bias for local feature extraction with Vision Transformer self-attention for global scene structure.
- **Composite Knowledge Distillation Loss:** Multi-objective loss combining Scale-Invariant Logarithmic (SILog), Gradient Matching, Intermediate Feature Matching, and Attention Map Matching.
- **Mobile Edge Acceleration:** Exported to **ONNX** format, running through a cross-platform **Flutter** app with background Dart Isolates to maintain smooth UI rendering.
- **Complete End-to-End Pipeline:** Scripts for training, distillation loss tuning, ONNX conversion, and native mobile deployment.

---

## 📊 Benchmark & Performance

### 1. Accuracy Comparison (NYU Depth V2 Test Set)
Evaluated zero-shot on NYU Depth V2:

| Model | Parameters | Abs Rel ↓ | RMSE ↓ | RMSE Log ↓ | $\delta < 1.25$ ↑ | $\delta < 1.25^2$ ↑ | $\delta < 1.25^3$ ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Depth Anything V2 (Teacher)** | 24.8M | 0.053 | 0.340 | 0.054 | 0.973 | 0.983 | 0.992 |
| **Delta (Student, Ours)** | **3.4M** *(~7× smaller)* | **0.203** | **0.781** | **1.371** | **0.702** | **0.832** | **0.933** |

### 2. Runtime Latency Across Hardware
Measured at $384 \times 384$ resolution:

| Device Configuration | Execution Engine | Inference Time |
| :--- | :---: | :---: |
| **Colab (Tesla T4 GPU)** | PyTorch (FP32) | **13.2 ± 3 ms** |
| **Laptop (Intel Core i7)** | PyTorch (CPU) | **576.8 ± 10 ms** |
| **iPhone 13 (Apple A15)** | **ONNX Runtime (Mobile)** | **186 ± 10 ms** *(~5.4 FPS)* |

*In comparison, the teacher model takes over 10.7 seconds per frame on an iPhone 13, making Delta uniquely suited for mobile deployment.*

---

## 📂 Project Structure

```text
delta/
├── app/                  # Cross-platform Flutter mobile application
│   ├── assets/           # Location for the optimized Delta.onnx model
│   ├── lib/              # Application UI, ONNX Runtime bridge, Dart Isolates
│   └── pubspec.yaml
└── model/                # PyTorch training and distillation pipeline
    ├── config/           # Training hyperparameters and loss weight configs
    ├── criterions/       # Composite distillation loss functions (SILog, Grad, Feat, Attn)
    ├── data/             # Training and evaluation image splits
    ├── datasets/         # Data loaders and augmentations
    ├── models/           # Teacher wrapper and Delta (MobileViT + MiniDPT)
    ├── notebooks/        # Prototyping and analysis notebooks
    ├── scripts/          # train.py, infer.py, pth2onnx.py
    ├── utils/            # Model conversion, metrics, and helper tools
    └── requirements.txt  # Python environment dependencies
```

For more documentation, see:
- [**Model Documentation**](docs/model_docs.md)
- [**Flutter App Documentation**](docs/flutter_demo.md)

---

## 🚀 Getting Started

### 1. Python Environment & Model Training

```bash
# Clone the repository
git clone https://github.com/GitGud-f/Delta.git
cd Delta/model

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Run Inference
```bash
python -m scripts.infer --model-path checkpoints/Delta.pth --input-path data/TestImages
```

#### Export to ONNX
```bash
python -m scripts.pth2onnx checkpoints/Delta.pth --output_dir export --verbose
```

---

### 2. Running the Mobile App (Flutter)

**Prerequisites:**
- [Flutter SDK](https://docs.flutter.dev/get-started/install) (3.x+)
- Physical Android/iOS device or simulator

```bash
# Navigate to the app directory
cd ../app

# Install Flutter packages
flutter pub get

# Launch the app
flutter run
```

---

## 🙏 Acknowledgements

- **Depth Anything V2**: We sincerely thank the authors of [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) for their foundational work and open-source models.
- **MobileViT**: Implementation based on [MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer](https://arxiv.org/abs/2110.02178).

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).