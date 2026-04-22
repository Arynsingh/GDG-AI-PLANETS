# AI Drug Discovery — Pix2Pix GAN

An AI-powered drug discovery tool that takes **molecular compound images** as input and generates new synthesized compound images using a **Pix2Pix GAN** (Generative Adversarial Network) built with TensorFlow.

---

## Overview

This project applies deep learning to accelerate drug discovery by learning the visual structure of known drug compounds and generating novel molecular variations. The model is trained on compound images for three drugs — **Amoxicillin**, **Atorvastatin**, and **Metformin** — and produces synthesized output images that can be used for further analysis.

---

## How It Works

```
Input: Drug compound image (256×256 PNG)
         ↓
   Pix2Pix Generator (CNN Encoder–Decoder)
         ↓
Output: Generated compound image saved to /output/<drug_class>/
```

1. **Input images** are loaded from `drug discovery/<class>/` folders (600 images per drug class).
2. The **Pix2Pix GAN** trains a generator and discriminator in tandem.
3. The generator learns to synthesize new compound images from the input distribution.
4. **Generated images** are saved to `output/output_<class>/` (30 images per class by default).

---

## Project Structure

```
├── drug discovery/
│   ├── amoxicillin/       # 600 input compound images
│   ├── atorvastatin/      # 600 input compound images
│   └── metformin/         # 600 input compound images
├── output/
│   ├── output_amoxicillin/
│   ├── output_atorvastatin/
│   └── output_metformin/
├── templates/
│   └── drugassistant.html  # Web UI
├── static/
│   └── xyzz.css            # Stylesheet
├── app.py                  # Flask web application
├── Untitled.ipynb          # Model training notebook
└── README.md
```

---

## Model Architecture

| Component       | Details                                      |
|----------------|----------------------------------------------|
| Framework       | TensorFlow / Keras                           |
| Architecture    | Pix2Pix GAN (Encoder–Decoder + Discriminator)|
| Input shape     | 256 × 256 × 3                                |
| Loss function   | Binary Crossentropy (from logits)            |
| Optimizer       | Adam (lr=2e-4, β₁=0.5)                       |
| Training epochs | 10                                           |
| Output          | Generated PNG images per drug class          |

---

## Drug Classes

| Drug           | Category              | Use Case                        |
|----------------|-----------------------|---------------------------------|
| Amoxicillin    | Antibiotic            | Bacterial infection treatment   |
| Atorvastatin   | Statin                | Cholesterol management          |
| Metformin      | Antidiabetic          | Type 2 diabetes management      |

---

## Getting Started

### Prerequisites

```bash
pip install tensorflow numpy matplotlib flask pillow
```

### Train the Model

Open and run `Untitled.ipynb` in Jupyter:

```bash
jupyter notebook Untitled.ipynb
```

Generated images will be saved to the `output/` directory.

### Run the Web App

```bash
python app.py
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

---

## Web Interface

The Flask web app allows you to:
- Upload a compound image
- Select a drug class (Amoxicillin, Atorvastatin, Metformin)
- View the AI-generated output compound image

---

## Results

After training for 10 epochs, the model achieves a generator loss of ~**0.033**, indicating stable convergence. Generated images are stored in `output/output_<class>/` and can be used for downstream molecular analysis.

---

## Tech Stack

- **Python 3.10**
- **TensorFlow 2.x** — model training
- **Flask** — web interface
- **NumPy / Matplotlib** — data processing & visualization
- **HTML / CSS** — frontend UI

---

## License

This project is intended for research and educational purposes in the field of AI-assisted pharmaceutical sciences.
