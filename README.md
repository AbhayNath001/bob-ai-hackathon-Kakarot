# 🚀 AI-Driven Selective MMP-12 Inhibitor Prediction and Structure-Aware Binding Analysis

> An integrated deep-learning framework for chemical-space-guided MMP-12 activity and selectivity prediction, ligand-conditioned 3D protein structure prediction, and ligand-aware binding affinity scoring.

---

## 👥 Team

| Field         | Value                                                                      |
| ------------- | -------------------------------------------------------------------------- |
| **Team Name** | Kakarot                                                                    |
| **Track**     | AI | Drug Discovery | Chemoinformatics                                     |
| **Team Lead** | Abhay Nath — [26pgce013@charusat.edu.in](mailto:26pgce013@charusat.edu.in) |
| **Members**   | Viren Patel, Zalak Patel, Bhavan Patel                                     |

---

## 🎯 Problem Statement

Selective MMP-12 inhibitor prediction is challenging because closely related MMP isoforms share highly similar catalytic architectures, while molecular activity, selectivity, protein conformation, and ligand binding are strongly interdependent. Conventional ligand-based and structure-based workflows generally address activity prediction, protein structure prediction, docking, and affinity estimation as separate tasks, making it difficult to connect chemical-space-guided candidate selection with ligand-aware structural assessment.

---

## 💡 Solution

We developed an integrated AI pipeline that connects **molecular prediction with structure-aware binding analysis**. The system uses the **MSR-ARN (Multi-Scale Residual Attentive Regression Network)** to learn molecular patterns and predict MMP-12 inhibitory activity, together with selectivity modelling against related MMP isoforms.

Promising candidates are subsequently evaluated through a **ligand-conditioned diffusion-transformer structure-prediction module**, which predicts a 3D protein conformation using ligand information and SE(3)-equivariant coordinate refinement. The resulting protein structure is combined with the ligand and passed to an **SE(3)-equivariant docking and affinity module** for iterative pose refinement and ligand-aware binding-energy prediction.

The complete workflow therefore connects:

```text
Chemical Space
      ↓
MMP-12 Activity Prediction
      ↓
MMP-12 Selectivity Prediction
      ↓
Candidate Prioritization
      ↓
Ligand-Conditioned 3D Protein Structure
      ↓
SE(3)-Equivariant Docking
      ↓
Binding Affinity Prediction
      ↓
Final Candidate Ranking
```

---

## ✨ Key Features

* **Chemical-Space-Guided Activity Prediction:** MSR-ARN learns nonlinear relationships from molecular representations to predict MMP-12 inhibitory activity.

* **MMP-12 Selectivity Prediction:** Candidates are evaluated against related MMP isoforms, particularly MMP-2 and MMP-13, to identify molecules with predicted MMP-12 selectivity.

* **Ligand-Conditioned 3D Structure Prediction:** Protein sequence and ligand information are jointly processed through transformer-based pair representations and diffusion-style coordinate refinement.

* **SE(3)-Equivariant Docking and Affinity:** The system performs iterative ligand-pose refinement while predicting ligand-aware binding energy from three-dimensional protein-ligand geometry.

* **Explainable Candidate Prioritization:** Molecular predictions can be interpreted using feature-level analysis, while predicted protein-ligand complexes can be inspected through structural visualization.

---

## 🛠️ Tech Stack

| Category             | Technologies                                                    |
| -------------------- | --------------------------------------------------------------- |
| **Languages**        | Python                                                          |
| **Frameworks**       | TensorFlow, Keras, PyTorch, Scikit-learn                        |
| **IBM Technologies** | IBM Bob                                                         |
| **Databases**        | Not required for the current core prototype                     |
| **Other**            | RDKit, Mordred, DeepChem, SHAP, NumPy, Pandas, NetworkX, GitHub |

---

## 📁 Repository Structure

```text
├── src/                              # Core source code
│   ├── data/                         # Data preparation and preprocessing
│   ├── models/
│   │   ├── msr_arn/                  # MSR-ARN molecular prediction
│   │   ├── protein_structure/        # Attention, Pairformer and SE(3) modules
│   │   └── docking_affinity/         # Docking and affinity prediction
│   └── .env.example                  # Environment configuration template
│
├── docs/                             # Project documentation
│   ├── problem-statement.md
│   ├── solution-overview.md
│   ├── architecture.md
│   └── setup-guide.md
│
├── demo/                             # Demonstration artifacts
│   ├── screenshots/
│   ├── demo-video-link.txt
│   └── live-demo-url.txt
│
├── presentation/                     # Hackathon presentation
│
├── submission.yaml                   # Structured hackathon metadata
├── README.md                         # Project documentation
└── .gitignore
```

The model layer is explicitly separated into **MSR-ARN**, **protein-structure**, and **docking-affinity** components in the current repository.

---

## ⚡ How to Run

> **The detailed setup instructions are available in [`docs/setup-guide.md`](docs/setup-guide.md).**

### 1. Clone the repository

```bash
git clone https://github.com/AbhayNath001/bob-ai-hackathon-Kakarot.git
cd bob-ai-hackathon-Kakarot
```

### 2. Create a Python virtual environment

```bash
python -m venv venv
```

Activate it:

```bash
# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment

The repository provides an environment template at:

```text
src/.env.example
```

Copy it to your local environment configuration and update the required paths/settings.

### 5. Run the relevant model pipeline

For the MSR-ARN model:

```bash
python src/models/msr_arn/model.py
```

For the integrated protein-structure and binding workflow:

```bash
python src/models/docking_affinity/complete_sample.py
```

> Use the exact entry-point filenames present in the current `src/` tree when running the latest repository version.

### 6. Run the complete workflow

The intended inference sequence is:

```text
Ligand SMILES / Molecular Input
          ↓
Chemical Preprocessing
          ↓
MSR-ARN
          ↓
MMP-12 Activity + Selectivity
          ↓
Candidate Selection
          ↓
Protein Sequence + Ligand
          ↓
Diffusion-Transformer Structure Prediction
          ↓
Ligand-Conditioned 3D Protein
          ↓
SE(3)-Equivariant Docking
          ↓
Binding Affinity
          ↓
Final Ranking
```

---

## 🔬 Model Pipeline

### MSR-ARN

The MSR-ARN component is designed for molecular activity prediction and contains:

```text
Molecular Features
       ↓
Multi-Scale Conv1D
       ↓
Residual Feature Learning
       ↓
Squeeze-and-Excitation Attention
       ↓
Dilated Gated Convolutions
       ↓
Multi-Head Self-Attention
       ↓
Global Feature Aggregation
       ↓
FiLM Feature Modulation
       ↓
Regression Head
       ↓
Predicted MMP-12 Activity
```

The current implementation includes preprocessing, feature columns, and a fitted feature scaler for consistency with the trained molecular representation.

### Ligand-Aware Protein Structure Prediction

The structure-prediction branch combines:

```text
Protein Sequence
       +
Ligand Representation
       ↓
Sequence Embedding
       ↓
Pairformer
       ↓
Ligand Cross-Attention
       ↓
Diffusion-Style Structure Head
       ↓
SE(3)-Equivariant Coordinate Refinement
       ↓
Ligand-Conditioned 3D Protein Structure
```

The repository contains dedicated attention and pairformer modules together with an SE(3) implementation and supporting utilities.

### Docking and Binding Affinity

The structure-aware binding stage performs:

```text
Predicted Protein Structure
          +
Initial Ligand Pose
          ↓
SE(3)-Equivariant Graph Network
          ↓
Iterative Message Passing
          ↓
Pose Refinement
          ↓
Rigid-Body + Torsional Updates
          ↓
Affinity Prediction Head
          ↓
Predicted Binding Energy
```

The implemented docking pipeline uses geometric graph processing and iterative coordinate refinement for ligand-aware affinity prediction.

---

## 🖥️ Demo

| Artifact        | Link                                                       |
| --------------- | ---------------------------------------------------------- |
| 🖼️ Screenshots | [See `demo/screenshots/`](demo/screenshots/)               |
| 📊 Presentation | [See `presentation/`](presentation/)                       |
| 📹 Demo Video | [See `demo/demo-video-link.txt`](demo/demo-video-link.txt)                               |

---

## ⚠️ Known Limitations

> This project is intended as an **in-silico computational prediction framework** and should not be interpreted as experimental confirmation.

* The MMP-12 activity and selectivity datasets are moderate in size and may not represent the full chemical diversity of potential inhibitor scaffolds.
* Predictions for substantially novel chemical structures require additional external validation.
* Protein structure prediction, iterative SE(3)-equivariant refinement, and affinity inference can require substantial GPU computational resources.
* Binding-affinity and selectivity predictions are computational estimates and require biochemical and structural experiments for prospective validation.
* The current hackathon implementation focuses on the AI modelling pipeline rather than a complete production-grade pharmaceutical deployment.

---

## 🏅 What We're Most Proud Of

Our strongest contribution is the **integration of complementary deep-learning approaches into one drug-discovery workflow**.

Rather than stopping at molecular activity prediction, the system connects **chemical-space-guided MMP-12 prediction with ligand-aware structural modelling and binding-affinity estimation**. MSR-ARN provides the molecular activity/selectivity layer, while the diffusion-transformer and SE(3)-equivariant components extend the analysis into three-dimensional protein-ligand space.

This creates a complete computational path from:

```text
Molecular Structure
        ↓
Activity
        ↓
Selectivity
        ↓
3D Protein Structure
        ↓
Ligand Pose
        ↓
Binding Affinity
        ↓
Candidate Prioritization
```

The architecture is also intentionally modular: the molecular, protein-structure, and docking-affinity components are maintained as separate model modules, making the system easier to evaluate, improve, and extend.
