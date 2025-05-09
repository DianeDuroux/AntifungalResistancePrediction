# 📄 Early antifungal resistance prediction using MALDI-TOF mass spectrometry and machine learning

## Overview

This project implements a machine learning pipeline for early antifungal resistance prediction using MALDI-TOF mass spectrometry spectra and drug features.

It includes:

- Data preprocessing  
- Nested cross-validation for model comparison  
- Model training and evaluation
- Generation of performance metrics and visualizations 
- Feature importance analysis  

---

## ⚙️ Installation

Clone this repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

## 📁 Data
The MALDI-TOF spectra used in this project can be obtained by downloading the DRIAMS dataset from:

🔗 https://datadryad.org/stash/dataset/doi:10.5061/dryad.bzkh1899q

The data/ folder in this repository contains only the fungal subset of the DRIAMS dataset.

## 🚀 How to Run
Open the main.ipynb notebook.

Set the correct path for the working_dir variable according to your environment.

Run the notebook cells step-by-step or convert the notebook to a Python script.

## 📊 Output
The pipeline generates the following outputs:

- Final prediction results stored in the `results/` directory

- Performance metrics for model evaluation in the `results/` directory

- Visualizations saved in the `plot/` directory


## Funding
This research was primarily supported by the ETH AI Center.
