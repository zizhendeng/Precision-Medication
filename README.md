# Temporal Causal Precision Medication (TCPM)

A generalist framework for translating routine physiological indicators into personalized treatment strategies using temporal causal inference.

## Overview

TCPM (Temporal Causal Precision Medication) is a novel framework designed to address treatment response heterogeneity in clinical practice by leveraging temporal causal inference and treatment-free physiological profiles. This repository contains the implementation of the TCPM framework as described in our paper:

**A generalist precision medication framework using temporal causal inference based on treatment-free physiological profiles**  

TCPM integrates counterfactual prediction, adversarial learning, and optimal search algorithms to derive personalized longitudinal treatment strategies across various acute and chronic diseases.

## Key Features

- **Treatment-Free Physiological Profile (TFPP)**: A novel representation of heterogeneous physiological states that eliminates confounding bias through adversarial training
- **Counterfactual Prediction**: Accurate prediction of patient outcomes under alternative treatment strategies
- **A*-Powered Optimal Search**: Derives optimal longitudinal treatment sequences balancing immediate and long-term outcomes
- **Heterogeneity Analysis**: Identifies patient subgroups with divergent response patterns and reveals underlying physiological mechanisms
- **Broad Applicability**: Validated across 6 disease cohorts (sepsis, diabetes, hypertension, coronary heart disease, macular degeneration, pediatric burns)

## Framework Architecture

![TCPM Framework](https://example.com/tcpm_framework.png)  
*Figure 1: Overview of the TCPM framework analysis workflow*

1. **Treatment-Free Physiological Profile Learning**: Encoder-decoder architecture with adversarial training to eliminate treatment confounding
2. **Treatment Strategy Optimization**: A*-based search algorithm to identify optimal personalized treatment sequences
3. **Subgroup Identification**: Stratifies patients into subgroups with distinct therapeutic response patterns
4. **Physiological Network Analysis**: Constructs correlation and causal graphs to reveal mechanistic differences between subgroups

## Installation

```bash
# Clone the repository
git clone https://github.com/zizhendeng/Precision-Medication.git
cd Precision-Medication

# Install required dependencies
pip install -r requirements.txt
```

## Usage

### Data Preparation

The framework supports multiple disease cohorts. Example data loaders for public datasets (MIMIC-III derived) and private datasets are provided in `LoadData.py`.

```python
# Example: Loading sepsis dataset
from LoadData import MIMICDataLoader

# Initialize data loader
data_loader = MIMICDataLoader(dataset_type='sepsis')

# Load and preprocess data
train_data, val_data, test_data = data_loader.load_and_preprocess()
```

### Model Training

Train the TFPP (Treatment-Free Physiological Profile) encoder:

```python
from models.CRN_model import TFPPEncoder

# Initialize model
tfpp_encoder = TFPPEncoder(input_dim=39, hidden_dim=64, treatment_dim=2)

# Train the model
tfpp_encoder.train(
    train_data=train_data,
    val_data=val_data,
    epochs=50,
    batch_size=32,
    learning_rate=1e-3
)
```

### Counterfactual Prediction

```python
from models.CRN_Lightning_model import CounterfactualPredictor

# Initialize predictor
predictor = CounterfactualPredictor(tfpp_encoder=tfpp_encoder)

# Load trained model
predictor.load_weights('models/pretrained/tfpp_sepsis.pth')

# Predict outcomes under alternative treatments
patient_data = test_data[0]  # Example patient data
predicted_outcomes = predictor.predict_counterfactual(
    patient_data=patient_data,
    treatment_strategies=[[0, 0], [1, 0], [0, 1], [1, 1]]  # Example treatment sequences
)
```

### Optimal Treatment Search

```python
from AStarSearch import OptimalTreatmentSearch

# Initialize search algorithm
search_algorithm = OptimalTreatmentSearch(predictor=predictor)

# Find optimal treatment sequence
optimal_treatment = search_algorithm.find_optimal(
    patient_data=patient_data,
    horizon=4  # Predict 4 time steps ahead
)
```

### Subgroup Analysis

```python
from utils.subgroup_analysis import SubgroupAnalyzer

# Initialize analyzer
analyzer = SubgroupAnalyzer(tfpp_encoder=tfpp_encoder)

# Identify subgroups
subgroups = analyzer.identify_subgroups(
    data=test_data,
    n_subgroups=4
)

# Generate physiological networks
correlation_networks = analyzer.construct_correlation_networks(subgroups)
causal_graphs = analyzer.construct_causal_graphs(subgroups)
```

## Results

TCPM outperformed standard clinical protocols and state-of-the-art reinforcement learning methods across multiple metrics:

- Superior therapeutic efficacy across 6 disease cohorts
- Accurate counterfactual predictions (PCC=0.73-0.86 across cohorts)
- Clinically meaningful patient subgroup stratification
- Mechanistic insights through physiological network analysis

Detailed results can be found in our paper and supplementary materials.

## Directory Structure

```
├── data/                 # Data processing utilities
├── models/               # Model architectures
│   ├── CRN_model.py      # Causal recurrent network for TFPP
│   └── CRN_Lightning_model.py # PyTorch Lightning implementation
├── utils/                # Utility functions
│   ├── subgroup_analysis.py
│   └── network_construction.py
├── AStarSearch.py        # Optimal treatment search implementation
├── LoadData.py           # Data loading and preprocessing
├── SearchAdapter.py      # Adapter for search algorithms
├── SearchApp.py          # Application for treatment search
└── test.py               # Testing scripts
```

## Citation

If you use this framework in your research, please cite our paper:

```bibtex
@article{tcpmpaper,
    title={A generalist precision medication framework using temporal causal inference based on treatment-free physiological profiles},
    author={Deng, Zizhen and Wu, Wei and Zhang, Chi and Zhao, Xitong and Pu, Minghao and Bu, Yanbin and Liao, Yanfeng and Wang, Changguan and Yang, Jiarui and Wang, Yanni and Wang, Jinzhuo},
    journal={待发表},
    year={2025}
}
```

## Contact

For questions or issues, please contact:

- Jinzhuo Wang: wangjinzhuo@pku.edu.cn
- Zizhen Deng: [corresponding email]

## License

This project is licensed under the MIT License - see the LICENSE file for details.
