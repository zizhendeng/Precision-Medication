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

![TCPM Framework](fig_1.png)  
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

# Use conda to create a new environment according to environment.yaml
conda env create -f environment.yaml
```

## Usage

### Data Preparation

The framework supports multiple disease cohorts. Example data loaders for public datasets (MIMIC-III derived) and private datasets are provided in `LoadData.py`.

```python
# Example: Loading sepsis dataset
csv_data, headers = LoadData.load('./data/disease_data.csv')

# Initialize data loader
encoder_data = LoadData.getEncoderData(csv_data, headers,
                                        LoadData.getMaxSequenceLength(csv_data),
                                        LoadData.getMaxTreatmentNumber(csv_data, headers))

### Model Training

```python

en_lit_crn = LitCRN(params, best_hyperparams, False, learning_rate=0.01)
en_trainer = L.Trainer(max_epochs=30, deterministic=True, devices=devices)

de_lit_model = LitCRN(params, best_hyperparams, True, 0.01)
de_trainer = L.Trainer(max_epochs=30,  deterministic=True, devices=devices)

```


### Optimal Treatment Search

```python

# Find optimal treatment sequence
 results = SearchApp.getResult(predictor, en_lit_crn, L.Trainer(logger=False, deterministic=True,  devices=devices), de_lit_model,
                                    [csv_data[index]], headers, num_treatments, horizons,
                                    best_hyperparams['rnn_hidden_units'],
                                    lip_const_K, 1.0, baseline_steps, 10)
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
  
## License

This project is licensed under the MIT License - see the LICENSE file for details.
