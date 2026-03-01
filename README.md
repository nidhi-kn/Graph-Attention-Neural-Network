# Graph Attention Neural Network for FDI Attack Detection

This project implements and compares various graph neural network architectures for detecting False Data Injection (FDI) attacks in power grid systems. The main focus is on the Enhanced Graph Attention Network (ENGAT) and Graphon-based Weighted Neural Network (GraphonWNN) models.

## 🚀 Key Features

- **Enhanced Graph Attention Network (ENGAT)**: Advanced GAT architecture with multi-head attention and enhanced feature engineering
- **Graphon-based Weighted Neural Network (GraphonWNN)**: Novel approach using graphon theory for graph neural networks
- **Comprehensive Baseline Comparison**: Comparison with GCN, GAT, GraphSAGE, and MLP models
- **Robust Evaluation**: Multiple metrics including balanced accuracy, recall, and ROC AUC
- **Hyperparameter Optimization**: Genetic algorithm and PSO-based feature selection

## 📊 Performance Results

Based on extensive evaluation using the `tune_graphon_data1_big.csv` dataset:

| Model | Balanced Acc | Normal Recall | Attack Recall | ROC AUC | Train Time (ms) |
|-------|-------------|---------------|---------------|---------|-----------------|
| **ENGAT** | 0.886 ± 0.009 | 0.865 ± 0.026 | 0.907 ± 0.017 | 0.935 ± 0.008 | 404286.3 ± 61550.8 |
| GraphSAGE | 0.756 ± 0.019 | 0.784 ± 0.064 | 0.728 ± 0.062 | 0.836 ± 0.021 | 9800.9 ± 2156.8 |
| MLP | 0.774 ± 0.040 | 0.801 ± 0.048 | 0.747 ± 0.097 | 0.839 ± 0.043 | 4060.3 ± 1795.1 |
| GCN | 0.740 ± 0.013 | 0.845 ± 0.093 | 0.635 ± 0.091 | 0.806 ± 0.015 | 8731.0 ± 2102.3 |
| GAT | 0.723 ± 0.023 | 0.837 ± 0.084 | 0.608 ± 0.074 | 0.780 ± 0.018 | 15189.6 ± 3282.4 |

**Key Finding**: ENGAT significantly outperforms all baseline models with a balanced accuracy of **88.6%** and attack recall of **90.7%**.

## 🏗️ Project Structure

```
├── baseline_compare.py    # Main comparison script
├── enhanced_wnn.py        # ENGAT and GraphonWNN implementations
├── tune_graphon_data1_big.csv  # Performance results
└── README.md             # This file
```

## �️ Installation

Install the required dependencies:

```bash
pip install torch torch-geometric scikit-learn pandas numpy
```

## 🚀 Usage

### Basic Comparison

Run the baseline comparison with default settings:

```bash
python baseline_compare.py
```

### Advanced Options

```bash
# Quick run for testing
python baseline_compare.py --quick

# Full feature selection with GA and PSO
python baseline_compare.py --full-wnn

# Use GraphonWNN variant
python baseline_compare.py --wnn-variant graphon

# Custom hyperparameters
python baseline_compare.py --wnn-hidden-dim 512 --wnn-epochs 50 --wnn-lr 0.001

# Multiple seeds for robust evaluation
python baseline_compare.py --seeds 42 43 44 45 46 47

# Transfer learning with different test data
python baseline_compare.py --train-data data1.csv --test-data data2.csv
```

## � Model Architectures

### Enhanced Graph Attention Network (ENGAT)
- Multi-head attention mechanism (8 heads)
- Enhanced feature engineering with genetic algorithm optimization
- Label smoothing and weight decay regularization
- Adaptive learning rate scheduling

### Graphon-based Weighted Neural Network (GraphonWNN)
- Novel graphon-based approach for graph neural networks
- Limit-based graphon approximation
- Multi-layer architecture with residual connections

### Baseline Models
- **GCN**: Graph Convolutional Network
- **GAT**: Graph Attention Network
- **GraphSAGE**: Graph Sample and Aggregated
- **MLP**: Multi-Layer Perceptron

## 📊 Evaluation Metrics

- **Balanced Accuracy**: Accounts for class imbalance
- **Normal Recall**: True positive rate for normal operations
- **Attack Recall**: True positive rate for FDI attacks
- **ROC AUC**: Area under the ROC curve
- **Training Time**: Computational efficiency measure

## � Technical Details

### Data Preprocessing
- Stratified train/validation/test split (60/20/20)
- StandardScaler for feature normalization
- Weighted random sampling for class balance
- Graph construction from power grid topology

### Training Configuration
- **Optimizer**: AdamW with weight decay
- **Loss Function**: Cross-entropy with class weights
- **Early Stopping**: Patience-based validation monitoring
- **Gradient Clipping**: Max norm of 1.0

### Hyperparameter Optimization
- **Genetic Algorithm**: Population size 30, 15 generations
- **Particle Swarm Optimization**: For feature selection
- **Mutation Rate**: 0.1 for genetic operations

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{graph_attention_neural_network,
  title={Graph Attention Neural Network for FDI Attack Detection},
  author={Nidhi K N},
  year={2025},
  url={https://github.com/nidhi-kn/Graph-Attention-Neural-Network}
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## � License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔍 Future Work

- [ ] Integration with real-time power grid monitoring systems
- [ ] Extension to other types of cyber-physical attacks
- [ ] Distributed training for large-scale power grids
- [ ] Explainable AI for attack interpretation
- [ ] Federated learning for multi-grid deployment

## 📞 Contact

Nidhi K N - [GitHub Profile](https://github.com/nidhi-kn)

---

**Note**: This project is part of ongoing research in cyber-physical security for smart grid systems. The results demonstrate significant improvements in FDI attack detection accuracy compared to existing methods.