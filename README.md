# Part 1 Neural Network Analysis: Customer Churn Prediction

This repository contains a full analysis and feed-forward neural network pipeline to predict whether a customer is likely to churn using the `customer_churn_nn.csv` dataset.

## Repository Structure
- `notebook.ipynb`: Core Python notebook executing the exploration, preprocessing, modeling, and experimentation.
- `requirements.txt`: Python package dependency list.
- `results/`:
  - `model_comparison_table.csv` / `.png`: Results across various architectures and hyperparameter choices.
  - `evaluation_outputs.png`: Plot of training/validation curves and confusion matrix.

## Key Insights & Hyperparameter Performance
Four architectures were tested:
1. **Baseline**: 1 Hidden Layer (16 units, ReLU, lr=0.001) - Caught 66.7% of churners but with low precision due to major class imbalance.
2. **Exp 1 (Deeper)**: 2 Hidden Layers (32, 16 units) - Improved Test F1 score dramatically to 0.285 and reached 96.25% test accuracy.
3. **Exp 2 (High LR)**: 1 Hidden Layer (lr=0.05) - Unstable updates led to suboptimal performance and lower AUC (0.735).
4. **Exp 3 (Tanh)**: 1 Hidden Layer (Tanh activation) - Captured 83.3% of churners with a high ROC AUC of 0.903.

## Reflection Responses
- **Weights and Biases**: Weights represent the strength of connections between features and neurons, controlling the slope of the activation. Biases shift the activation function horizontally, determining how high the input signal needs to be to fire a neuron.
- **Activation Functions**: Introduce non-linearity into the network, allowing it to learn complex patterns and decision boundaries rather than acting as a simple linear regression model.
- **Learning Rate Limits**: If too high, the optimizer overshoots the minimum, causing oscillation or divergence. If too low, training is extremely slow and can get stuck in local minima or saddle points.
- **Underfitting/Overfitting**: The baseline model showed some underfitting on the minority class due to the severe class imbalance (1.55% churners). The deeper model adjusted well without severe overfitting as training and validation losses remained closely aligned.
