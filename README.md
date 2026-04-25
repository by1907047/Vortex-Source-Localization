# High-sensitive flexible dual-modal flow sensing array for accurate vortex source localization

This repository contains data, codes, and models for vortex source localization using advanced dual-modal flow sensing technology. The project demonstrates high-sensitivity flow detection capabilities across multiple experimental scenarios with support for velocity (V), pressure (P), and combined (V+P) sensing modalities.

## Cylinder Localization

This section addresses vortex source localization for cylinder-generated flows. The cylinder acts as the vortex source while the sensor array measures the resulting flow field.

### 1. Training and Test Datasets
- Comprehensive datasets collected from cylinder flow experiments
- Data organized for velocity (V), pressure (P), and combined (V+P) sensing modalities
- Separated into training and test subsets for model validation
- Preprocessed and normalized for optimal model performance

### 2. MLP Model Training Code
- Multi-layer perceptron implementation for vortex source localization
- Training scripts supporting all three sensing modalities (V, P, V+P)
- Loss functions and optimization algorithms configured for high accuracy
- Hyperparameter configurations and training procedures documented

### 3. Pretrained Models and Test Results
- Pre-trained MLP models for each sensing modality:
  - **V (Velocity-only)**: Model trained exclusively on velocity field data
  - **P (Pressure-only)**: Model trained exclusively on pressure field data
  - **V+P (Combined)**: Model trained on fused velocity and pressure data
- Comprehensive test results including accuracy metrics, error analysis, and performance comparisons across modalities
- Model weights and prediction outputs available for reproducibility

## Fixed Robotic Fish Localization

This section addresses vortex source localization for stationary robotic fish. The robotic fish performs tail oscillations to generate vortex sources while maintaining a fixed position.

### 1. Training and Test Datasets
- Experimental data collected from fixed robotic fish tail oscillation experiments
- Data organized for velocity (V), pressure (P), and combined (V+P) sensing modalities
- Training and test subsets with balanced representation of fish oscillation patterns
- Preprocessing includes signal normalization and feature alignment

### 2. MLP Model Training Code
- Multi-layer perceptron architecture optimized for fixed fish scenarios
- Training implementations for all three sensing modalities (V, P, V+P)
- Specialized loss functions accounting for tail oscillation dynamics
- Complete training pipelines with validation procedures

### 3. Pretrained Models and Test Results
- Pre-trained MLP models for each sensing modality:
  - **V (Velocity-only)**: Localization based on velocity field measurements
  - **P (Pressure-only)**: Localization based on pressure field measurements
  - **V+P (Combined)**: Multi-modal fusion for enhanced localization accuracy
- Detailed test results showing localization accuracy, spatial error distributions, and modality-specific performance metrics
- Model checkpoints and inference results for benchmark comparisons

## Swimming Robotic Fish Localization

This section addresses the more challenging problem of localizing actively swimming robotic fish in a water tank. The fish generates dynamic vortex sources while moving through the fluid.

### 1. Time-Domain Signal Collection
- Multi-run acquisition of temporal signals from swimming robotic fish experiments
- Multiple repeated trials capturing diverse swimming behaviors and trajectories
- Raw time-domain measurements from the dual-modal sensor array
- Complete datasets with synchronized velocity and pressure measurements

### 2. Short-Time Fourier Transform Feature Extraction and Dataset Splitting
- Short-time Fourier transform (STFT) feature extraction from raw time-domain signals
- Frequency-domain feature representation for capturing dynamic characteristics
- Extracted features organized into training and test subsets
- Proper data splitting ensuring temporal consistency and avoiding temporal leakage

### 3. Bi-GRU Model Training Code
- Bidirectional Gated Recurrent Unit (Bi-GRU) architecture for sequential data processing
- Implementation designed for time-series analysis of swimming fish localization
- Training scripts with recurrent network optimization
- Sequence-to-sequence modeling capturing temporal dependencies in fish movement

### 4. Pretrained Models and Test Results
- Pre-trained Bi-GRU models for swimming fish localization
- Combined model evaluation results integrating all training runs
- Comprehensive test performance metrics including:
  - Trajectory prediction accuracy
  - Real-time localization error rates
  - Temporal consistency measures
  - Robustness across different swimming patterns
- Model checkpoints and benchmark results for dynamic source localization

---

**Key Features:**
- High-sensitivity dual-modal sensing (velocity and pressure)
- Flexible sensor array design compatible with multiple scenarios
- Support for multiple sensing modalities: V, P, and V+P
- Comprehensive model implementations: MLP for static scenes, Bi-GRU for dynamic scenes
- Accurate vortex source position estimation across diverse experimental conditions
- Full reproducibility with pre-trained models and detailed documentation

For questions or contributions, please refer to the individual folders and accompanying documentation.