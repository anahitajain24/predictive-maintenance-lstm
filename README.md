# NASA CMAPSS RUL Prediction with Bi-LSTM + Attention

### 🔍 Deep Learning for Predictive Maintenance

A PyTorch implementation of a Remaining Useful Life (RUL) estimator for turbofan engines (FD001). 
Unlike standard regression models, this architecture utilizes a **Bi-Directional LSTM** coupled with an **Attention Mechanism** to weigh specific time-steps, providing both high accuracy and model interpretability.

![Attention Heatmap](heatmap.png)
*(Above: Attention weights highlighting critical sensor fluctuations prior to failure)*

## 🧠 Model Architecture
* **Input:** Rolling window sequences (Window Size: 50) of 14 sensor readings.
* **Core:** 2-Layer Bi-Directional LSTM (Hidden Dim: 64).
* **Head:** Dot-Product Attention Layer $\to$ Fully Connected MLP.
* **Optimization:** Adam Optimizer with `ReduceLROnPlateau` and Gradient Clipping.

## 📊 Performance
* **Dataset:** NASA CMAPSS FD001
* **Metric:** RMSE (Root Mean Square Error)
* **Results:** Achieved stable convergence with explainable attention maps.

## 🛠 Dependencies
```bash
pip install torch pandas numpy scikit-learn matplotlib seaborn
