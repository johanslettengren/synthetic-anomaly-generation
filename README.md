## Synthetic Data Generation for Anomaly Detection in Water Pipe Networks

This project is the result of a research internship at the Division of Decision and Control Systems at KTH, Digital Futures program throughout the summer of 2025. The results are presented in this [poster](SyntheticDataGeneration.pdf).

### Synopsis
This project explored the use of **physics-informed neural networks (PINNs)** to accelerate simulations of water pipe networks, which are typically performed using EPANET. The key focus was on generating large volumes of **synthetic data for leak scenarios** via amortized prediction, enabling scalable anomaly detection in settings where real leaks are rare. By embedding governing hydraulic equations into the learning process, the PINN model generalized across multiple leak scenarios simultaneously, achieving accurate flow predictions while substantially reducing computational costs.

### Key Results
- PINN-based amortized data generation produced accurate predictions (NRMSD ≤ 10%) across leak scenarios.  
- Showed strong scalability compared to traditional one-simulation-per-scenario methods.  
- Demonstrated reduced runtime for large parameter sweeps (e.g., PINN outperformed simulator at higher grid resolutions and number of demand junctions).
