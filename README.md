# regression-using-pytorch-tutorial
A tutorial on building different regression models using the PyTorch modules 

# Weather Temperature Regression Using PyTorch

A machine learning project that implements neural network regression models to predict temperature based on weather features using PyTorch.

## 📊 Dataset

**Source**: [Weather Dataset from Kaggle](https://www.kaggle.com/code/krystianadammolenda/weather-dataset-cleaning-regression/notebook)

**Dataset Description**: 
This project uses a comprehensive weather dataset containing 8,784 hourly weather observations with multiple meteorological features.

### Features Used:
- **Dew Point Temperature (°C)**: Primary predictor variable
- **Relative Humidity (%)**: Secondary predictor variable

### Target Variable:
- **Temperature (°C)**: The variable we aim to predict

### Data Preprocessing:
- Removed non-numeric columns (`Date/Time`, `Weather`)
- Selected the two most correlated features based on exploratory data analysis
- Split data into 80% training and 20% testing sets

## 🔍 Exploratory Data Analysis

The project includes comprehensive EDA with:
- **2D scatter plots** analyzing individual feature relationships with temperature
- **3D visualizations** exploring combined feature effects
- **Correlation analysis** to identify the strongest predictors

Key findings:
- Dew Point Temperature shows the strongest linear relationship with Temperature (R ≈ 1.0)
- Relative Humidity has a moderate negative correlation with Temperature
- Other features (Wind Speed, Visibility, Pressure) show weaker relationships

## 🏗️ Model Architecture

### Model 1: ReLU Activation
```
Input (2) → Linear(20) → ReLU → Linear(10) → ReLU → Linear(1) → Output
```
- **Activation Function**: ReLU
- **Hidden Layers**: 2 layers (20, 10 neurons)
- **Optimizer**: Stochastic Gradient Descent (SGD)
- **Learning Rate**: 0.0001

### Model 2: Sigmoid Activation
```
Input (2) → Linear(15) → Sigmoid → Linear(8) → Sigmoid → Linear(1) → Output
```
- **Activation Function**: Sigmoid
- **Hidden Layers**: 2 layers (15, 8 neurons)
- **Optimizer**: Adam
- **Learning Rate**: 0.001

## 📈 Results

Both models achieved exceptional performance:

| Metric | Model 1 (ReLU) | Model 2 (Sigmoid) |
|--------|----------------|-------------------|
| **MAE** | 0.1432 | 0.1432 |
| **MSE** | 0.0434 | 0.0434 |
| **RMSE** | 0.2082 | 0.2082 |
| **R²** | 0.9997 | 0.9997 |

## 🚀 Getting Started

### Prerequisites
```bash
pip install torch torchvision
pip install pandas numpy matplotlib
pip install scikit-learn
```

### Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/weather-temperature-regression.git
cd weather-temperature-regression
```

2. Download the dataset:
   - Place `Weather Data.csv` in the project directory
   - Or download from the Kaggle link provided above

3. Run the Jupyter notebook:
```bash
Jupyter Notebook regression-with-pythrch.ipynb
```


## 🔧 Usage

The notebook is structured in the following sections:

1. **Data Loading and Exploration**
2. **Exploratory Data Analysis with Visualizations**
3. **Data Preprocessing and Feature Selection**
4. **Model 1 Implementation (ReLU)**
5. **Model 2 Implementation (Sigmoid)**
6. **Performance Evaluation and Comparison**
7. **Analysis and Recommendations**

## 📊 Key Insights

- **Strong Linear Relationship**: Dew point temperature is an excellent predictor of air temperature
- **Model Performance**: Both activation functions (ReLU and Sigmoid) performed equally well for this regression task
- **Feature Engineering**: Careful feature selection based on correlation analysis significantly improved model performance
- **Generalization**: High R² score (99.97%) indicates excellent model fit without apparent overfitting

## 🤖 Technical Implementation

### Neural Network Features:
- **PyTorch Framework**: Built using PyTorch's nn.Module
- **Custom Model Classes**: Implemented two distinct architectures
- **Multiple Optimizers**: Compared SGD vs Adam optimization
- **Comprehensive Metrics**: MAE, MSE, RMSE, and R² evaluation

### Training Configuration:
- **Epochs**: 100,000 for both models
- **Loss Function**: Mean Squared Error (MSE)
- **Data Split**: 80/20 train-test split with random_state=42

## 📝 Recommendations

Based on the analysis:

1. **Model Choice**: Either model is suitable for this task, given identical performance
2. **Activation Function**: ReLU might be preferred for scalability to larger networks
3. **Feature Selection**: Dew point temperature alone could potentially achieve similar results
4. **Production Deployment**: Consider the ReLU model for computational efficiency

## 🎓 Academic Context

This project was completed as part of:
- **Course**: ANLY-6500: Advanced Data Analytics
- **Module**: 3 - Regression Using PyTorch

## 📧 Contact

**Author**: Ebube Ndubuisi

For questions or collaborations, please open an issue or submit a pull request.

## 📄 License

This project is available under the MIT License. See LICENSE file for details.

---

*This project demonstrates the application of deep learning techniques for weather prediction, showcasing the effectiveness of neural networks in capturing meteorological relationships.*
