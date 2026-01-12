
# Iris Flower Classification 🌸

A machine learning project that classifies iris flowers into three species (Setosa, Versicolor, and Virginica) based on their physical features using multiple classification algorithms.

**AICTE OASIS INFOBYTE - Data Science Internship Task 1**

## 📋 Project Overview

This project implements and compares three machine learning algorithms to classify iris flowers based on four features: sepal length, sepal width, petal length, and petal width. The best-performing model achieves **93.33% accuracy** using Support Vector Machine (SVM).

## 🎯 Features

- Data loading and preprocessing from the Iris dataset
- Exploratory Data Analysis (EDA) with visualizations
- Feature scaling using StandardScaler
- Implementation of three classification algorithms:
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Random Forest Classifier
- 5-fold cross-validation for model evaluation
- Detailed performance metrics and confusion matrix visualization
- Model comparison and selection

## 🛠️ Technologies Used

- **Python 3.x**
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computations
- **matplotlib** - Data visualization
- **seaborn** - Statistical data visualization
- **scikit-learn** - Machine learning algorithms and tools

## 📦 Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd iris-flower-classification
Install required dependencies:

bash
pip install pandas numpy matplotlib seaborn scikit-learn
🚀 Usage
Run the Python script:

bash
python iris_flower_classification.py
The script will:

Load and explore the Iris dataset

Visualize feature distributions

Train three different classification models

Perform cross-validation

Generate performance reports and visualizations

Save visualizations as PNG files

📊 Dataset
The project uses the famous Iris Dataset from scikit-learn, which contains:

150 samples (50 per species)

4 features: sepal length, sepal width, petal length, petal width

3 classes: Setosa, Versicolor, Virginica

Data Split:

Training set: 70% (105 samples)

Testing set: 30% (45 samples)

🎯 Model Performance
Model	Test Accuracy	Cross-Validation Accuracy
Support Vector Machine (SVM)	93.33%	Best performer
Logistic Regression	High accuracy	Competitive
Random Forest Classifier	High accuracy	Competitive
The SVM model with RBF kernel was selected as the best performer based on both test accuracy and cross-validation results.

📈 Outputs
The script generates the following visualizations:

iris_distribution.png - Feature distribution histograms for all three species

confusion_matrix.png - Confusion matrix heatmap for the SVM model

Console output includes:

Dataset statistics and information

Model training progress

Accuracy scores for all models

Cross-validation results

Detailed classification report

Confusion matrix

📁 Project Structure
text
iris-flower-classification/
│
├── iris_flower_classification.py    # Main Python script
├── README.md                         # Project documentation
├── iris_distribution.png             # Generated visualization
└── confusion_matrix.png              # Generated visualization
👨‍💻 Author
Nallabolu Venkata Gowtham

📅 Date
December 2025

🎓 Acknowledgments
AICTE OASIS INFOBYTE - Data Science Internship Program

UCI Machine Learning Repository for the Iris dataset

scikit-learn for providing the dataset and ML tools

📝 License
This project is part of an internship task and is available for educational purposes.
