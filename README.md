# Crop Recommendation

A crop recommendation project using multinomial logistic regression. The model predicts the most suitable crop from soil nutrients and weather conditions.

## Features

The model uses these inputs:

- `N`: Nitrogen
- `P`: Phosphorus
- `K`: Potassium
- `temperature`: Temperature in degrees Celsius
- `humidity`: Humidity percentage
- `ph`: Soil pH
- `rainfall`: Rainfall in millimeters

The target column is `label`, which contains the recommended crop.

## Model

The project uses a pipeline containing:

1. `StandardScaler` for feature scaling
2. Multinomial `LogisticRegression` for classification

The model is evaluated with:

- A shuffled stratified 5-fold cross-validation
- A stratified 80/20 train/test split
- Accuracy, macro F1-score, classification report, and confusion matrix

The model achieved approximately 97% accuracy and macro F1-score on this dataset.

## Project Files

```text
Crop_recommendation.csv          Dataset
train_logistic_regression.py     Train and evaluate the model
app.py                           Streamlit user interface
requirements.txt                 Python dependencies
model_results.json               Generated model metrics and settings
```

`model_results.json` is created when the training script runs. It contains the dataset summary, model configuration, cross-validation results, and holdout test metrics.

## Installation

Create and activate a virtual environment if needed, then install the dependencies:

```powershell
pip install -r requirements.txt
```

## Train the Model

Run:

```powershell
python train_logistic_regression.py
```

This prints the evaluation results and creates or updates `model_results.json`.

## Run the Streamlit App

Start the interface with:

```powershell
streamlit run app.py
```

Enter values for the seven soil and weather features and select **Recommend Crop**. The app displays the main recommendation and the top three suitable crops.

## Notes

- The CSV headers are automatically stripped of extra spaces.
- Input fields accept numeric values as text for a simple interface.
- The dataset is balanced across 22 crop classes.
- The dataset may be synthetic, so real-world deployment should be validated with data from different farms, seasons, and regions.
