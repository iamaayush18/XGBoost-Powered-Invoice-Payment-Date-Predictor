
# XGBoost-Powered Invoice Payment Date Predictor

### Overview
This project leverages the power of **XGBoost**, a highly efficient and accurate machine learning algorithm, to predict the **due payment dates** of pending invoices for various companies. The model aims to provide businesses with an effective way to anticipate when invoices will be paid, aiding in cash flow management and financial planning.

### Problem Statement
Businesses often face uncertainties when it comes to receiving payments on time. Unpaid invoices can create bottlenecks in cash flow and financial operations. This project addresses this challenge by building an XGBoost-based regressor to accurately predict the payment dates of pending invoices based on historical data.

### Project Structure
- **`H2HBABBA1250.csv`**: The dataset used to train and test the XGBoost model.
- **`ML-BASED-PAYMENT-DATE-PREDICTION-MODEL.ipynb`**: The Jupyter notebook containing the entire workflow of the project, including data preprocessing, model training, and evaluation.
- **`README.md`**: Project overview and usage instructions (this file).

### Features
- **Data Cleaning & Preprocessing**: Prepares the invoice data for use in the model by handling missing values, formatting dates, and feature engineering.
- **Model Training**: Trains an XGBoost regressor to predict payment dates using past invoices.
- **Model Evaluation**: Measures model performance using multiple regression evaluation metrics.

### Dataset
The dataset (`H2HBABBA1250.csv`) contains historical invoice payment data with the following features:
- **Invoice ID**: Unique identifier for each invoice.
- **Company Name**: The name of the company responsible for the payment.
- **Invoice Amount**: The amount due for each invoice.
- **Payment Date**: The actual date when the invoice was paid.
- **Due Date**: The expected payment date for the invoice.

### Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/iamaayush18/XGBoost-Powered-Invoice-Payment-Date-Predictor.git
   cd XGBoost-Powered-Invoice-Payment-Date-Predictor
   ```

2. **Install Dependencies**: Install the necessary Python libraries using pip.
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Jupyter Notebook**: Open and execute the `.ipynb` file to run the model.
   ```bash
   jupyter notebook ML-BASED-PAYMENT-DATE-PREDICTION-MODEL.ipynb
   ```

4. **Model Training**: Follow the notebook steps to preprocess the data, train the XGBoost model, and evaluate the results.

### Model Used
The primary model used in this project is **XGBoost**, which is known for:
- High performance on structured/tabular data.
- Robust handling of missing data.
- High predictive accuracy and efficient training.

### Evaluation Metrics
The performance of the XGBoost model is evaluated using the following metrics:
- **Mean Absolute Error (MAE)**: Measures the average magnitude of errors in the predictions.
- **Mean Squared Error (MSE)**: Measures the squared difference between actual and predicted dates.
- **R-squared (R²)**: Indicates the proportion of variance in the dependent variable that the model can explain.

### Future Enhancements
- **Model Optimization**: Tune hyperparameters to improve accuracy.
- **Feature Expansion**: Include additional data points such as payment terms, company size, and industry-specific factors.
- **Deployment**: Develop a web interface for real-time invoice payment predictions.

### Contributing
Contributions are welcome! If you have suggestions or want to add new features, feel free to fork this repository and submit a pull request.


