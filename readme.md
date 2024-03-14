# Anomaly Detection using LSTM Autoencoder for Financial Data

## Introduction
In this project, we implemented an anomaly detection system using a combination of Long Short-Term Memory (LSTM) neural networks and autoencoders. The goal was to detect anomalies in financial data, which often exhibit patterns that deviate significantly from normal behavior.

## LSTM (Long Short-Term Memory)
LSTM is a type of recurrent neural network (RNN) architecture that is well-suited for processing sequences of data. It has mechanisms called "gates" that allow it to learn and remember long-term dependencies in sequential data, making it suitable for tasks such as time series prediction and anomaly detection.

## Autoencoder
An autoencoder is an unsupervised learning algorithm that consists of an encoder and a decoder. It compresses the input data into a lower-dimensional representation and reconstructs the original input from the compressed representation. Anomalies are detected by comparing the reconstruction error to a predefined threshold.

## Financial Dataset
The financial dataset used in this project contains the following columns:
1. Timestamp: Date and time of the transaction
2. TransactionID: Unique identifier for each transaction
3. AccountID: Identifier for the account involved in the transaction
4. Location: Location of the transaction
5. Merchant: Merchant involved in the transaction
6. TransactionType: Type of transaction (e.g., purchase, withdrawal)
7. Amount: Monetary value of the transaction

## Steps Taken
1. **Data Preprocessing**: We preprocessed the financial data by scaling numerical features using StandardScaler and one-hot encoding categorical features using OneHotEncoder from the scikit-learn library.

2. **Model Design**: We designed an LSTM autoencoder architecture using PyTorch to capture temporal dependencies in the sequential financial data.

3. **Training**: The LSTM autoencoder model was trained using the training dataset to minimize the reconstruction error between the input and the reconstructed output.

4. **Anomaly Detection**: After training the model, we applied it to the testing dataset to compute reconstruction errors. Anomalies were detected based on a predefined threshold determined through analysis or cross-validation.

5. **Visualization**: We visualized the anomalies detected in the testing dataset by plotting the original financial data with anomalies highlighted.

## Conclusion
In conclusion, we successfully implemented an anomaly detection system for financial data using LSTM autoencoders. The project demonstrated the effectiveness of deep learning techniques for detecting anomalies in sequential financial data and highlighted the importance of preprocessing, model design, and threshold selection in building robust anomaly detection systems.

## References
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)

