# PROJECT TITLE
  Drug Classification Using Artificial Neural Networks
 This project classifies drugs into five categories using an Artificial Neural Network (ANN). The dataset includes features like age, sex, blood pressure, cholesterol levels, and Na-to-K ratio, which are used as inputs to the model.
# Getting Started
  These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.
# Prerequisites 
What things you need to install the software and how to install them:

- Python 3.7 or higher
- Required libraries:
  - `pandas`
  - `numpy`
  - `scikit-learn`

You can install these libraries using:
pip install pandas numpy scikit-learn

# Installing
A step-by-step series of instructions to get the development environment running:

Clone the repository:
git clone https://github.com/your-username/drug-classification-ann.git

Navigate to the project directory:
cd drug-classification-ann
Ensure the dataset drugdataset.csv is placed in the same directory as the script.

Run the script:
python drug_classification_ann.py

The script includes the following steps

# Loading the Dataset

    #Load Libraries
    import pandas as pd
    import numpy as np

    #Load Data
    data = pd.read_csv("drugdataset.csv")
    print(data.head())

 # Data Preprocessing

      #Preprocess Data
      X = data.drop('Drug', axis=1).to_numpy()
      y = data['Drug'].to_numpy()

     #Split into training and testing sets
     from sklearn.model_selection import train_test_split
     X_train, X_test, y_train, y_test = train_test_split(
     X, y, stratify=y, test_size=0.2, random_state=100
    )

    #Scale the data
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)

# Training the ANN

      #Train an Artificial Neural Network
       from sklearn.neural_network import MLPClassifier
       mlp = MLPClassifier(
       hidden_layer_sizes=(5, 4, 5),
       activation='relu',
       solver='adam',
       max_iter=10000,
       random_state=100
    )
       mlp.fit(X_train_scaled, y_train)

# Making Predictions and Evaluating

#Predictions

    predictions = mlp.predict(X_test_scaled)

#Evaluation

     from sklearn.metrics import classification_report, confusion_matrix
     print("Confusion Matrix:")
     print(confusion_matrix(y_test, predictions))

     print("\nClassification Report:")
     print(classification_report(y_test, predictions))

# Sample Output:

Confusion Matrix:
[[ 5  0  0  0  0]
 [ 0  2  0  0  1]
 [ 0  0  3  0  0]
 [ 0  0  0 11  0]
 [ 0  0  0  1 17]]

Classification Report:
              precision    recall  f1-score   support
       drugA       1.00      1.00      1.00         5
       drugB       1.00      0.67      0.80         3
       drugC       1.00      1.00      1.00         3
       drugX       0.92      1.00      0.96        11
       drugY       0.94      0.94      0.94        18

# Running the Tests
Explain how to run the automated tests for this system:

Confusion Matrix:

Used to evaluate the performance for each drug category.

    print(confusion_matrix(y_test, predictions))

Classification Report:

provides metrics such as precision, recall, and F1-score.

    print(classification_report(y_test, predictions))

# Break down into end-to-end tests
The script includes end-to-end testing through evaluation metrics such as:

# Confusion Matrix:
Explains how well the model performs for each class.
Confusion Matrix:
[[ 5  0  0  0  0]
 [ 0  2  0  0  1]
 [ 0  0  3  0  0]
 [ 0  0  0 11  0]
 [ 0  0  0  1 17]]

# Classification Report:
 Provides detailed metrics like precision, recall, and F1-score.
 Classification Report:
              precision    recall  f1-score   support
       drugA       1.00      1.00      1.00         5
       drugB       1.00      0.67      0.80         3
       ...

# And coding style tests
No specific coding style tests included. All code adheres to PEP8 style guidelines.

# Deployment
Add additional notes about how to deploy this on a live system:
    This project is for educational purposes and is not intended for deployment.
    If needed, you can deploy it in a local environment using tools like Jupyter Notebook.

# Built With

    scikit-learn - Library for machine learning algorithms
    pandas - Used for data manipulation
    numpy - Utilized for numerical computations

# Contributing
Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests to us.

# Versioning
We use SemVer for versioning. For the versions available, see the tags on this repository.

# Authors
Shubh  - Initial work- The Artificial Neural Network implementation for drug classification 

# License
This project is licensed under the MIT License - see the LICENSE.md file for details.

# Acknowledgments
    Inspiration from the DATA 1200 and DATA 1202 course
    Guidance from the course instructor
    


<!---
ShubhNDA46/ShubhNDA46 is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
