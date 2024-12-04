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

End with an example of getting some data out of the system:
The script will output evaluation metrics like confusion matrix and classification report for testing accuracy.

# Running the tests
Explain how to run the automated tests for this system:

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
    Hat tip to anyone whose code was used
    Inspiration from the DATA 1200 and DATA 1202 course
    Guidance from the course instructor
    


<!---
ShubhNDA46/ShubhNDA46 is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
