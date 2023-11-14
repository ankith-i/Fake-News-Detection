# Python-Based Fake News Detection

This project focuses on classifying fake news articles using state-of-the-art natural language processing techniques and machine learning algorithms. By employing Python's sci-kit libraries, this initiative stands at the forefront of combating misinformation.

## Introduction

This guide will help you set up and run the Fake News Detection project on your local environment for both development and testing purposes. It also includes instructions for deploying the project in a live environment.

### Prerequisites

Before beginning, ensure you have the following installed:

1. **Python 3.9:**
   - Ensure Python 3.9 is installed on your machine. You can download it from [Python's official website](https://www.python.org/downloads/).
   - Optionally, set up PATH variables to run Python programs directly. Follow the guide [here](https://www.pythoncentral.io/add-python-to-path-python-is-not-recognized-as-an-internal-or-external-command/).

2. **Anaconda (Alternative to Python 3.6):**
   - Download and install Anaconda from [here](https://www.anaconda.com/download/).
   
3. **Required Packages:**
   - After installing Python or Anaconda, install the following packages:
     - Sklearn (scikit-learn)
     - numpy
     - scipy
   - Installation commands:
     - For Python:
       ```
       pip install -U scikit-learn
       pip install numpy
       pip install scipy
       ```
     - For Anaconda:
       ```
       conda install -c scikit-learn
       conda install -c anaconda numpy
       conda install -c anaconda scipy
       ```

#### Dataset

The project uses the LIAR dataset, which comprises test, train, and validation files in .tsv format. However, for simplicity, we have reduced the dataset to include just two variables for classification: Statement and Label (True/False).

### Project Files

- **DataPrep.py:** Preprocesses input documents and texts, including tokenizing, stemming, and exploratory data analysis.
- **FeatureSelection.py:** Feature extraction and selection using sci-kit learn techniques like bag-of-words, n-grams, and tf-tdf weighting.
- **Classifier.py:** Builds classifiers for prediction, including Naive-Bayes, Logistic Regression, and others. Implements parameter tuning and selects the best model.
- **Prediction.py:** Uses the best model (`Logistic Regression`) for classification, providing the probability of truth for a given news article.

### Project Workflow

![Process Flow](https://github.com/nishitpatel01/Fake_News_Detection/blob/master/images/ProcessFlow.PNG)

### Performance Metrics

**Learning Curves for Candidate Models:**

- Logistic Regression Classifier:

  ![LR Learning Curve](https://github.com/nishitpatel01/Fake_News_Detection/blob/master/images/LR_LCurve.PNG)

- Random Forest Classifier:

  ![RF Learning Curve](https://github.com/nishitpatel01/Fake_News_Detection/blob/master/images/RF_LCurve.png)

### Future Enhancements

Future implementations might include advanced feature selection methods like POS tagging, word2vec, and topic modeling, as well as increasing the training data size.

### Setup and Execution

Follow these steps to get the environment running:

1. Clone the repository:
''' 
git clone https://github.com/nishitpatel01/Fake_News_Detection.git 
'''

2. Navigate to the project folder and run `prediction.py`:
- For Anaconda:
  ```
  cd /path/to/cloned/folder
  python prediction.py
  ```
- For Python (without PATH variable):
  ```
  /path/to/python.exe /path/to/cloned/folder/prediction.py
  ```
- For Python (with PATH variable):
  ```
  cd /path/to/cloned/folder
  python prediction.py
  ```

3. Enter the news headline when prompted, and the program will classify it as 'True' or 'False', providing a probability score.

Happy Detecting!


