# Churn-prediction
Customer churn prediction slashes marketing costs upto 40%. Significant historical work has been done on it and almost every company in the world uses such a tool. I present a holistic data science pipeline and the use of LightGBM which is very robust to unusual distributions and is one of the **fastest** machine learning algorithm till date.  
#### Basics of LGBM

## Installation
### Download the data
* Clone this repository to your computer.
* Get into the folder using cd Churn-prediction.
### Installing the requirements
* pip install keras
* pip install tensorflow
## Usage from scratch
* Run each cell in preprocessing.ipynb for preprocessing steps like one-hot encoding and dropping id and name columns
* Then, run each cell of Feature Engineering.ipynb for engineering some features by having some domain knowledge
* Then, run each cell of Balancing_classes.ipynb for balancing classes using SMOTE
* Then, run each cell of EDA.ipynb for knowing your features and their realtion with churning. Feature selection using exploration and correlations is done.
* Lastly, run each cell of train.ipynb to train the Lightgbm model.
