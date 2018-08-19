# Churn-prediction
Customer churn predicted with 87% accuracy using state-of-the-art technique.  
Customer churn prediction slashes marketing costs upto 40%. Significant historical work has been done on it and almost every company in the world uses such a tool. I present a holistic data science pipeline and the use of **LightGBM** which is very robust to unusual distributions and is one of the **fastest** machine learning algorithm till date.  

### Some results of EDA
<img src="https://drive.google.com/uc?id=1kNrfIdDnCbCWuY_-jDNiP3xfrZb-DSYn"> 
<img src="https://drive.google.com/uc?id=1DjG0nOHiJfD0nVvc-i0blXPd4n22n6gG"> 
<img src="https://drive.google.com/uc?id=1Z1nVlxm7c183YIPNA2FH50gaeGfHX9fX"> 

Exited vs Age plot -> People in the middle stages of their life leave bank more often, elderly people refrain from leaving their banks  
Exited vs Balance plot -> People with very high or very low balance are more likely to leave the bank  
Exited vs credit score -> No suprises, low credit score leads to people changing their banks  

### How are classes balanced?
Customer churn is a typicall class imbalance problem. I found SMOTE the best method for solving this problem.  
**SMOTE** (Synthetic Minority Oversampling Technique) is a synthetic data generation technique, it works as follows:
1. Identify a data point and its nearest neighbour
2. Take their difference
3. Multiply this difference by a random number between 0 and 1
4. Identify a new point on this line segment by adding the random number to the data point choosen
5. Repeat the process

Other techniques which can be used are:
* Collect more data
* Use a different performance metric like AUROC or kappa
* Undersample or Oversample the data
* Try different algorithms like decision tree and its derivates can work here as the splitting rule can force both classes to be adressed
* Try penalized models

### Installation
* Download the data from superdatascience.com
* Clone this repository to your computer.
* Get into the folder using cd Churn-prediction.
### Installing the requirements
* pip install lightgbm
* pip install pandas
* pip install numpy
* pip install matplotlib
* pip install seaborn
### Usage from scratch
* Run each cell in preprocessing.ipynb for preprocessing steps like one-hot encoding and dropping id and name columns
* Then, run each cell of Feature Engineering.ipynb for engineering some features by having some domain knowledge
* Then, run each cell of Balancing_classes.ipynb for balancing classes using SMOTE
* Then, run each cell of EDA.ipynb for knowing your features and their realtion with churning. Feature selection using exploration and correlations is done.
* Lastly, run each cell of train.ipynb to train the Lightgbm model.
