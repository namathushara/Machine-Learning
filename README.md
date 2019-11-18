# Machine-Learning
Evaluation of classification algorithms on numerical and categorical datasets
Dataset description:
Dataset1: The dataset is about Appliances energy prediction from UCI Machine Learning Repository.(link- https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction#)
•	The dataset consists of 19735 instances with 29 features
•	The dependent variable is Appliances which is the total energy usage in Wh
•	There are no missing values in the data set

Data Normalisation: The data is normalised by substracting with mean and dividing with standard deviation so that all the features are on a similar scale for making better predictions.
data=data-data.mean()/data.std()

Dataset 2: The dataset is about mushroom classification into Poisonous or edible,downloaded from Kaggle(link: https://www.kaggle.com/uciml/mushroom-classification)
•	The dataset consists of 8124 instances with 23 features
•	The target variable is mushroom class(target)
•	There are no missing values in the dataset
•	All the features in the dataset are categorical,which is the motive behind selecting this particular dataset to explore various algorithms with categorical data

One Hot Encoding: As all of the features are categorical,one hot encoding is done on Mushroom dataset to convert into numerical features for using scikit libraries.

Data split: The dataset is partitioned into training data, and test data with 80%,20% of data respectively.
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)

Linear regression with gradient descent from scratch without using sk-learn

Logistic regression

Support Vector Machine:
Kernels experimented here are Linear, Rbf, Poly, Sigmoid

Decision Tree Classifier: Entropy metric is used here in calculating information gain.

Pruning: Here cross validated datasets are experimented with pruning
The datsets are experimented with various tree depths

Adaptive boosting: Adaboost algorithm is used here to boost the tree. Decision tree classifiers with depth 1 are used as weak learners in this algorithm.The accuracy is tested for different number of weak learners which is similar to pruning

Artificial Neural Networks: Used keeras open source neural network library which runs on top of Tensorflow and Theano 

K Nearest Neighbors: Knn is experimented with k= 1 t0 25, and p=1(Manhattan distance), p=2(Euclidean distance)
