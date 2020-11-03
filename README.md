# plantery-machine-learning-predictor
### Written by Jay Sueno
_Python, Scikit, Machine Learning ALgorithms, Pandas, Hypertuning, Feature Reduction_

![exoplanets](images/exoplanets.jpg)

Outspace is pretty cool. It's vast and filled with stellar objects that tease the imagination.Over a period of nine years in deep space, the NASA Kepler space telescope has been out on a planet-hunting mission to discover hidden planets outside of our solar system. It has recorded features of each, and classified them with labels. Essentially, is that object a candidate to be an "exoplanet"?

In this project, I've tested different Machine Learning models to see which one could predict if an object is in fact an exoplanet. Using supervised learning and focusing on classifcation type algorithms, I've chosen the best model and hypertuned it to return the highest level of accuracy.

Here are the steps to choose the best fit model, and the findings.

## The Data And Which Features To Choose

The data set includes 1 column for the label or y-value - "koi_disposition". It includes 40 feature columns or x-values. Therefore, I wanted to test choosing features randomly versus using an algorithm to find optimal features. 

I created a [notebook](feature_selection.ipynb) to test different algorithms to find the best features to use in the ML models. I decided to use the [Extra Tree Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html) method and use the following features: 
```python
['koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec', 'koi_model_snr', 'koi_duration_err2']
```

## Pre-processing

The data must be cleaned and converted in order for it to be for the model to be trained and tested. This includes:
* Dropping irrelevant columns 
* Scaling features to be relative to each other so that values aren't way out there ```python MinMaxScalar() ```
* Encoding the label data and one-hot encoder so that it is in binary form using functions: ```python LabelEncoder() and to_categorical() ```

## Hypertuning

In order to further optimize our model, we used the GridSearch hyptertuning method. This method looks at specific parameters of the various ML alogrithmic models and tests which parameter settings will yield the highest accuracy. The outcomes are different for each model because the parameters are unique to each model.

## Machine Learning Alogrithm Appoaches

### Approach 1 Decision Tree ([notebook](ml_notebooks/model_1_decisiontree.ipynb))
Decision Trees are a non-parametric supervised learning method used for classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.

In this model we see the following:
* Pre-hypertuned test accuracy - 65.3%
* Hypertuned test accuracy - 67.1%

We can see that tuning the paramaters will increase the model's accuracy be a couple points.

### Approach 2 Random Forest ([notebook](ml_notebooks/model_2_randomforest.ipynb))

A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. Essentially it's looks as many Decision Trees and creates an forest of them or an ensamble. 

In this model we did not use feature selection and used the same features as the Decision Tree model. Here are our findings:
* Pre-hypertuned test accuracy - 70.8%
* Hypertuned test accuracy - 71.4%

We find this model to be better than the Decision Tree (~4%). Hypertuning doesn't yield a significant increase in the model's accuracy (~1%).

### Approach 3 Random Forest w/ [Feature Selector](ml_notebooks/feature_selection.ipynb) ([notebook](ml_notebooks/model_3_randomforest_extratreeclassifier.h5))

In this version of the Random Forest model we utlize the [Feature Selection](ml_notebooks/feature_selection.ipynb) to choose the best features. We re-train the model with the optimized features and find the following results:
* Pre-hypertuned test accuracy - 87.6%
* Hypertuned test accuracy - 87.9%

Here we find that there is a siginificant boost to accuracy. Feature reduction using an algorithm improves models especially with a data set that has many features.

### Approach 4 SVM ([notebook](ml_notebooks/model_4_svm.ipynb))

SVMs are based on the idea of finding a hyperplane that best divides a dataset into two classes. Support vectors are the data points nearest to the hyperplane, the points of a data set that, if removed, would alter the position of the dividing hyperplane. Because of this, they can be considered the critical elements of a data set.

* Pre-hypertuned test accuracy - 87.4%
* Hypertuned test accuracy - 87.8%

Here we must choose the right parameters to not over train our data. We utilized a k=5 and n_neighbors=5 parameters. It's just marginally less accurate as the Random Forest with feature selection model.

### Approach 5 Neural Network ([notebook](ml_notebooks/model_5_neuralnetwork.ipynb))

A neural network is machine learning technology that looks to replicate the human brain. Using multiple layers and neurons, the network maps different scenarios and iterates to find the best output. 

In the notebook I tested 2 neural networks: 4 layers w/ 2 hidden layers; and 3 layers with 1 hidden layer. Here are the outcomes:
* 4 layers - test accuracy 87.8%
* 3 layers - test accuracy 89.3%

Here we find an interesting finding. More layers do not mean better accuracy. There'a 1.6% better accuracy rate for only 1 hidden layer versus 2. 

## The Best Fit Model

In our exercise, we find that our tests have yielded that the most accurate model is the Neural Network with 1 hidden layer. Accuracy of 89.3% on the test data. 

Overall, we find that a nearly 90% accuracy rate for classifying planets makes our model a useful one. We have a high confidence that any future images captured by the Keplar telescope can use our model to classify if the object is a planet!

You can find the trained model file here: [model_5_neuralnetwork.h5](model_5_neuralnetwork.h5)

(An important note, in the other models it was more important to select proper features than to hypertune.)

### About the Data

The data comes from NASA's Keplar space telescope study found here: https://www.kaggle.com/nasa/kepler-exoplanet-search-results

### To learn more about Jay Sueno visit his [LinkedIn Profile](https://www.linkedin.com/in/jaysueno)

##### All rights reserved 2020. All code is created and owned by Jay Sueno. If you use his code, please visit his LinkedIn and give him a skill endorsement in python and data science. Visit him at https://www.linkedin.com/in/jaysueno/
