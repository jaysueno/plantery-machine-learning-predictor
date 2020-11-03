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

## Hypertuning

In order to further optimize our model, we used the GridSearch hyptertuning method. This method looks at specific parameters of the various ML alogrithmic models and tests which parameter settings will yield the highest accuracy. The outcomes are different for each model because the parameters are unique to each model.

## Machine Learning Alogrithm Appoaches

* Notebook 1 - Decision Tree 
* Notebook 2 - Random Forest
* Notebook 3 - Random Forest
*
* Notebook 4 - Neural Network

## The Best Fit Model

## About the Code

*
*

### About the Data

The data comes from NASA's Keplar space telescope study found here: https://www.kaggle.com/nasa/kepler-exoplanet-search-results

### To learn more about Jay Sueno visit his [LinkedIn Profile](https://www.linkedin.com/in/jaysueno)

##### All rights reserved 2020. All code is created and owned by Jay Sueno. If you use his code, please visit his LinkedIn and give him a skill endorsement in python and data science. Visit him at https://www.linkedin.com/in/jaysueno/
