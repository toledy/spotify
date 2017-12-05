---
title: Ensemble Models
notebook: ensemble_methods.ipynb
nav_include: 4
---

## Contents
{:.no_toc}
*  
{: toc}

<font color=#6699ff> **Strategy of Fitting Ensemble Models:**<br/> 

<font color=#6699ff> 1) **Data Pre-Processing**: after reading the dataframe, we first split the training/test data by (90%-10% split) due to the small size of the dataset, then standardize the numerical columns before fitting the models, and finally checking for any missing data and impute accordingly. 

<font color=#6699ff> 2) **Model Score Function**: for the simplicity of model summary, we will create a model scoring function encompassing the following 6 metrics <br/> 
- $R^2$ (R Squared)
- $EVar$ (Explained Variance Score)
- $MAE$ (Mean Absolute Error)
- $MSE$ (Mean Squared Error)
- $MSLE$ (Mean Squared Log Error)
- $MEAE$ (Median Absolute Error)

<font color=#6699ff> 3) **Model Fitting**: here, we will fit 9 different ensemble regressors on the training data and then predict using the test data
- Gradient Boosting Regressor
- Random Forest Regressor
- Huber Regressor
- Elastic Net
- SVR
- Neural Network
- Adaboost Regressor
- Bagging Regressor
- Extra Trees Regressor

<font color=#6699ff> 4) **Model Summary**: after fitting all the models, we will present 3 summary tables based on training score, test score and qualitative metrics for the models

<font color=#6699ff> 5) **Cross Validation**: based on the summary, we will further fine-tune the parameters on the best model by cross validation

## Section 0. Import Packages



```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import PolynomialFeatures
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 200)
```




```python
from sklearn.ensemble import GradientBoostingRegressor,AdaBoostRegressor,RandomForestRegressor,ExtraTreesRegressor,BaggingRegressor
from sklearn.linear_model import LinearRegression,HuberRegressor,ElasticNet,LassoCV,RidgeCV,PassiveAggressiveRegressor,SGDRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import explained_variance_score,mean_absolute_error,mean_squared_error,mean_squared_log_error,median_absolute_error
```


## Section 1. Data Pre-Processing

### Section 1.1 Reading Dataframe



```python
data = pd.read_csv('data/Final_Dataframe.csv')
```




```python
data = data.drop("Unnamed: 0", axis=1)
```


### Section 1.2 Training/Test Data Split



```python

train_size = 0.9
test_size = 1-train_size

def data_splitter(df, train, validate=False, seed=9001):
    
    if validate:
        np.random.seed(seed)
        perm = np.random.permutation(df.index)
        m = len(df)
        train_end = int(train * m)
        validate_end = int(validate * m) + train_end
        train = df.ix[perm[:train_end]]
        validate = df.ix[perm[train_end:validate_end]]
        test = df.ix[perm[validate_end:]]
        return train, validate, test
    else:
        np.random.seed(seed)
        perm = np.random.permutation(df.index)
        m = len(df)
        train_end = int(train * m)
        train = df.ix[perm[:train_end]]
        test = df.ix[perm[train_end:]]
        return train, test
```




```python
train_df, test_df = data_splitter(data, train_size)

print("Train Size: {}".format(train_df.shape))
print("Test Size: {}".format(test_df.shape))
```


    Train Size: (1278, 949)
    Test Size: (142, 949)


### Section 1.3 Standardization



```python
numerical_columns = ['acousticness_mean','acousticness_std','dance_mean','dance_std',\
                    'energy_mean','energy_std','instrumentalness_mean','instrumentalness_std',\
                    'key_mean','key_std','liveness_mean','liveness_std','loudness_mean',\
                    'loudness_std','mode_mean','mode_std','speech_mean','speech_std',\
                    'tempo_mean','tempo_std','valence_mean','valence_std','followers_mean',\
                    'followers_std','popularity_mean','popularity_std',\
                    'house_acousticness_mean', 'hip hop_acousticness_std','pop_liveness_std', \
                     'dance_liveness_std', 'r&b_acousticness_std','rap_energy_std', 'rap_key_std',\
                     'acoustic_acousticness_std','acoustic_acousticness_mean', 'acoustic_energy_std',\
                     'acoustic_key_std']
```




```python
mean = train_df[numerical_columns].mean()
std = train_df[numerical_columns].std()

train_df[numerical_columns] = (train_df[numerical_columns] - mean)/std
test_df[numerical_columns] = (test_df[numerical_columns] - mean)/std
```


### Section 1.4 Imputation



```python
null_vals = train_df.isnull().sum()
missing_vals = null_vals[null_vals > 0].index.tolist() 
missing_vals
```





    ['acousticness_std',
     'dance_std',
     'energy_std',
     'instrumentalness_std',
     'key_std',
     'liveness_std',
     'loudness_std',
     'mode_std',
     'speech_std',
     'tempo_std',
     'time_std',
     'valence_std',
     'followers_std',
     'popularity_std',
     'hip hop_acousticness_std',
     'pop_liveness_std',
     'dance_liveness_std',
     'r&b_acousticness_std',
     'rap_energy_std',
     'rap_key_std',
     'acoustic_acousticness_std',
     'acoustic_energy_std',
     'acoustic_key_std',
     'soul_acousticness_std']



#### Method 1: KNN-Based Imputation



```python
```




```python
```




```python
```


#### Method 2: Median-Based Imputation



```python
imp = Imputer(missing_values='NaN', strategy='median', axis=1)
train_df = pd.DataFrame(imp.fit_transform(train_df), columns=data.columns)
test_df = pd.DataFrame(imp.transform(test_df), columns=data.columns)
```




```python
train_df = train_df[train_df['Followers'] != 0]
test_df = test_df[test_df['Followers'] != 0]
```




```python

y_train = np.log(train_df['Followers'])
x_train = train_df.drop('Followers', axis=1)

y_test = np.log(test_df['Followers'])
x_test = test_df.drop('Followers', axis=1)
```


## Section 2. Model Score Function

Here, we choose 6 metrics to evaluate our models: 
- $R^2$ (R Squared) = measures how well future datasets are likely to be predicted by the model. The score ranges from negative (because the model can be arbitrarily worse) to a best possible value of 1.0. Usually, the bigger the $R^2$, the better the model. Yet we do acknowledge the tedency of over-fitting with $R^2$ as with more predictors, it will only remain constant or increase.
$$R^2(y, \hat{y}) = 1 - \frac{\sum_{i=0}^{n-1}(y_i-\hat{y}_i)^2)}{\sum_{i=0}^{n-1}(y_i-\bar{y})^2}, n = \text{sample size}$$


- $EVar$ (Explained Variance Score) = measures how good the model explains the variance in the response variable. The score ranges from a minimum of 0 to a maximum of 1.0. Similar to $R^2$, the higher the score, the better the model. 
$$EVar(y, \hat{y}) = 1 - \frac{\text{Var}(y - \hat{y})}{\text{Var}(y)}$$


- $MAE$ (Mean Absolute Error) = computes the expected value of the absolute error or the $l1$ loss function. For all the error functions, the smaller the error, the better the model.
$$MAE(y, \hat{y}) = \frac{1}{n} \sum_{i=0}^{n-1} |y_i-\hat{y}_i| $$


- $MSE$ (Mean Squared Error) = computes the expected value of the squared error
$$MSE(y, \hat{y}) = \frac{1}{n} \sum_{i=0}^{n-1} (y_i-\hat{y}_i)^2 $$


- $MSLE$ (Mean Squared Log Error) = computes the expected value of the squared logarithmic error. This would probably be the most appropriate metric to evalute our models as we log-transformed our response variable - number of followers for the playlist. 
$$MSLE(y, \hat{y}) = \frac{1}{n} \sum_{i=0}^{n-1} [\ln(1+y_i)-\ln(1+\hat{y}_i)]^2$$


- $MEAE$ (Median Absolute Error) = computes the loss function by using the median of all absolute differences between the actual values and the predicted values. This metric is robust to outliers. 
$$MEAE(y, \hat{y}) = \text{median}(|y_1-\hat{y}_1|, \cdots, |y_n-\hat{y}_n|)$$



```python
def expected_score1(model, x, y):
    R2 = 0
    EVar = 0
    MAE = 0
    MSE = 0
    MSLE = 0
    MEAE = 0

    R2 += model.score(x, y)
    EVar += explained_variance_score(y, model.predict(x))
    MAE += mean_absolute_error(y, model.predict(x))
    MSE += mean_squared_error(y, model.predict(x))
    MSLE += mean_squared_log_error(y, model.predict(x))
    MEAE += median_absolute_error(y, model.predict(x))

    return pd.Series([R2 / 100., 
                      EVar / 100., 
                      MAE / 100., 
                      MSE / 100.,
                      MSLE / 100.,
                      MEAE / 100.],
                      index = ['R2', 'EVar', 'MAE', 'MSE', 'MSLE', 'MEAE'])

score = lambda model, x, y: pd.Series([model.score(x, y), 
                                       explained_variance_score(y, model.predict(x)),
                                       mean_absolute_error(y, model.predict(x)),
                                       mean_squared_error(y, model.predict(x)),
                                       mean_squared_log_error(y, model.predict(x)),
                                       median_absolute_error(y, model.predict(x))], 
                                      index=['R2', 'EVar', 'MAE', 'MSE', 'MSLE', 'MEAE'])
```


## Section 3. Model Fitting

### Section 3.1 Gradient Boosting Regressor

- According to Ben Gorman, if Linear Regression were a Toyota Camry, the Gradient Boosting Regressor would easily be a UH-60 Blackhawk Helicopter
- Gradient Boosting Regressor is an ensemble machine learning procedure that fits new models consecutively to provide a more reliable estimate of the response variable. It constructs new base-learners to be correlated with the negative gradient of the loss function 
 - least square regression (ls), 
 - least absolute deviation (lad), 
 - huber (a combination of ls and lad), 
 - quantile - which allows for quantile regression
- The choice of the loss function allows for great flexibility in Gradient Boosting and the best error function is huber for our model based on trial and error / cross-validation



```python
estgb = GradientBoostingRegressor(alpha=0.99, loss='huber', max_depth=5, learning_rate=0.04, 
                                  n_estimators=200, max_features='auto')
estgb.fit(x_train, y_train)
```





    GradientBoostingRegressor(alpha=0.99, criterion='friedman_mse', init=None,
                 learning_rate=0.04, loss='huber', max_depth=5,
                 max_features='auto', max_leaf_nodes=None,
                 min_impurity_decrease=0.0, min_impurity_split=None,
                 min_samples_leaf=1, min_samples_split=2,
                 min_weight_fraction_leaf=0.0, n_estimators=200,
                 presort='auto', random_state=None, subsample=1.0, verbose=0,
                 warm_start=False)





```python
estgb_training = score(estgb, x_train, y_train)
```




```python
estgb_test = score(estgb, x_test, y_test)
estgb_test
```





    R2      0.373701
    EVar    0.373745
    MAE     1.840267
    MSE     5.357701
    MSLE    0.076379
    MEAE    1.579995
    dtype: float64



### Section 3.2 Random Forest Regressor



```python
rfrg = RandomForestRegressor(n_estimators=200, max_depth=15, max_features='auto', min_samples_leaf=2, random_state=2)
rfrg.fit(x_train, y_train)
```





    RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=15,
               max_features='auto', max_leaf_nodes=None,
               min_impurity_decrease=0.0, min_impurity_split=None,
               min_samples_leaf=2, min_samples_split=2,
               min_weight_fraction_leaf=0.0, n_estimators=200, n_jobs=1,
               oob_score=False, random_state=2, verbose=0, warm_start=False)





```python
rfrg_training = score(rfrg, x_train, y_train)
```




```python
rfrg_test = score(rfrg, x_test, y_test)
rfrg_test
```





    R2      0.327955
    EVar    0.328377
    MAE     1.908780
    MSE     5.749035
    MSLE    0.085960
    MEAE    1.667047
    dtype: float64



### Section 3.3 Huber Regressor



```python
hubrg = HuberRegressor(max_iter=100, epsilon=1.0, alpha=10000)
hubrg.fit(x_train, y_train)
```





    HuberRegressor(alpha=10000, epsilon=1.0, fit_intercept=True, max_iter=100,
            tol=1e-05, warm_start=False)





```python
hubrg_training = score(hubrg, x_train, y_train)
```




```python
hubrg_test = score(hubrg, x_test, y_test)
hubrg_test
```





    R2     -10.592332
    EVar    -0.096876
    MAE      9.485701
    MSE     99.167105
    MSLE     5.374158
    MEAE     9.923143
    dtype: float64



### Section 3.4 Elastic Net



```python
elarg = ElasticNet(max_iter=1000, alpha=0.05, l1_ratio=1.0)
elarg.fit(x_train, y_train)
```





    ElasticNet(alpha=0.05, copy_X=True, fit_intercept=True, l1_ratio=1.0,
          max_iter=1000, normalize=False, positive=False, precompute=False,
          random_state=None, selection='cyclic', tol=0.0001, warm_start=False)





```python
elarg_training = score(elarg, x_train, y_train)
```




```python
elarg_test = score(elarg, x_test, y_test)
elarg_test
```





    R2      0.249145
    EVar    0.251129
    MAE     2.026195
    MSE     6.423224
    MSLE    0.098008
    MEAE    1.736543
    dtype: float64



### Section 3.5 SVR



```python
svrrg = SVR(kernel='rbf', C=10.0, epsilon=2.0)
svrrg.fit(x_train, y_train)
```





    SVR(C=10.0, cache_size=200, coef0=0.0, degree=3, epsilon=2.0, gamma='auto',
      kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)





```python
svrrg_training = score(svrrg, x_train, y_train)
```




```python
svrrg_test = score(svrrg, x_test, y_test)
svrrg_test
```





    R2      0.145395
    EVar    0.148554
    MAE     2.139065
    MSE     7.310753
    MSLE    0.110786
    MEAE    1.882777
    dtype: float64



### Section 3.6 Neural Network



```python
mlprg = MLPRegressor(alpha=0.000001)
mlprg.fit(x_train, y_train)
```





    MLPRegressor(activation='relu', alpha=1e-06, batch_size='auto', beta_1=0.9,
           beta_2=0.999, early_stopping=False, epsilon=1e-08,
           hidden_layer_sizes=(100,), learning_rate='constant',
           learning_rate_init=0.001, max_iter=200, momentum=0.9,
           nesterovs_momentum=True, power_t=0.5, random_state=None,
           shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
           verbose=False, warm_start=False)





```python
mlprg_training = score(mlprg, x_train, y_train)
```




```python
mlprg_test = score(mlprg, x_test, y_test)
mlprg_test
```





    R2     -2.984810e+05
    EVar   -2.942779e+05
    MAE     1.953335e+02
    MSE     2.553377e+06
    MSLE    1.149829e+00
    MEAE    3.280435e+00
    dtype: float64



### Section 3.7 Adaboost Regressor



```python
estgb_small = GradientBoostingRegressor(alpha=0.95, loss='huber', max_depth=3, learning_rate=0.01, 
                                        n_estimators=200, max_features='auto')
```




```python
adarg = AdaBoostRegressor(base_estimator=estgb_small, loss='exponential', learning_rate=0.04, n_estimators=200)
adarg.fit(x_train, y_train)
```





    AdaBoostRegressor(base_estimator=GradientBoostingRegressor(alpha=0.95, criterion='friedman_mse', init=None,
                 learning_rate=0.01, loss='huber', max_depth=3,
                 max_features='auto', max_leaf_nodes=None,
                 min_impurity_decrease=0.0, min_impurity_split=None,
                 min_samples_leaf=1, min_samples_split=2,
                 min_weight_fraction_leaf=0.0, n_estimators=200,
                 presort='auto', random_state=None, subsample=1.0, verbose=0,
                 warm_start=False),
             learning_rate=0.04, loss='exponential', n_estimators=200,
             random_state=None)





```python
adarg_training = score(adarg, x_train, y_train)
```




```python
adarg_test = score(adarg, x_test, y_test)
adarg_test
```





    R2      0.252027
    EVar    0.253087
    MAE     2.055731
    MSE     6.398564
    MSLE    0.094815
    MEAE    1.923715
    dtype: float64



### Section 3.8 Bagging Regressor



```python
bagrg = BaggingRegressor(base_estimator=estgb_small, n_estimators=10, max_samples=1.0, max_features=1.0)
bagrg.fit(x_train, y_train)
```





    BaggingRegressor(base_estimator=GradientBoostingRegressor(alpha=0.95, criterion='friedman_mse', init=None,
                 learning_rate=0.01, loss='huber', max_depth=3,
                 max_features='auto', max_leaf_nodes=None,
                 min_impurity_decrease=0.0, min_impurity_split=None,
                 min_samples_leaf=1, min_samples_split=2,
                 min_weight_fraction_leaf=0.0, n_estimators=200,
                 presort='auto', random_state=None, subsample=1.0, verbose=0,
                 warm_start=False),
             bootstrap=True, bootstrap_features=False, max_features=1.0,
             max_samples=1.0, n_estimators=10, n_jobs=1, oob_score=False,
             random_state=None, verbose=0, warm_start=False)





```python
bagrg_training = score(bagrg, x_train, y_train)
```




```python
bagrg_test = score(bagrg, x_test, y_test)
bagrg_test
```





    R2      0.216010
    EVar    0.224754
    MAE     2.027904
    MSE     6.706680
    MSLE    0.102559
    MEAE    1.696994
    dtype: float64



### Section 3.9 Extra Trees Regressor



```python
etreerg = ExtraTreesRegressor(n_estimators=100, criterion='mse', max_depth=15, max_features='auto')
etreerg.fit(x_train, y_train)
```





    ExtraTreesRegressor(bootstrap=False, criterion='mse', max_depth=15,
              max_features='auto', max_leaf_nodes=None,
              min_impurity_decrease=0.0, min_impurity_split=None,
              min_samples_leaf=1, min_samples_split=2,
              min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
              oob_score=False, random_state=None, verbose=0, warm_start=False)





```python
etreerg_training = score(etreerg, x_train, y_train)
```




```python
etreerg_test = score(etreerg, x_test, y_test)
etreerg_test
```





    R2      0.086071
    EVar    0.086101
    MAE     2.193672
    MSE     7.818249
    MSLE    0.106772
    MEAE    1.768799
    dtype: float64



## Section 4. Model Summary

### Section 4.1 Training Data

**Insights**
- Based on the training data summary, **9.Extra Trees Regressor** best explains training data, followed by **2. Random Forest Regressor**, and **1. Gradient Boosting Regressor**
- However, we need to bear in mind that too-good a fit on the training data suggests over-fitting - that is, the model has high variance and does not generalize the trends well (because fitted too well to the noise). This is unsurprising as just like classification trees, regression trees have the tendency to over-fit
- We could also easily eliminate **3. Huber Regressor**, **6. Neural Network** because of their terrible performance on the training data and they might not be a good fit for the Spotify data



```python
training_scores = pd.DataFrame({'1. Gradient Boosting': estgb_training,
                                '2. Random Forest': rfrg_training,
                                '3. Huber': hubrg_training,
                                '4. Elastic Net': elarg_training,
                                '5. SVR': svrrg_training,
                                '6. Neural Network': mlprg_training, 
                                '7. Adaboost': adarg_training,
                                '8. Bagging': bagrg_training,
                                '9. Extra Trees': etreerg_training})
print ('Training Scores:')
training_scores
```


    Training Scores:





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1. Gradient Boosting</th>
      <th>2. Random Forest</th>
      <th>3. Huber</th>
      <th>4. Elastic Net</th>
      <th>5. SVR</th>
      <th>6. Neural Network</th>
      <th>7. Adaboost</th>
      <th>8. Bagging</th>
      <th>9. Extra Trees</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>R2</th>
      <td>0.784816</td>
      <td>0.847263</td>
      <td>-10.140200</td>
      <td>0.236849</td>
      <td>0.318037</td>
      <td>-7.874661e+05</td>
      <td>0.335357</td>
      <td>0.252666</td>
      <td>0.912714</td>
    </tr>
    <tr>
      <th>EVar</th>
      <td>0.784823</td>
      <td>0.847285</td>
      <td>-0.366807</td>
      <td>0.236849</td>
      <td>0.320108</td>
      <td>-7.557535e+05</td>
      <td>0.340286</td>
      <td>0.255436</td>
      <td>0.912714</td>
    </tr>
    <tr>
      <th>MAE</th>
      <td>1.093995</td>
      <td>0.935232</td>
      <td>9.558437</td>
      <td>2.123556</td>
      <td>2.012630</td>
      <td>5.439287e+02</td>
      <td>2.065094</td>
      <td>2.110733</td>
      <td>0.626782</td>
    </tr>
    <tr>
      <th>MSE</th>
      <td>1.967414</td>
      <td>1.396467</td>
      <td>101.854169</td>
      <td>6.977440</td>
      <td>6.235147</td>
      <td>7.199764e+06</td>
      <td>6.076788</td>
      <td>6.832826</td>
      <td>0.798053</td>
    </tr>
    <tr>
      <th>MSLE</th>
      <td>0.028613</td>
      <td>0.025686</td>
      <td>5.343244</td>
      <td>0.094873</td>
      <td>0.089896</td>
      <td>2.523113e+00</td>
      <td>0.082842</td>
      <td>0.097111</td>
      <td>0.008469</td>
    </tr>
    <tr>
      <th>MEAE</th>
      <td>0.903242</td>
      <td>0.796263</td>
      <td>10.386407</td>
      <td>1.798746</td>
      <td>1.982707</td>
      <td>3.022437e+00</td>
      <td>1.930811</td>
      <td>1.795980</td>
      <td>0.442280</td>
    </tr>
  </tbody>
</table>
</div>





```python
a = training_scores[0:2].rank(1, ascending=False, method='first').reset_index()
b = training_scores[2:6].rank(1, ascending=True, method='first').reset_index()
training_ranking = a.merge(b, how = 'outer').set_index('index')
training_ranking.index.names = ['']
print ('Training Scores Ranking:')
training_ranking
```


    Training Scores Ranking:





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1. Gradient Boosting</th>
      <th>2. Random Forest</th>
      <th>3. Huber</th>
      <th>4. Elastic Net</th>
      <th>5. SVR</th>
      <th>6. Neural Network</th>
      <th>7. Adaboost</th>
      <th>8. Bagging</th>
      <th>9. Extra Trees</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>R2</th>
      <td>3.0</td>
      <td>2.0</td>
      <td>8.0</td>
      <td>7.0</td>
      <td>5.0</td>
      <td>9.0</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>EVar</th>
      <td>3.0</td>
      <td>2.0</td>
      <td>8.0</td>
      <td>7.0</td>
      <td>5.0</td>
      <td>9.0</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>MAE</th>
      <td>3.0</td>
      <td>2.0</td>
      <td>8.0</td>
      <td>7.0</td>
      <td>4.0</td>
      <td>9.0</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>MSE</th>
      <td>3.0</td>
      <td>2.0</td>
      <td>8.0</td>
      <td>7.0</td>
      <td>5.0</td>
      <td>9.0</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>MSLE</th>
      <td>3.0</td>
      <td>2.0</td>
      <td>9.0</td>
      <td>6.0</td>
      <td>5.0</td>
      <td>8.0</td>
      <td>4.0</td>
      <td>7.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>MEAE</th>
      <td>3.0</td>
      <td>2.0</td>
      <td>9.0</td>
      <td>5.0</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>6.0</td>
      <td>4.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



### Section 4.2 Test Data

**Insights**
- In terms of the test data, the top 3 performers are **1. Gradient Boosting Regressor**, **2. Random Forest Regressor**, and **7. Adaboost Regressor** if we focus on $R^2$ and $MSLE$
- The parameters in the aforementioned 3 ensemble methods could be further fine-tuned to enhance their performance in the cross-validation section



```python
test_scores = pd.DataFrame({'1. Gradient Boosting': estgb_test,
                            '2. Random Forest': rfrg_test,
                            '3. Huber': hubrg_test,
                            '4. Elastic Net': elarg_test,
                            '5. SVR': svrrg_test,
                            '6. Neural Network': mlprg_test, 
                            '7. Adaboost': adarg_test,
                            '8. Bagging': bagrg_test,
                            '9. Extra Trees': etreerg_test})
print ('Test Scores:')
test_scores
```


    Test Scores:





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1. Gradient Boosting</th>
      <th>2. Random Forest</th>
      <th>3. Huber</th>
      <th>4. Elastic Net</th>
      <th>5. SVR</th>
      <th>6. Neural Network</th>
      <th>7. Adaboost</th>
      <th>8. Bagging</th>
      <th>9. Extra Trees</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>R2</th>
      <td>0.373701</td>
      <td>0.327955</td>
      <td>-10.592332</td>
      <td>0.249145</td>
      <td>0.145395</td>
      <td>-2.984810e+05</td>
      <td>0.252027</td>
      <td>0.216010</td>
      <td>0.086071</td>
    </tr>
    <tr>
      <th>EVar</th>
      <td>0.373745</td>
      <td>0.328377</td>
      <td>-0.096876</td>
      <td>0.251129</td>
      <td>0.148554</td>
      <td>-2.942779e+05</td>
      <td>0.253087</td>
      <td>0.224754</td>
      <td>0.086101</td>
    </tr>
    <tr>
      <th>MAE</th>
      <td>1.840267</td>
      <td>1.908780</td>
      <td>9.485701</td>
      <td>2.026195</td>
      <td>2.139065</td>
      <td>1.953335e+02</td>
      <td>2.055731</td>
      <td>2.027904</td>
      <td>2.193672</td>
    </tr>
    <tr>
      <th>MSE</th>
      <td>5.357701</td>
      <td>5.749035</td>
      <td>99.167105</td>
      <td>6.423224</td>
      <td>7.310753</td>
      <td>2.553377e+06</td>
      <td>6.398564</td>
      <td>6.706680</td>
      <td>7.818249</td>
    </tr>
    <tr>
      <th>MSLE</th>
      <td>0.076379</td>
      <td>0.085960</td>
      <td>5.374158</td>
      <td>0.098008</td>
      <td>0.110786</td>
      <td>1.149829e+00</td>
      <td>0.094815</td>
      <td>0.102559</td>
      <td>0.106772</td>
    </tr>
    <tr>
      <th>MEAE</th>
      <td>1.579995</td>
      <td>1.667047</td>
      <td>9.923143</td>
      <td>1.736543</td>
      <td>1.882777</td>
      <td>3.280435e+00</td>
      <td>1.923715</td>
      <td>1.696994</td>
      <td>1.768799</td>
    </tr>
  </tbody>
</table>
</div>





```python
a = test_scores[0:2].rank(1, ascending=False, method='first').reset_index()
b = test_scores[2:6].rank(1, ascending=True, method='first').reset_index()
test_ranking = a.merge(b, how = 'outer').set_index('index')
test_ranking.index.names = ['']
print ('Test Scores Ranking:')
test_ranking
```


    Test Scores Ranking:





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1. Gradient Boosting</th>
      <th>2. Random Forest</th>
      <th>3. Huber</th>
      <th>4. Elastic Net</th>
      <th>5. SVR</th>
      <th>6. Neural Network</th>
      <th>7. Adaboost</th>
      <th>8. Bagging</th>
      <th>9. Extra Trees</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>R2</th>
      <td>1.0</td>
      <td>2.0</td>
      <td>8.0</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>9.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>EVar</th>
      <td>1.0</td>
      <td>2.0</td>
      <td>8.0</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>9.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>MAE</th>
      <td>1.0</td>
      <td>2.0</td>
      <td>8.0</td>
      <td>3.0</td>
      <td>6.0</td>
      <td>9.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>MSE</th>
      <td>1.0</td>
      <td>2.0</td>
      <td>8.0</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>9.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>MSLE</th>
      <td>1.0</td>
      <td>2.0</td>
      <td>9.0</td>
      <td>4.0</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>MEAE</th>
      <td>1.0</td>
      <td>2.0</td>
      <td>9.0</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>8.0</td>
      <td>7.0</td>
      <td>3.0</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
</div>





```python
```


## Section 5. Cross Validation



```python
grid = {'max_depth': [1, 2, 3, 5, 10],
        'learning_rate': [0.02, 0.04, 0.06, 0.08, 0.10], 
        'n_estimators': [50, 100, 200], 
        'alpha': [0.1, 0.3, 0.5, 0.7, 0.9, 0.99], 
        'max_features': ['sqrt','auto', 'log2']}
```




```python
clf = GridSearchCV(estimator=GradientBoostingRegressor(), param_grid=grid, n_jobs=1, cv=5)
```




```python
clf.fit(x_train, y_train)
```




```python
clf_test = score(clf, x_test, y_test)
clf_test
```




```python
print ("Best Estimator Parameters")
print ("loss:", clf.best_estimator_.loss)
print ("max_depth: %d" %clf.best_estimator_.max_depth)
print ("n_estimators: %d" %clf.best_estimator_.n_estimators)
print ("learning rate: %.1f" %clf.best_estimator_.learning_rate)
print ("alpha: %.1f" %clf.best_estimator_.alpha)
print ("max_features:", clf.best_estimator_.max_features)
```




```python
estgb_lin = GradientBoostingRegressor(loss='huber', max_depth=3, learning_rate=0.1, n_estimators=200)
estgb_lin.fit(x_train, y_train)
```




```python
estgb_lin.score(x_test, y_test)
```




```python
from sklearn.decomposition import PCA

pca_var = PCA(n_components=0.99, whiten=True)

x_train_pca = pca_var.fit_transform(x_train)
x_test_pca = pca_var.transform(x_test)

print('Original Number of Predictors:', x_train.shape[1])
print('Reduced Number of Predictors:', x_train_pca.shape[1])

print('Total Explained Variance:', np.sum(pca_var.explained_variance_ratio_))
```


    Original Number of Predictors: 948
    Reduced Number of Predictors: 1
    Total Explained Variance: 1.0




```python
estgb_lin = GradientBoostingRegressor(loss='huber', max_depth=3, learning_rate=0.1, n_estimators=200)
estgb_lin.fit(x_train, y_train)
```





    GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
                 learning_rate=0.1, loss='huber', max_depth=3,
                 max_features=None, max_leaf_nodes=None,
                 min_impurity_decrease=0.0, min_impurity_split=None,
                 min_samples_leaf=1, min_samples_split=2,
                 min_weight_fraction_leaf=0.0, n_estimators=200,
                 presort='auto', random_state=None, subsample=1.0, verbose=0,
                 warm_start=False)





```python
estgb_lin.score(x_test, y_test)
```





    0.30593063863110215


