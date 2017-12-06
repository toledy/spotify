---
title: Baseline Models
notebook: baseline_models.ipynb
nav_include: 3
---

## Contents
{:.no_toc}
*  
{: toc}




## Baseline models 

In this section, our goal is to examine the baseline performance of simple models on the test set. We would use these baseline test set r2 score as a reference for building more complex models. The models included in this section are mostly multilinear regression models with different subset of predictors and possible polynomial/interaction terms. PCA and LASSO/RRIDGE are explored here as well. 

We choose to start with linear regression because of high interpretability and low computational compexity. We can easily interpret the result of regression coeffcients in such fashion: holding all other predictors constant, one unit incrase in the specified predictor leads to a change in units indicated by the coefficent in the response variables. Linear regression also has a closed form solution for coefficients which reduces computational complexity.

Linear regression has the following assumptions: 
* There is a linear relationship between response varibles and predictors
* Residuals are independent
* Residuals are normally distributed
* Residuals has constant variance 

To evaluate baseline models, I use the metric r2. R2 is the proportion of overall variability of Y explained by the model. R2 has a caveat in the sense that it will always go up for the training set as we include more predictors. R2 will tend towards overfitting if we conduct model selection through R2. However, model selection in not the main goal 

### Part(a) Multilinear regression model with all predictors

As the first step, we fit a multilinear regression with all predictors that we have. This model would not neccessarily perform well since it has 949 predictors. However, we want to conduct a sanity check through this model to make sure that prediction power is reasonable.

The test r2 score for multilinear regression model with all predictors is -2.779. Our prediction does not perform well compared to the mean of response varaibles. This is evidence suggesting that we are overfitting our model with too many predictors. Therefore, going forward, we would like to fit a regression model with a subset of predictors in this model.



```python
X=sm.add_constant(x_train)
X_test=sm.add_constant(x_test)
model=sm.OLS(y_train,X)
results=model.fit()
r2_test_a=r2_score(y_test,results.predict(X_test))
print("For multilinear regression with all terms,R2 score for training set: {}".format(r2_score(y_train,results.predict(X))))
print("For multilinear regression with all terms,R2 score for test set: {}".format(r2_test_a))
#results.summary()
```


    For multilinear regression with all terms,R2 score for training set: 0.7655523815712503
    For multilinear regression with all terms,R2 score for test set: -2.779712758436489


###  Part(b) Multilinear regression model with top artists predictors

In our prelimnary EDA anlaysis, we believed that top artists would be a good predictor for the success of a playlist. Therefore, here we fit two models in part (b) and (c) that only include predictors including artists. In part (b), we use the dummy variables of top 30 artists as predictors. As a note, for part (b), top artist are those who appear most often in playlists with 350,000+ followers. With more than 350,000 followers, a playlist will beat 80% of playlists in terms of followers. 

Our regression generates a test r2 score of 0.017, which is a lot better than -2.8 in part (a). Therefore, there is good reason to consider these predictors in future model building.

From the regression summary table, we get the list of significant top 30 artist predictors:
* 'Galantis', 'Post Malone', 'Yo Gotti', 'Ellie Goulding'





    With tio 30 artists,R2 score for test set: 0.017373740567596885




```python
print("With top 30 artists,R2 score for training set: {}".format(r2_score(y_train,results2.predict(X1))))
print("With top 30 artists,R2 score for test set: {}".format(r2_score(y_test,results2.predict(X2))))
results2.summary()
```


    With top 30 artists,R2 score for training set: 0.06172694503105225
    With top 30 artists,R2 score for test set: 0.017373740567596885





<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>Followers</td>    <th>  R-squared:         </th> <td>   0.062</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.040</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   2.784</td>
</tr>
<tr>
  <th>Date:</th>             <td>Wed, 06 Dec 2017</td> <th>  Prob (F-statistic):</th> <td>1.60e-06</td>
</tr>
<tr>
  <th>Time:</th>                 <td>13:37:49</td>     <th>  Log-Likelihood:    </th> <td> -3134.4</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>  1257</td>      <th>  AIC:               </th> <td>   6329.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  1227</td>      <th>  BIC:               </th> <td>   6483.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>    29</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
           <td></td>             <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>              <td>    9.5080</td> <td>    0.091</td> <td>  104.136</td> <td> 0.000</td> <td>    9.329</td> <td>    9.687</td>
</tr>
<tr>
  <th>Lil Wayne</th>          <td>    0.8115</td> <td>    0.828</td> <td>    0.980</td> <td> 0.327</td> <td>   -0.813</td> <td>    2.436</td>
</tr>
<tr>
  <th>Van Morrison</th>       <td>    0.0322</td> <td>    0.833</td> <td>    0.039</td> <td> 0.969</td> <td>   -1.603</td> <td>    1.667</td>
</tr>
<tr>
  <th>Galantis</th>           <td>    3.3362</td> <td>    1.093</td> <td>    3.051</td> <td> 0.002</td> <td>    1.191</td> <td>    5.481</td>
</tr>
<tr>
  <th>Wiz Khalifa</th>        <td>    0.3097</td> <td>    0.889</td> <td>    0.348</td> <td> 0.728</td> <td>   -1.435</td> <td>    2.054</td>
</tr>
<tr>
  <th>Rihanna</th>            <td>    0.5662</td> <td>    0.777</td> <td>    0.728</td> <td> 0.467</td> <td>   -0.959</td> <td>    2.091</td>
</tr>
<tr>
  <th>Post Malone</th>        <td>    2.4444</td> <td>    1.041</td> <td>    2.349</td> <td> 0.019</td> <td>    0.402</td> <td>    4.486</td>
</tr>
<tr>
  <th>Axwell /\ Ingrosso</th> <td>    2.0688</td> <td>    1.121</td> <td>    1.846</td> <td> 0.065</td> <td>   -0.130</td> <td>    4.267</td>
</tr>
<tr>
  <th>Young Thug</th>         <td>   -1.0316</td> <td>    0.937</td> <td>   -1.101</td> <td> 0.271</td> <td>   -2.870</td> <td>    0.806</td>
</tr>
<tr>
  <th>JAY Z</th>              <td>    0.6612</td> <td>    0.756</td> <td>    0.875</td> <td> 0.382</td> <td>   -0.822</td> <td>    2.144</td>
</tr>
<tr>
  <th>A$AP Rocky</th>         <td>    0.7276</td> <td>    0.969</td> <td>    0.751</td> <td> 0.453</td> <td>   -1.174</td> <td>    2.629</td>
</tr>
<tr>
  <th>Yo Gotti</th>           <td>    3.6842</td> <td>    1.057</td> <td>    3.485</td> <td> 0.001</td> <td>    1.610</td> <td>    5.758</td>
</tr>
<tr>
  <th>Chance The Rapper</th>  <td>    1.9356</td> <td>    1.074</td> <td>    1.801</td> <td> 0.072</td> <td>   -0.172</td> <td>    4.043</td>
</tr>
<tr>
  <th>Led Zeppelin</th>       <td>    1.7855</td> <td>    0.925</td> <td>    1.931</td> <td> 0.054</td> <td>   -0.029</td> <td>    3.600</td>
</tr>
<tr>
  <th>Otis Redding</th>       <td>    0.3608</td> <td>    0.989</td> <td>    0.365</td> <td> 0.715</td> <td>   -1.580</td> <td>    2.301</td>
</tr>
<tr>
  <th>21 Savage</th>          <td>    2.1001</td> <td>    1.173</td> <td>    1.790</td> <td> 0.074</td> <td>   -0.202</td> <td>    4.402</td>
</tr>
<tr>
  <th>Deorro</th>             <td>    1.9751</td> <td>    1.074</td> <td>    1.840</td> <td> 0.066</td> <td>   -0.131</td> <td>    4.081</td>
</tr>
<tr>
  <th>Elton John</th>         <td>    1.2259</td> <td>    0.875</td> <td>    1.401</td> <td> 0.161</td> <td>   -0.491</td> <td>    2.943</td>
</tr>
<tr>
  <th>SZA</th>                <td>    0.9203</td> <td>    0.914</td> <td>    1.007</td> <td> 0.314</td> <td>   -0.873</td> <td>    2.713</td>
</tr>
<tr>
  <th>Ty Dolla $ign</th>      <td>    0.7080</td> <td>    0.955</td> <td>    0.741</td> <td> 0.459</td> <td>   -1.166</td> <td>    2.582</td>
</tr>
<tr>
  <th>Ryan Adams</th>         <td>    0.8033</td> <td>    0.744</td> <td>    1.080</td> <td> 0.280</td> <td>   -0.656</td> <td>    2.262</td>
</tr>
<tr>
  <th>Birdy</th>              <td>    0.5316</td> <td>    1.076</td> <td>    0.494</td> <td> 0.621</td> <td>   -1.579</td> <td>    2.642</td>
</tr>
<tr>
  <th>Miguel</th>             <td>    1.0180</td> <td>    1.024</td> <td>    0.994</td> <td> 0.320</td> <td>   -0.991</td> <td>    3.027</td>
</tr>
<tr>
  <th>Niall Horan</th>        <td>    1.3225</td> <td>    0.952</td> <td>    1.390</td> <td> 0.165</td> <td>   -0.545</td> <td>    3.190</td>
</tr>
<tr>
  <th>Ellie Goulding</th>     <td>    1.7334</td> <td>    0.803</td> <td>    2.159</td> <td> 0.031</td> <td>    0.158</td> <td>    3.309</td>
</tr>
<tr>
  <th>Commodores</th>         <td>    0.9663</td> <td>    1.004</td> <td>    0.962</td> <td> 0.336</td> <td>   -1.004</td> <td>    2.937</td>
</tr>
<tr>
  <th>Radiohead</th>          <td>    0.2564</td> <td>    0.882</td> <td>    0.291</td> <td> 0.771</td> <td>   -1.474</td> <td>    1.987</td>
</tr>
<tr>
  <th>SYML</th>               <td>    1.8656</td> <td>    1.176</td> <td>    1.586</td> <td> 0.113</td> <td>   -0.442</td> <td>    4.173</td>
</tr>
<tr>
  <th>First Aid Kit</th>      <td>    0.9468</td> <td>    0.958</td> <td>    0.988</td> <td> 0.323</td> <td>   -0.933</td> <td>    2.827</td>
</tr>
<tr>
  <th>Lord Huron</th>         <td>    0.5109</td> <td>    0.861</td> <td>    0.593</td> <td> 0.553</td> <td>   -1.179</td> <td>    2.201</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>79.124</td> <th>  Durbin-Watson:     </th> <td>   1.955</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  88.754</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.630</td> <th>  Prob(JB):          </th> <td>5.34e-20</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.672</td> <th>  Cond. No.          </th> <td>    15.2</td>
</tr>
</table>







    Index(['Galantis', 'Post Malone', 'Yo Gotti', 'Ellie Goulding'], dtype='object')


###  Part(c) Multilinear regression model with top artists counts predictors

We continue to evaluate artist predictors in part (c). Here, top artists are defined differently from part (b).We first sum up the total number of followers for playlists that include an artist. Then, we rank the artists basing on the aggregated playlist followers. For part (c), the predictors are the number of top 10/10-20/20-30/30-40/40-50 artists that a playlist has.

From regression result, we see that r2 training result is 0.023 and test result is -0.03. The significant predictors include: "top_30_40", "top_40_50". It seems like part(c) r2 test score are lower than that in part(b), indicating that part(b) artist predictors has more power in predicting playlist followers thatn part(c) predictors. We should put more emphasis on part(b) predictors when constructing our best model.







```python
print("With top artists count predictors,R2 score for training set: {}".format(r2_score(y_train,results3.predict(X3))))
print("With top artists count predictors,R2 score for test set: {}".format(r2_score(y_test,results3.predict(X4))))
results3.summary()
```


    With top artists count predictors,R2 score for training set: 0.012645473737053159
    With top artists count predictors,R2 score for test set: -0.031235636069632644





<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>Followers</td>    <th>  R-squared:         </th> <td>   0.013</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.009</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   3.204</td>
</tr>
<tr>
  <th>Date:</th>             <td>Wed, 06 Dec 2017</td> <th>  Prob (F-statistic):</th>  <td>0.00702</td>
</tr>
<tr>
  <th>Time:</th>                 <td>13:46:04</td>     <th>  Log-Likelihood:    </th> <td> -3166.5</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>  1257</td>      <th>  AIC:               </th> <td>   6345.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  1251</td>      <th>  BIC:               </th> <td>   6376.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     5</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>     <td>    9.6438</td> <td>    0.097</td> <td>   99.900</td> <td> 0.000</td> <td>    9.454</td> <td>    9.833</td>
</tr>
<tr>
  <th>top_0_10</th>  <td>    0.1075</td> <td>    0.274</td> <td>    0.392</td> <td> 0.695</td> <td>   -0.431</td> <td>    0.646</td>
</tr>
<tr>
  <th>top_10_20</th> <td>    0.2819</td> <td>    0.358</td> <td>    0.788</td> <td> 0.431</td> <td>   -0.420</td> <td>    0.984</td>
</tr>
<tr>
  <th>top_20_30</th> <td>   -0.1965</td> <td>    0.280</td> <td>   -0.702</td> <td> 0.483</td> <td>   -0.745</td> <td>    0.352</td>
</tr>
<tr>
  <th>top_30_40</th> <td>    0.7624</td> <td>    0.273</td> <td>    2.789</td> <td> 0.005</td> <td>    0.226</td> <td>    1.299</td>
</tr>
<tr>
  <th>top_40_50</th> <td>    0.7486</td> <td>    0.321</td> <td>    2.330</td> <td> 0.020</td> <td>    0.118</td> <td>    1.379</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>79.718</td> <th>  Durbin-Watson:     </th> <td>   1.945</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  83.686</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.595</td> <th>  Prob(JB):          </th> <td>6.73e-19</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.574</td> <th>  Cond. No.          </th> <td>    4.32</td>
</tr>
</table>



###  Part(d) Multilinear regression model with genre predictors

In our prelimnary EDA anlaysis, we also believed that genres would be a good predictor for playlist followers. Therefore, we fit a regression model with only genre predictors. Here，each predictors is a categorical variable indicating whether the playlist belongs to a specified genre.

When we use all of genre predictors, we see that training r2 score is 0.59 and test r2 score is -3.785. This again is the result of overfitting since we have 865 genre columns. Therefore, we fit another regression model with only significant genre predictors from the full genre regression model. This time, we have a training r2 score of 0.009 and test r2 score of 0.007. The number of significant genre predictors is 54.

Therefore, a subset of genre predictors could still be important and should be considered for building future models.







```python
print("With all genre predictors,R2 score for training set: {}".format(r2_score(y_train,results4.predict(X5))))
print("With all genre predictors,R2 score for test set: {}".format(r2_score(y_test,results4.predict(X6))))
print("Number of genre predictors:{}".format(len(genre_columns)))
print("Significant genre predictors in regression:\n{}".format(results4.pvalues[results4.pvalues < 0.05]))
```


    With all genre predictors,R2 score for training set: 0.5974648743813241
    With all genre predictors,R2 score for test set: -3.7847086814421367
    Number of genre predictors:865
    Significant genre predictors in regression:
    const                              1.678665e-87
     'alternative metal'               4.049398e-02
     'brooklyn indie'                  4.809165e-02
     'christian punk'                  1.321873e-02
     'classic rock'                    6.111137e-03
     'country christmas'               2.945994e-02
     'dirty south rap'                 4.232302e-02
     'dixieland'                       1.657556e-02
     'ectofolk'                        3.147615e-02
     'filter house'                    6.523992e-03
     'garage punk'                     4.496653e-02
     'garage rock'                     1.192386e-02
     'hauntology'                      3.862767e-02
     'indie garage rock'               1.106413e-02
     'indie psych-pop'                 4.345163e-02
     'industrial rock'                 3.083634e-02
     'industrial'                      4.519249e-02
     'jangle pop'                      2.091888e-02
     'kraut rock'                      1.918933e-02
     'memphis hip hop'                 4.410499e-02
     'metal'                           2.472603e-02
     'modern blues'                    4.417157e-02
     'new orleans jazz'                2.401043e-02
     'outsider'                        1.112002e-02
     'pop punk'                        2.095434e-02
     'post-teen pop'                   4.203491e-02
     'power metal'                     3.328351e-02
     'power pop'                       2.371500e-02
     'progressive electro house'       2.438308e-02
     'progressive house'               3.076225e-02
     'progressive uplifting trance'    2.805253e-02
     'pub rock'                        3.587669e-02
     'rap metal'                       1.668566e-02
     'retro electro'                   6.837326e-03
     'retro metal'                     4.648427e-02
     'shibuya-kei'                     2.839880e-02
     'soft rock'                       6.490938e-03
     'soul christmas'                  4.282753e-02
     'stomp and holler'                3.701798e-03
     'traditional folk'                2.421117e-02
    "children's christmas"             1.072646e-02
    'abstractro'                       4.026550e-02
    'afrobeat'                         1.706527e-02
    'alternative americana'            1.569490e-02
    'alternative rock'                 2.427948e-02
    'art rock'                         4.098453e-02
    'blues-rock'                       5.686620e-03
    'country gospel'                   1.407087e-02
    'dance-punk'                       1.954791e-03
    'deep new americana'               1.595271e-02
    'escape room'                      2.422625e-02
    'g funk'                           5.312503e-03
    'metalcore'                        1.137579e-02
    'neo-synthpop'                     4.452120e-02
    'psychedelic doom'                 4.648427e-02
    'speed garage'                     2.949958e-03
    dtype: float64








```python
print("With significant genre predictor,R2 score for training set: {}".format(r2_score(y_train,results9.predict(X9))))
print("With significant genre predictor,R2 score for test set: {}".format(r2_score(y_test,results9.predict(X10))))
print("Number of genre predictors:{}".format(len(sig_genre)))
```


    With significant genre predictor,R2 score for training set: 0.09225710866655823
    With significant genre predictor,R2 score for test set: 0.007097541941422314
    Number of genre predictors:54


### Part (e) Multilinear regression with only siginifcant predictors from part (a)

Previously in part(a), our model includes all predictors and thus tend towards overfitting. In part(e), we fit a model with siginificant predictors from model in part(a) to reduce overfitting. We have a total of 49 predictors (cut down from 949 originally).

We see that test r2 score goes up to 0.085, which is the best r2 score so far. This indicates that our previous model indeed suffer from overfitting. We should work on choosing a subset of original predictors as predictors for furture models to improve prediction power.



```python
print("Significant predictors from regression in part(a):\n{}".format(results.pvalues[results.pvalues < 0.05]))

```


    Significant predictors from regression in part(a):
    instrumentalness_mean              0.010220
    liveness_mean                      0.001904
    loudness_std                       0.038317
    speech_mean                        0.021664
    time_std                           0.005990
    valence_mean                       0.000028
     'bass music'                      0.006956
     'big band'                        0.015642
     'christian punk'                  0.042034
     'country gospel'                  0.020285
     'crunk'                           0.026326
     'deep house'                      0.048536
     'dubstep'                         0.014862
     'ectofolk'                        0.016383
     'electro house'                   0.004962
     'escape room'                     0.043086
     'experimental'                    0.021957
     'filter house'                    0.032306
     'garage rock'                     0.011979
     'latin pop'                       0.031294
     'modern blues'                    0.022673
     'modern country rock'             0.033201
     'new orleans jazz'                0.007798
     'pop emo'                         0.041783
     'pop punk'                        0.016774
     'progressive electro house'       0.049407
     'progressive house'               0.044417
     'progressive uplifting trance'    0.007200
     'traditional folk'                0.039684
     'trip hop'                        0.044841
    'alternative rock'                 0.031086
    'austindie'                        0.007331
    'blues-rock'                       0.010399
    'canadian metal'                   0.029284
    'chillhop'                         0.002849
    'columbus ohio indie'              0.031607
    'dance-punk'                       0.032371
    'deep new americana'               0.034513
    'edm'                              0.014690
    'g funk'                           0.000820
    'indie punk'                       0.039368
    'pop'                              0.048267
    'progressive post-hardcore'        0.005071
    'speed garage'                     0.002887
    Radiohead                          0.000706
    Str_Best                           0.000583
    Str_Acoustic                       0.000383
    Str_2000s                          0.000014
    dance_liveness_std                 0.020839
    dtype: float64








```python
print("With only the significant predictors in part(a),R2 score for test set: {}".format(r2_score(y_train, results5.predict(X7))))
print("With only the significant predictors in part(a),R2 score for test set: {}".format(r2_test_e))
results5.summary()
```


    With only the significant predictors in part(a),R2 score for test set: 0.22290282653744364
    With only the significant predictors in part(a),R2 score for test set: 0.0825894720034378





<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>Followers</td>    <th>  R-squared:         </th> <td>   0.223</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.191</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   7.066</td>
</tr>
<tr>
  <th>Date:</th>             <td>Wed, 06 Dec 2017</td> <th>  Prob (F-statistic):</th> <td>1.25e-39</td>
</tr>
<tr>
  <th>Time:</th>                 <td>14:06:25</td>     <th>  Log-Likelihood:    </th> <td> -3016.0</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>  1257</td>      <th>  AIC:               </th> <td>   6132.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  1207</td>      <th>  BIC:               </th> <td>   6389.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>    49</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
                 <td></td>                    <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>                           <td>   15.2241</td> <td>    0.950</td> <td>   16.027</td> <td> 0.000</td> <td>   13.361</td> <td>   17.088</td>
</tr>
<tr>
  <th>instrumentalness_mean</th>           <td>   -3.1447</td> <td>    1.089</td> <td>   -2.887</td> <td> 0.004</td> <td>   -5.282</td> <td>   -1.008</td>
</tr>
<tr>
  <th>liveness_mean</th>                   <td>  -12.1780</td> <td>    2.358</td> <td>   -5.164</td> <td> 0.000</td> <td>  -16.804</td> <td>   -7.552</td>
</tr>
<tr>
  <th>loudness_std</th>                    <td>    3.3418</td> <td>    1.299</td> <td>    2.572</td> <td> 0.010</td> <td>    0.793</td> <td>    5.891</td>
</tr>
<tr>
  <th>speech_mean</th>                     <td>   -2.5326</td> <td>    0.971</td> <td>   -2.608</td> <td> 0.009</td> <td>   -4.438</td> <td>   -0.627</td>
</tr>
<tr>
  <th>time_std</th>                        <td> -1.25e-08</td> <td> 3.99e-09</td> <td>   -3.131</td> <td> 0.002</td> <td>-2.03e-08</td> <td>-4.67e-09</td>
</tr>
<tr>
  <th>valence_mean</th>                    <td>   -7.3241</td> <td>    1.413</td> <td>   -5.183</td> <td> 0.000</td> <td>  -10.096</td> <td>   -4.552</td>
</tr>
<tr>
  <th> 'bass music'</th>                   <td>   -0.4514</td> <td>    0.405</td> <td>   -1.116</td> <td> 0.265</td> <td>   -1.245</td> <td>    0.342</td>
</tr>
<tr>
  <th> 'big band'</th>                     <td>    0.2267</td> <td>    0.316</td> <td>    0.718</td> <td> 0.473</td> <td>   -0.393</td> <td>    0.847</td>
</tr>
<tr>
  <th> 'christian punk'</th>               <td>   -1.1516</td> <td>    0.685</td> <td>   -1.682</td> <td> 0.093</td> <td>   -2.495</td> <td>    0.192</td>
</tr>
<tr>
  <th> 'country gospel'</th>               <td>   -0.1673</td> <td>    0.326</td> <td>   -0.514</td> <td> 0.607</td> <td>   -0.806</td> <td>    0.471</td>
</tr>
<tr>
  <th> 'crunk'</th>                        <td>    1.4932</td> <td>    0.785</td> <td>    1.901</td> <td> 0.058</td> <td>   -0.048</td> <td>    3.034</td>
</tr>
<tr>
  <th> 'deep house'</th>                   <td>    0.3671</td> <td>    0.299</td> <td>    1.226</td> <td> 0.220</td> <td>   -0.220</td> <td>    0.954</td>
</tr>
<tr>
  <th> 'dubstep'</th>                      <td>    0.6460</td> <td>    0.540</td> <td>    1.196</td> <td> 0.232</td> <td>   -0.414</td> <td>    1.706</td>
</tr>
<tr>
  <th> 'ectofolk'</th>                     <td>   -2.1576</td> <td>    1.148</td> <td>   -1.880</td> <td> 0.060</td> <td>   -4.409</td> <td>    0.094</td>
</tr>
<tr>
  <th> 'electro house'</th>                <td>   -0.2006</td> <td>    0.233</td> <td>   -0.859</td> <td> 0.390</td> <td>   -0.659</td> <td>    0.257</td>
</tr>
<tr>
  <th> 'escape room'</th>                  <td>    0.6030</td> <td>    0.182</td> <td>    3.308</td> <td> 0.001</td> <td>    0.245</td> <td>    0.961</td>
</tr>
<tr>
  <th> 'experimental'</th>                 <td>    0.1237</td> <td>    0.213</td> <td>    0.580</td> <td> 0.562</td> <td>   -0.295</td> <td>    0.543</td>
</tr>
<tr>
  <th> 'filter house'</th>                 <td>    1.0122</td> <td>    0.338</td> <td>    2.993</td> <td> 0.003</td> <td>    0.349</td> <td>    1.676</td>
</tr>
<tr>
  <th> 'garage rock'</th>                  <td>    0.2319</td> <td>    0.201</td> <td>    1.154</td> <td> 0.249</td> <td>   -0.163</td> <td>    0.626</td>
</tr>
<tr>
  <th> 'latin pop'</th>                    <td>    0.4265</td> <td>    0.220</td> <td>    1.942</td> <td> 0.052</td> <td>   -0.004</td> <td>    0.857</td>
</tr>
<tr>
  <th> 'modern blues'</th>                 <td>    0.5259</td> <td>    0.187</td> <td>    2.809</td> <td> 0.005</td> <td>    0.159</td> <td>    0.893</td>
</tr>
<tr>
  <th> 'modern country rock'</th>          <td>    0.4418</td> <td>    0.190</td> <td>    2.321</td> <td> 0.020</td> <td>    0.068</td> <td>    0.815</td>
</tr>
<tr>
  <th> 'new orleans jazz'</th>             <td>   -0.4096</td> <td>    0.363</td> <td>   -1.128</td> <td> 0.260</td> <td>   -1.122</td> <td>    0.303</td>
</tr>
<tr>
  <th> 'pop emo'</th>                      <td>   -0.1501</td> <td>    0.209</td> <td>   -0.719</td> <td> 0.472</td> <td>   -0.560</td> <td>    0.259</td>
</tr>
<tr>
  <th> 'pop punk'</th>                     <td>    0.4633</td> <td>    0.185</td> <td>    2.504</td> <td> 0.012</td> <td>    0.100</td> <td>    0.826</td>
</tr>
<tr>
  <th> 'progressive electro house'</th>    <td>    0.5527</td> <td>    0.332</td> <td>    1.666</td> <td> 0.096</td> <td>   -0.098</td> <td>    1.203</td>
</tr>
<tr>
  <th> 'progressive house'</th>            <td>   -0.3463</td> <td>    0.331</td> <td>   -1.048</td> <td> 0.295</td> <td>   -0.995</td> <td>    0.302</td>
</tr>
<tr>
  <th> 'progressive uplifting trance'</th> <td>   -0.3538</td> <td>    0.817</td> <td>   -0.433</td> <td> 0.665</td> <td>   -1.957</td> <td>    1.250</td>
</tr>
<tr>
  <th> 'traditional folk'</th>             <td>   -0.0593</td> <td>    0.365</td> <td>   -0.163</td> <td> 0.871</td> <td>   -0.776</td> <td>    0.657</td>
</tr>
<tr>
  <th> 'trip hop'</th>                     <td>    0.9485</td> <td>    0.391</td> <td>    2.429</td> <td> 0.015</td> <td>    0.182</td> <td>    1.715</td>
</tr>
<tr>
  <th>'alternative rock'</th>              <td>    0.1924</td> <td>    0.197</td> <td>    0.974</td> <td> 0.330</td> <td>   -0.195</td> <td>    0.580</td>
</tr>
<tr>
  <th>'austindie'</th>                     <td>    0.6574</td> <td>    0.505</td> <td>    1.301</td> <td> 0.194</td> <td>   -0.334</td> <td>    1.649</td>
</tr>
<tr>
  <th>'blues-rock'</th>                    <td>   -0.3016</td> <td>    0.243</td> <td>   -1.241</td> <td> 0.215</td> <td>   -0.778</td> <td>    0.175</td>
</tr>
<tr>
  <th>'canadian metal'</th>                <td>    0.7083</td> <td>    0.857</td> <td>    0.827</td> <td> 0.409</td> <td>   -0.972</td> <td>    2.389</td>
</tr>
<tr>
  <th>'chillhop'</th>                      <td>   -0.6586</td> <td>    0.847</td> <td>   -0.778</td> <td> 0.437</td> <td>   -2.320</td> <td>    1.003</td>
</tr>
<tr>
  <th>'columbus ohio indie'</th>           <td>   -0.8179</td> <td>    0.535</td> <td>   -1.528</td> <td> 0.127</td> <td>   -1.868</td> <td>    0.233</td>
</tr>
<tr>
  <th>'dance-punk'</th>                    <td>   -0.3826</td> <td>    0.571</td> <td>   -0.670</td> <td> 0.503</td> <td>   -1.503</td> <td>    0.738</td>
</tr>
<tr>
  <th>'deep new americana'</th>            <td>    0.0373</td> <td>    0.218</td> <td>    0.171</td> <td> 0.864</td> <td>   -0.391</td> <td>    0.465</td>
</tr>
<tr>
  <th>'edm'</th>                           <td>    0.6196</td> <td>    0.301</td> <td>    2.062</td> <td> 0.039</td> <td>    0.030</td> <td>    1.209</td>
</tr>
<tr>
  <th>'g funk'</th>                        <td>   -0.5112</td> <td>    0.420</td> <td>   -1.217</td> <td> 0.224</td> <td>   -1.335</td> <td>    0.313</td>
</tr>
<tr>
  <th>'indie punk'</th>                    <td>    0.0738</td> <td>    0.430</td> <td>    0.172</td> <td> 0.864</td> <td>   -0.769</td> <td>    0.917</td>
</tr>
<tr>
  <th>'pop'</th>                           <td>    0.5996</td> <td>    0.276</td> <td>    2.175</td> <td> 0.030</td> <td>    0.059</td> <td>    1.140</td>
</tr>
<tr>
  <th>'progressive post-hardcore'</th>     <td>    0.7197</td> <td>    0.449</td> <td>    1.603</td> <td> 0.109</td> <td>   -0.161</td> <td>    1.600</td>
</tr>
<tr>
  <th>'speed garage'</th>                  <td>   -0.3341</td> <td>    0.646</td> <td>   -0.517</td> <td> 0.605</td> <td>   -1.602</td> <td>    0.933</td>
</tr>
<tr>
  <th>Radiohead</th>                       <td>   -0.8771</td> <td>    0.821</td> <td>   -1.068</td> <td> 0.286</td> <td>   -2.489</td> <td>    0.735</td>
</tr>
<tr>
  <th>Str_Best</th>                        <td>   -1.2647</td> <td>    0.330</td> <td>   -3.837</td> <td> 0.000</td> <td>   -1.911</td> <td>   -0.618</td>
</tr>
<tr>
  <th>Str_Acoustic</th>                    <td>    3.1248</td> <td>    0.617</td> <td>    5.064</td> <td> 0.000</td> <td>    1.914</td> <td>    4.335</td>
</tr>
<tr>
  <th>Str_2000s</th>                       <td>   -3.0511</td> <td>    0.381</td> <td>   -8.006</td> <td> 0.000</td> <td>   -3.799</td> <td>   -2.303</td>
</tr>
<tr>
  <th>dance_liveness_std</th>              <td>   -0.1096</td> <td>    0.030</td> <td>   -3.650</td> <td> 0.000</td> <td>   -0.169</td> <td>   -0.051</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>31.737</td> <th>  Durbin-Watson:     </th> <td>   1.975</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  33.675</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.395</td> <th>  Prob(JB):          </th> <td>4.87e-08</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.868</td> <th>  Cond. No.          </th> <td>6.39e+08</td>
</tr>
</table>



### Part (f) Bootstrapping for 10% Predictors

From previous parts, we observe that a smaller subset of original predictors may do a lot better in terms of test set prediction. Therefore, in part (f), we randomly choose 10% of predictors and fit a regression model. We do 500 iterations and record corresponding r2 test core and the associated predictors.

We achieve a r2 test score of 0.21. However,since we are just randomly choosing predictors, this result could come from chance alone and may not be very robust.











```python
print("After bootstrapping for 10% of predictors, the best R2 score for test set: {}".format(r2_test[findLargest(r2_test)]))
print("\n")
print('The assocaited predictors are:{}'.format(pred[findLargest(r2_test)]))

```


    After bootstrapping for 10% of predictors, the best R2 score for test set: 0.21855216986291626
    
    
    The assocaited predictors are:['acousticness_mean', 'liveness_std', 'loudness_mean', 'valence_mean', 'followers_std', 'top_0_10', " 'ambient'", " 'bebop'", " 'bluegrass'", " 'bow pop'", " 'chillhop'", " 'classic rock'", " 'classical piano'", " 'contemporary jazz'", " 'country dawn'", " 'deep talent show'", " 'desi hip hop'", " 'ethereal wave'", " 'freestyle'", " 'future garage'", " 'garage punk blues'", " 'heavy alternative'", " 'indie dream pop'", " 'indie garage rock'", " 'indie psych-rock'", " 'intelligent dance music'", " 'latin metal'", " 'metropopolis'", " 'modern uplift'", " 'motown'", " 'nashville sound'", " 'neo-industrial rock'", " 'new americana'", " 'noise rock'", " 'nu gaze'", " 'ok indie'", " 'piano blues'", " 'pop christmas'", " 'pop'", " 'power metal'", " 'progressive house'", " 'progressive post-hardcore'", " 'retro electro'", " 'scorecore'", " 'sludge metal'", " 'southern soul'", " 'swedish indie rock'", " 'teen pop'", " 'trance'", " 'triangle indie'", " 'tropical house'", " 'uk post-punk'", " 'vocal house'", '"children\'s christmas"', "'alternative rock'", "'australian alternative rock'", "'baroque'", "'blackgaze'", "'canadian country'", "'canadian pop'", "'cantautor'", "'chillhop'", "'deep contemporary country'", "'desert blues'", "'djent'", "'dreamo'", "'drone'", "'east coast hip hop'", "'house'", "'indie pop'", "'indie poptimism'", "'indie psych-rock'", "'mellow gold'", "'minimal techno'", "'ninja'", "'skate punk'", "'vancouver indie'", "'wrestling'", 'Rihanna', 'Chance The Rapper', 'Otis Redding', 'SZA', 'Birdy', 'Commodores', 'Str_Party', 'Str_Acoustic', 'Str_2000s', 'acoustic_key_std']


### Part(g) PCA

PCA (Principle Component Analysis) is another way to reduce the number of predictors. Each component is a linear combination of all 949 orginal predictors. The components are ordered in such a wa so that the amount of captured observed variance descends. In part(g), we implement PCA here. We try different numbers of PCA components from 1 to 100 and choose the optimal number of PCA components according to test r2 score.

We acheive the best r2 test score of 0.13 with 30 PCA components. Although we gain a higher test r2 score and also have less predictors, we lose a lot interpretability. We cannot pinpoint how change in one predictor will change the response varibable because each component is a linear combination of all original columns.








    0.13027619329603601





```python
print("After PCA, the best R2 score for test set: {}".format(r2_test_pca[findLargest(r2_test_pca)]))
print("The optimal number of components is: {}".format(findLargest(r2_test_pca)+1))
```


    After PCA, the best R2 score for test set: 0.130276193296036
    The optimal number of components is: 30


### Part(h) Lasso and Ridge

Lasso and Ridge regularizations are also methods to penalize overly complex models. To penalize coefficeints that has large magnitude, Lasso and ridge include the magnitude of the cofficients in the loss functions. Specific, Lasso includes the sum of the absolute values of coefficients multiplied by a constant lambda. Ridge includes the sum of the square of coefficients multiplied by a constant lambda.In part (h), we fit Ridge and Lasso with cross validation and with lamda ranging from 1e^-5 to 10^5.

With Lasso, the test r2 score is 0.109. With Ridge, the test r2 score is 0.112.





    With lasso, R2 score for test set: 0.1094129599174094






    With ridge, R2 score for test set: 0.11234514421712671




```python
print("With lasso, R2 score for test set: {}".format(lasso_regression.score(x_test,y_test)))
print("With ridge, R2 score for test set: {}".format(ridge_regression.score(x_test,y_test)))
```


    With lasso, R2 score for test set: 0.1094129599174094
    With ridge, R2 score for test set: 0.11234514421712671


## Summary of Baseline Models

Since we have 949 predictors in our original data, we encounter the problem of overfitting when constructing regression models. Therefore, for basline models, we focus on selecting a subgroup of predictors that perform well in terms of r2 test score. By using only the significant predictors from the full model, test r2 score improves a lot (from -2.7 to 0.8). PCA gives us r2 score of 0.13. Lasso gives up 0.109 and Ridge gives us 0.11. In addition, we should note that iteratively choosing 10% of predictors gives us the best baseline r2 metric (0.21). However, this model is not robust and tend toward overfitting since we do not methodologically choose the predictors. Still, we could use 0.21 as a reference when evaluating our future models. In summary, the most robust and predictive baseline model is PCA with r2 socre of 0.13. But we may use 0.21 as baseline r2_test metric for the furture. In addition, we should consider including top 30 artists predictors since they alone gives us 0.03 r2 test metric. On the other hand, we should carefully select which genre predictors to include since there are 865 of them and would lead to overfitting.