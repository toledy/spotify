---
title: Baseline Models
notebook: baseline_models.ipynb
nav_include: 3
---

## Contents
{:.no_toc}
*  
{: toc}



```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LassoCV,RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 200)
import math
```


    /Users/liutianhan/anaconda/lib/python3.6/site-packages/statsmodels/compat/pandas.py:56: FutureWarning: The pandas.core.datetools module is deprecated and will be removed in a future version. Please use the pandas.tseries module instead.
      from pandas.core import datetools


## Read In Data

In this section, our goal is to generate processed training and test sets. First, we read in our data from csv file. We split the data into training and test, impute missing values through the mean and finally we transform response variables using log.



```python
data = pd.read_csv('Final_Dataframe.csv')
data = data.drop(["Unnamed: 0"], axis=1)
```




```python
data.head()
```





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
      <th>Unnamed: 0</th>
      <th>acousticness_mean</th>
      <th>acousticness_std</th>
      <th>dance_mean</th>
      <th>dance_std</th>
      <th>energy_mean</th>
      <th>energy_std</th>
      <th>instrumentalness_mean</th>
      <th>instrumentalness_std</th>
      <th>key_mean</th>
      <th>key_std</th>
      <th>liveness_mean</th>
      <th>liveness_std</th>
      <th>loudness_mean</th>
      <th>loudness_std</th>
      <th>mode_mean</th>
      <th>mode_std</th>
      <th>speech_mean</th>
      <th>speech_std</th>
      <th>tempo_mean</th>
      <th>tempo_std</th>
      <th>time_mean</th>
      <th>time_std</th>
      <th>valence_mean</th>
      <th>valence_std</th>
      <th>Followers</th>
      <th>followers_mean</th>
      <th>followers_std</th>
      <th>popularity_mean</th>
      <th>popularity_std</th>
      <th>top_0_10</th>
      <th>top_10_20</th>
      <th>top_20_30</th>
      <th>top_30_40</th>
      <th>top_40_50</th>
      <th>'acid house'</th>
      <th>'album rock'</th>
      <th>'alternative country'</th>
      <th>'alternative dance'</th>
      <th>'alternative metal'</th>
      <th>'alternative pop'</th>
      <th>'alternative rock'</th>
      <th>'alternative roots rock'</th>
      <th>'ambient'</th>
      <th>'anthem emo'</th>
      <th>'anthem worship'</th>
      <th>'anti-folk'</th>
      <th>'art rock'</th>
      <th>'athens indie'</th>
      <th>'austindie'</th>
      <th>'australian alternative rock'</th>
      <th>'australian dance'</th>
      <th>'australian hip hop'</th>
      <th>'australian pop'</th>
      <th>'avant-garde'</th>
      <th>'azonto'</th>
      <th>'azontobeats'</th>
      <th>'bass music'</th>
      <th>'bay area indie'</th>
      <th>'bebop'</th>
      <th>'big band'</th>
      <th>'big beat'</th>
      <th>'big room'</th>
      <th>'bluegrass'</th>
      <th>'blues'</th>
      <th>'blues-rock'</th>
      <th>'boogie-woogie'</th>
      <th>'bossa nova'</th>
      <th>'boston rock'</th>
      <th>'bow pop'</th>
      <th>'breakbeat'</th>
      <th>'brill building pop'</th>
      <th>'british alternative rock'</th>
      <th>'british blues'</th>
      <th>'british folk'</th>
      <th>'british indie rock'</th>
      <th>'british invasion'</th>
      <th>'britpop'</th>
      <th>'brooklyn indie'</th>
      <th>'brostep'</th>
      <th>'bubblegum pop'</th>
      <th>'c86'</th>
      <th>'cabaret'</th>
      <th>'canadian indie'</th>
      <th>'canadian metal'</th>
      <th>'canadian pop'</th>
      <th>'candy pop'</th>
      <th>'canterbury scene'</th>
      <th>'catstep'</th>
      <th>'ccm'</th>
      <th>'cello'</th>
      <th>'celtic rock'</th>
      <th>'chamber pop'</th>
      <th>'chamber psych'</th>
      <th>'chaotic hardcore'</th>
      <th>'chicago blues'</th>
      <th>'chicago house'</th>
      <th>'chicago indie'</th>
      <th>'chicago soul'</th>
      <th>'chillhop'</th>
      <th>...</th>
      <th>'nu age'</th>
      <th>'nu metal'</th>
      <th>'permanent wave'</th>
      <th>'pixie'</th>
      <th>'polka'</th>
      <th>'pop emo'</th>
      <th>'pop rap'</th>
      <th>'pop rock'</th>
      <th>'pop'</th>
      <th>'post-screamo'</th>
      <th>'post-teen pop'</th>
      <th>'progressive deathcore'</th>
      <th>'progressive post-hardcore'</th>
      <th>'psychedelic doom'</th>
      <th>'punk'</th>
      <th>'quebecois'</th>
      <th>'rap'</th>
      <th>'redneck'</th>
      <th>'reggae fusion'</th>
      <th>'reggae'</th>
      <th>'reggaeton'</th>
      <th>'relaxative'</th>
      <th>'scorecore'</th>
      <th>'sheffield indie'</th>
      <th>'shimmer pop'</th>
      <th>'shiver pop'</th>
      <th>'singer-songwriter'</th>
      <th>'skate punk'</th>
      <th>'slow core'</th>
      <th>'slow game'</th>
      <th>'spanish pop'</th>
      <th>'speed garage'</th>
      <th>'talent show'</th>
      <th>'traditional country'</th>
      <th>'trap latino'</th>
      <th>'trap music'</th>
      <th>'trip hop'</th>
      <th>'tropical house'</th>
      <th>'uk drill'</th>
      <th>'uk hip hop'</th>
      <th>'underground hip hop'</th>
      <th>'vancouver indie'</th>
      <th>'vapor soul'</th>
      <th>'vegas indie'</th>
      <th>'violin'</th>
      <th>'viral pop'</th>
      <th>'west coast trap'</th>
      <th>'wrestling'</th>
      <th>'no_genre'</th>
      <th>Lil Wayne</th>
      <th>Van Morrison</th>
      <th>Galantis</th>
      <th>Wiz Khalifa</th>
      <th>Rihanna</th>
      <th>Post Malone</th>
      <th>Axwell /\ Ingrosso</th>
      <th>Young Thug</th>
      <th>JAY Z</th>
      <th>A$AP Rocky</th>
      <th>Yo Gotti</th>
      <th>Chance The Rapper</th>
      <th>Led Zeppelin</th>
      <th>Otis Redding</th>
      <th>21 Savage</th>
      <th>Deorro</th>
      <th>Elton John</th>
      <th>SZA</th>
      <th>Ty Dolla $ign</th>
      <th>Ryan Adams</th>
      <th>Birdy</th>
      <th>Miguel</th>
      <th>Niall Horan</th>
      <th>Ellie Goulding</th>
      <th>Commodores</th>
      <th>Radiohead</th>
      <th>SYML</th>
      <th>First Aid Kit</th>
      <th>Lord Huron</th>
      <th>Str_Best</th>
      <th>Str_Workout</th>
      <th>Str_Party</th>
      <th>Str_Chill</th>
      <th>Str_Acoustic</th>
      <th>Str_2000s</th>
      <th>Str_1990s</th>
      <th>Str_1980s</th>
      <th>Str_1970s</th>
      <th>Str_1960s</th>
      <th>house_acousticness_mean</th>
      <th>hip hop_acousticness_std</th>
      <th>pop_liveness_std</th>
      <th>dance_liveness_std</th>
      <th>r&amp;b_acousticness_std</th>
      <th>rap_energy_std</th>
      <th>rap_key_std</th>
      <th>acoustic_acousticness_std</th>
      <th>acoustic_acousticness_mean</th>
      <th>acoustic_energy_std</th>
      <th>acoustic_key_std</th>
      <th>soul_acousticness_std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.641282</td>
      <td>3.058644</td>
      <td>0.467911</td>
      <td>4.148403</td>
      <td>0.275940</td>
      <td>4.428289</td>
      <td>0.119650</td>
      <td>3.608691</td>
      <td>0.275940</td>
      <td>4.428289</td>
      <td>0.199440</td>
      <td>6.131128</td>
      <td>-18.000646</td>
      <td>0.111308</td>
      <td>0.630769</td>
      <td>2.056123</td>
      <td>0.383051</td>
      <td>2.479147</td>
      <td>101.045969</td>
      <td>0.019284</td>
      <td>3.338462</td>
      <td>0.643502</td>
      <td>0.319263</td>
      <td>4.061163</td>
      <td>24.0</td>
      <td>366.624695</td>
      <td>2.736285e-06</td>
      <td>42.833333</td>
      <td>0.051084</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>3.058644</td>
      <td>6.131128</td>
      <td>6.131128</td>
      <td>3.058644</td>
      <td>4.428289</td>
      <td>4.428289</td>
      <td>3.058644</td>
      <td>0.641282</td>
      <td>4.428289</td>
      <td>4.428289</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.278816</td>
      <td>3.805912</td>
      <td>0.634392</td>
      <td>7.129091</td>
      <td>0.596000</td>
      <td>5.991548</td>
      <td>0.192559</td>
      <td>2.928603</td>
      <td>0.596000</td>
      <td>5.991548</td>
      <td>0.164490</td>
      <td>7.768494</td>
      <td>-9.525804</td>
      <td>0.280768</td>
      <td>0.509804</td>
      <td>1.980676</td>
      <td>0.082210</td>
      <td>7.627447</td>
      <td>122.768255</td>
      <td>0.035441</td>
      <td>4.000000</td>
      <td>5.000000</td>
      <td>0.656235</td>
      <td>4.076658</td>
      <td>330.0</td>
      <td>321.435189</td>
      <td>3.011912e-06</td>
      <td>48.903226</td>
      <td>0.066535</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.278816</td>
      <td>3.805912</td>
      <td>7.768494</td>
      <td>7.768494</td>
      <td>3.805912</td>
      <td>5.991548</td>
      <td>5.991548</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.805912</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0.228810</td>
      <td>3.977396</td>
      <td>0.600400</td>
      <td>5.592818</td>
      <td>0.612200</td>
      <td>5.196620</td>
      <td>0.179571</td>
      <td>2.970854</td>
      <td>0.612200</td>
      <td>5.196620</td>
      <td>0.192563</td>
      <td>7.704437</td>
      <td>-7.845333</td>
      <td>0.307409</td>
      <td>0.666667</td>
      <td>2.085665</td>
      <td>0.052150</td>
      <td>38.557531</td>
      <td>114.439167</td>
      <td>0.045459</td>
      <td>4.000000</td>
      <td>3.807886</td>
      <td>0.481787</td>
      <td>3.980915</td>
      <td>73.0</td>
      <td>752.870879</td>
      <td>7.006194e-07</td>
      <td>60.280000</td>
      <td>0.064466</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.228810</td>
      <td>0.000000</td>
      <td>7.704437</td>
      <td>7.704437</td>
      <td>3.977396</td>
      <td>5.196620</td>
      <td>5.196620</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.977396</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0.394114</td>
      <td>2.758064</td>
      <td>0.599424</td>
      <td>6.611299</td>
      <td>0.541097</td>
      <td>3.451792</td>
      <td>0.203059</td>
      <td>3.008687</td>
      <td>0.541097</td>
      <td>3.451792</td>
      <td>0.211488</td>
      <td>5.238347</td>
      <td>-9.764303</td>
      <td>0.181115</td>
      <td>0.606061</td>
      <td>2.015326</td>
      <td>0.106724</td>
      <td>8.893022</td>
      <td>110.134788</td>
      <td>0.039801</td>
      <td>4.000000</td>
      <td>2.828427</td>
      <td>0.511997</td>
      <td>4.112335</td>
      <td>6173.0</td>
      <td>447.025150</td>
      <td>3.385402e-06</td>
      <td>58.696970</td>
      <td>0.063990</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.394114</td>
      <td>2.758064</td>
      <td>5.238347</td>
      <td>5.238347</td>
      <td>2.758064</td>
      <td>3.451792</td>
      <td>3.451792</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0.194509</td>
      <td>3.591057</td>
      <td>0.531067</td>
      <td>6.666606</td>
      <td>0.759400</td>
      <td>4.003118</td>
      <td>0.115499</td>
      <td>3.875675</td>
      <td>0.759400</td>
      <td>4.003118</td>
      <td>0.234787</td>
      <td>6.989281</td>
      <td>-6.465367</td>
      <td>0.246666</td>
      <td>0.666667</td>
      <td>2.085665</td>
      <td>0.129720</td>
      <td>4.482143</td>
      <td>124.789500</td>
      <td>0.037045</td>
      <td>3.933333</td>
      <td>3.941537</td>
      <td>0.443407</td>
      <td>3.941298</td>
      <td>145.0</td>
      <td>472.497380</td>
      <td>2.033166e-06</td>
      <td>49.516129</td>
      <td>0.051309</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.194509</td>
      <td>3.591057</td>
      <td>6.989281</td>
      <td>6.989281</td>
      <td>3.591057</td>
      <td>4.003118</td>
      <td>4.003118</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.591057</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 950 columns</p>
</div>





```python
## train, test split##


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


## Baseline models 

In this section, our goal is to examine the baseline performance of simple models on the test set. We would use these baseline test set r2 score as a reference for building more complex models. The models included in this section are mostly multilinear regression models with different subset of predictors and possible polynomial/interaction terms. PCA and LASSO/RRIDGE are explored here as well.

### Part(a) Multilinear regression model with all predictors

The test r2 score for multilinear regression model with all predictors is -2.779. This is evidence suggesting that we might be overfitting our model with too many predictors. Therefore, going forward (part (e)), we would like to fit a regression model with only significant predictors in this model.



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

In our prelimnary EDA anlaysis, we believed that top artists would be a good predictor for the success of a playlist. Therefore, here we fit two models in part (b) and (c) that only include predictors including artists. In part (b), we use the presence of top 30 artists as predictors. As a note, for part (b), top artist are those who appear most often in playlists with 350,000+ followers. With more than 350,000 followers, a playlist will be in the top 20% in terms of followers. 

Our regression generates a test r2 score of 0.017, which is a lot better than -2.8 in part (a). Therefore, there is good reason to consider these predictors in future model building.



```python
top_30_artist_col=['Lil Wayne', 'Van Morrison', 'Galantis',
       'Wiz Khalifa', 'Rihanna', 'Post Malone', 'Axwell /\ Ingrosso',
       'Young Thug', 'JAY Z', 'A$AP Rocky', 'Yo Gotti', 'Chance The Rapper',
       'Led Zeppelin', 'Otis Redding', '21 Savage', 'Deorro', 'Elton John',
       'SZA', 'Ty Dolla $ign', 'Ryan Adams', 'Birdy', 'Miguel', 'Niall Horan',
       'Ellie Goulding', 'Commodores', 'Radiohead', 'SYML', 'First Aid Kit',
       'Lord Huron']

x_train_art=x_train[top_30_artist_col]
x_test_art=x_test[top_30_artist_col]

X1=sm.add_constant(x_train_art)
X2=sm.add_constant(x_test_art)
model2=sm.OLS(y_train,X1)
results2=model2.fit()
print("With top 30 artists,R2 score for test set: {}".format(r2_score(y_test,results2.predict(X2))))
#results2.summary()
```


    With tio 30 artists,R2 score for test set: 0.017373740567596885




```python
#significant artistis
results2.pvalues[results2.pvalues < 0.05].index
```





    Index(['const', 'Galantis', 'Post Malone', 'Yo Gotti', 'Ellie Goulding'], dtype='object')



###  part(c) Multilinear regression model with top artists counts predictors

Top artists are defined different for part (b) and part (c).Here, top artists are ranked by the total number of followers that playlists including this artist aggregate to. For part (c), the predictors are the number of top 10/10-20/20-30/30-40/40-50 artists that a playlist has.

We see that r2 test result is -0.03.



```python
x_train_art_count=x_train[top_artist_count_columns]
x_test_art_count=x_test[top_artist_count_columns]

X3=sm.add_constant(x_train_art_count)
X4=sm.add_constant(x_test_art_count)
model3=sm.OLS(y_train,X3)
results3=model3.fit()
print("With top artists count predictors,R2 score for test set: {}".format(r2_score(y_test,results3.predict(X4))))
results3.summary()
```


    With only the significant terms from the last model,R2 score for test set: -0.031235636069632644





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
  <th>Date:</th>             <td>Tue, 05 Dec 2017</td> <th>  Prob (F-statistic):</th>  <td>0.00702</td>
</tr>
<tr>
  <th>Time:</th>                 <td>00:19:12</td>     <th>  Log-Likelihood:    </th> <td> -3166.5</td>
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

We see that test r2 score is -3.785. This could be the result of overfitting since we have 865 genre columns. A subset of genre predictors could still be important and will be considered for building future models.



```python
x_train_genre=x_train[genre_columns]
x_test_genre=x_test[genre_columns]

X5=sm.add_constant(x_train_genre)
X6=sm.add_constant(x_test_genre)
model4=sm.OLS(y_train,X5)
results4=model4.fit()
print("With only the significant terms from the last model,R2 score for test set: {}".format(r2_score(y_test,results4.predict(X6))))
#results4.summary()
len(genre_columns)
```


    With only the significant terms from the last model,R2 score for test set: -3.7847086814421367





    865



### Part (e) Multilinear regression with only siginifcant predictors from part (a)

In part(e), we fit a model with siginificant predictors from model in part(a) to reduce overfitting. We have a total of 49 predictors (cut down from 949 originally).

We see that test r2 score goes up to 0.085, which is the best r2 score so far. This indicates that our previous model indeed suffer from overfitting. We should workin on choosing a subset of original predictors as predictors for furture models.



```python
sig_preds=results.pvalues[results.pvalues < 0.05].index
len(sig_preds)
sig_preds
```





    Index(['instrumentalness_mean', 'liveness_mean', 'loudness_std', 'speech_mean',
           'time_std', 'valence_mean', ' 'bass music'', ' 'big band'',
           ' 'christian punk'', ' 'country gospel'', ' 'crunk'', ' 'deep house'',
           ' 'dubstep'', ' 'ectofolk'', ' 'electro house'', ' 'escape room'',
           ' 'experimental'', ' 'filter house'', ' 'garage rock'', ' 'latin pop'',
           ' 'modern blues'', ' 'modern country rock'', ' 'new orleans jazz'',
           ' 'pop emo'', ' 'pop punk'', ' 'progressive electro house'',
           ' 'progressive house'', ' 'progressive uplifting trance'',
           ' 'traditional folk'', ' 'trip hop'', ''alternative rock'',
           ''austindie'', ''blues-rock'', ''canadian metal'', ''chillhop'',
           ''columbus ohio indie'', ''dance-punk'', ''deep new americana'',
           ''edm'', ''g funk'', ''indie punk'', ''pop'',
           ''progressive post-hardcore'', ''speed garage'', 'Radiohead',
           'Str_Best', 'Str_Acoustic', 'Str_2000s', 'dance_liveness_std'],
          dtype='object')





```python
#fit a multilinear regression model with significant predictors
x_train2 = x_train[sig_preds]
x_test2 = x_test[sig_preds]

X7=sm.add_constant(x_train2)
X8=sm.add_constant(x_test2)
model5=sm.OLS(y_train,X7)
results5=model5.fit()
r2_test_e=r2_score(y_test,results5.predict(X8))
print("With only the significant terms from the last model,R2 score for test set: {}".format(r2_test_e))
results5.summary()
```


    With only the significant terms from the last model,R2 score for test set: 0.0825894720034378





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
  <th>Date:</th>             <td>Tue, 05 Dec 2017</td> <th>  Prob (F-statistic):</th> <td>1.25e-39</td>
</tr>
<tr>
  <th>Time:</th>                 <td>10:54:47</td>     <th>  Log-Likelihood:    </th> <td> -3016.0</td>
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

In part (e), we observe that a smaller subset of original predictors may do a lot better in terms of test set prediction. Therefore, in part (f), we randomly choose 10% of predictors and fit a regression model. We do 500 iterations and record corresponding r2 test core and the associated predictors.

We achieve a r2 test score of 0.21. However,since we are just randomly choosing predictors, this result could come from chance alone and may not be very robust.



```python
##bootstrapping for10% predictors
r2_test=[]
pred=[]
for i in range(500):
    train_col=[]
    while len(train_col)==0:
        for ele in x_train.columns:
            u=np.random.uniform(0,1)
            if u>0.9:
                if ele!='Followers':
                    train_col.append(ele)
    pred.append(train_col)
    x_train1 = x_train[train_col]
    x_test1 = x_test[train_col]
    multi2 =LinearRegression(fit_intercept=True)# no need to add constant when doing it this way
    multi2.fit(x_train1, y_train)

    r2_test.append(multi2.score(x_test1, y_test))
```




```python
def findLargest(r2):
    largest=r2[0]
    count=0
    for i in range(len(r2)):
        if r2[i]>largest:
            largest=r2[i]
            count=i
    return count
```




```python
print("After bootstrapping for 10% of predictors, the best R2 score for test set: {}".format(r2_test[findLargest(r2_test)]))
len(pred[findLargest(r2_test)])
print('The assocaited predictors are')
pred[findLargest(r2_test)]
```


    After bootstrapping for 10% of predictors, the best R2 score for test set: 0.21855216986291626
    The assocaited predictors are





    ['acousticness_mean',
     'liveness_std',
     'loudness_mean',
     'valence_mean',
     'followers_std',
     'top_0_10',
     " 'ambient'",
     " 'bebop'",
     " 'bluegrass'",
     " 'bow pop'",
     " 'chillhop'",
     " 'classic rock'",
     " 'classical piano'",
     " 'contemporary jazz'",
     " 'country dawn'",
     " 'deep talent show'",
     " 'desi hip hop'",
     " 'ethereal wave'",
     " 'freestyle'",
     " 'future garage'",
     " 'garage punk blues'",
     " 'heavy alternative'",
     " 'indie dream pop'",
     " 'indie garage rock'",
     " 'indie psych-rock'",
     " 'intelligent dance music'",
     " 'latin metal'",
     " 'metropopolis'",
     " 'modern uplift'",
     " 'motown'",
     " 'nashville sound'",
     " 'neo-industrial rock'",
     " 'new americana'",
     " 'noise rock'",
     " 'nu gaze'",
     " 'ok indie'",
     " 'piano blues'",
     " 'pop christmas'",
     " 'pop'",
     " 'power metal'",
     " 'progressive house'",
     " 'progressive post-hardcore'",
     " 'retro electro'",
     " 'scorecore'",
     " 'sludge metal'",
     " 'southern soul'",
     " 'swedish indie rock'",
     " 'teen pop'",
     " 'trance'",
     " 'triangle indie'",
     " 'tropical house'",
     " 'uk post-punk'",
     " 'vocal house'",
     '"children\'s christmas"',
     "'alternative rock'",
     "'australian alternative rock'",
     "'baroque'",
     "'blackgaze'",
     "'canadian country'",
     "'canadian pop'",
     "'cantautor'",
     "'chillhop'",
     "'deep contemporary country'",
     "'desert blues'",
     "'djent'",
     "'dreamo'",
     "'drone'",
     "'east coast hip hop'",
     "'house'",
     "'indie pop'",
     "'indie poptimism'",
     "'indie psych-rock'",
     "'mellow gold'",
     "'minimal techno'",
     "'ninja'",
     "'skate punk'",
     "'vancouver indie'",
     "'wrestling'",
     'Rihanna',
     'Chance The Rapper',
     'Otis Redding',
     'SZA',
     'Birdy',
     'Commodores',
     'Str_Party',
     'Str_Acoustic',
     'Str_2000s',
     'acoustic_key_std']



### Part(g) PCA

In order to cut down the number of predictors, we implement PCA here. We choose number of PCA components from 1 to 100 and choose the optimal number of PCA components.

We acheive the best r2 test score of 0.13 with 30 PCA components.



```python
from sklearn.decomposition import PCA
r2_test_pca=[]
for i in range(1,100):
    pca = PCA(n_components=i)
    pca.fit(x_train)
    x_train_pca = pca.transform(x_train)
    x_test_pca = pca.transform(x_test)
    pca_regression_model = LinearRegression(fit_intercept=True)
    pca_regression_model.fit(x_train_pca, y_train)
    r2_test_pca.append(pca_regression_model.score(x_test_pca, y_test))
```





    0.13027619329603601





```python
print("After PCA, the best R2 score for test set: {}".format(r2_test_pca[findLargest(r2_test_pca)]))
print("The optimal number of components is: {}".format(findLargest(r2_test_pca)+1))
```


    After PCA, the best R2 score for test set: 0.130276193296036
    The optimal number of components is: 30


### Part(h) Lasso and Ridge

In part (h), we fit Ridge and Lasso with cross validation and with lamda ranging from 1e^-5 to 10^5.

With Lasso, the test r2 score is 0.109.

With Ridge, the test r2 score is 0.112.



```python
#lasso CV
lambda_list=[pow(10,i) for i in range(-5,5)]
lasso_regression = LassoCV(alphas=lambda_list, fit_intercept=True)
lasso_regression.fit(x_train, y_train)
print("With lasso, R2 score for test set: {}".format(lasso_regression.score(x_test,y_test)))
```


    With lasso, R2 score for test set: 0.1094129599174094




```python
#ridge
ridge_regression = RidgeCV(cv=10,alphas=lambda_list, fit_intercept=True, normalize=True)
ridge_regression.fit(x_train, y_train)
print("With ridge, R2 score for test set: {}".format(ridge_regression.score(x_test,y_test)))
```


    With ridge, R2 score for test set: 0.11234514421712671


### Summary

Since we have 949 predictors in our original data, we encounter the problem of overfitting when constructing regression models. Therefore, for basline models, we focus on selecting a subgroup of predictors that perform well in terms of r2 test score. By using only the significant predictors from the full model, test r2 score improves a lot (from -2.7 to 0.8). PCA gives us r2 score of 0.13. Lasso gives up 0.109 and Ridge gives us 0.11. We should note that iteratively choosing 10% of predictors gives us the best baseline r2 metric (0.21). However, this model is not robust and tend toward overfitting since we do not methodologically choose the predictors. Still, we could use 0.21 as a reference when evaluating our future models. In summary, the most robust and predictive baseline model is PCA with r2 socre of 0.13. But we may use 0.21 as baseline r2_test metric for the furture. In addition, we should consider including top 30 artists predictors since they alone gives us 0.03 r2 test metric. On the other hand, we should carefully select which genre predictors to include since there are 865 of them and would lead to overfitting.



```python

```

