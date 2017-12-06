---
title: EDA
notebook: eda.ipynb
nav_include: 2
---

## Contents
{:.no_toc}
*  
{: toc}


## Exploratory Data Analysis

After gathering our neccessary data, the next step was to explore the data, to observe what information might be useful and organize it in a format ready for modeling. 













## Predictor Variables

The first step to cleaning the data was to first examine the dataframe for any problematic or duplicate columns. 













    Train Size: (1278, 950)
    Test Size: (142, 950)


























## Response Variable

The response variable is the number of followers of different playlists. Because there are cases in which some playlists accumulate an extremely large number of followers, it was observed that the distribution of the response variable is highly right skewed. In order to fix fo this, a log transform was applied and the playlists with no followers were discarded. As can be seen by the figure below, logging the number of followers for the dataset creates a more normal distribution, as well as Performing a log trasnformation helps with visualizing the data, as well as with modelling and making predictions. 










![png](eda_files/eda_19_0.png)


## Audio Features

Audio features are available on Spotify for each track. In total, eleven audio features were extracted for each track. For each playlist, the means and standard deviations of those features across the tracks in the playlist were used. 


These features are descriptors of the audio signals of each track. The full list of audio features and explanations for each are available in: 
https://developer.spotify.com/web-api/get-audio-features/


Descriptions for the features shown below:

**Energy:** Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy.

**Liveness:** Detects the presence of an audience in the recording.

**Tempo:** The overall estimated tempo of a track in beats per minute (BPM).

**Valence:** A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).


When viewed in scatterplots against the response variable, it can be seen that certain features seem to contain useful information. In particular, it seems that songs with high liveness (in which we can hear an audience in the recording)and songs with high valence (positive sounding) tend to have fewer followers.






![png](eda_files/eda_22_0.png)






    High five! You successfully sent some data to your account on plotly. View your plot in your browser at https://plot.ly/~jeehye95/0 or inside your plot.ly account where it is named 'update_dropdown'





<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~jeehye95/0.embed" height="525px" width="100%"></iframe>



## Genre Features

Genre features describe the genres of songs that the artists included in the playlist create. The following bar chart shows the mean number of playlist followers for the most common genres. (Common genres refer the the genres that many playlists fall under). It can be observed that **rap** has the highest number of mean followers. It can be observed that all of the common genres have at least 200,000 followers, which is at the high end of the spectrum.  









    Counts for Significant Genres: 
    
                         Num_tracks  Mean_Follow  Total_Follow     Std_Follow
     'rap'                    967.0     265222.0   251430004.0  821717.826915
    'dance pop'               956.0     255620.0   239515624.0  825409.275100
     'pop rap'               1030.0     254605.0   256896826.0  807502.717628
     'pop'                   1071.0     245154.0   256921483.0  793449.816572
     'modern rock'           1128.0     245142.0   271862478.0  772018.721861
     'rock'                   911.0     240249.0   214062114.0  787293.952886
     'indietronica'           934.0     240152.0   219979339.0  761555.292875
     'indie folk'             916.0     233859.0   210706655.0  526542.324277
     'classic rock'           917.0     233528.0   209708010.0  759646.322334
     'indie rock'             933.0     230190.0   210623582.0  523374.742379
    'no_genre'               1271.0     228706.0   285425695.0  732681.606393
     'nu metal'               917.0     228482.0   206090832.0  544527.585598
    'alternative metal'      1011.0     225606.0   224251968.0  534626.063859
     'post-grunge'            948.0     224583.0   209086816.0  517037.606322











![png](eda_files/eda_29_0.png)


The following histogram shows the distribution of two very different genres- alternative rock and dance pop. 
Both genres have ore than 500 tracks, but have a very differing number of playlist followers. 


**Genre:** 'alternative rock'

**Followers:** 219751.0

**Num Tracks:** 691.0

---------------------------------------------

**Genre:**  'dance pop'

**Followers:** 301507.0

**Num Tracks:** 614.0





    Least Followers (with at least 500 tracks): 
    Genre: 'alternative rock'
    Followers: 219751.0
    Num Tracks: 691.0
    ---------------------------------------------
    Most Followers (with at least 500 tracks): 
    Genre:  'dance pop'
    Followers: 301507.0
    Num Tracks: 614.0











![png](eda_files/eda_33_0.png)


### Interactions Between Genres and Audio Features

Interaction terms that could potentially provide useful insight are between genres and audio features. These could help answer questions such as: would the number of followers differ for different levels of 'danceability' for dance music vs. rap music? Because the number of genres exceed 100, the genres were first binned into broad genres such as 'house','hip hop','pop','dance','r&b','rap','acoustic','soul'. Then interaction terms were made with these broader genres. 

The following scatterplots and histograms show how the audio features may differ between two very different genres: rap music and dance music. Although it is difficult to observe any distinct differences between the relationship with the interaction terms with the response, we do see that their distributions differ (slightly). For example, rap music seems to be slightly higher in mean energy, and pop music seems to have a slightly higher valence. 









    Train Size: (1257, 893)
    Test Size: (139, 893)























![png](eda_files/eda_42_0.png)


## Artist Features

We think that the presence of  artists who appear most often in popular playlists would be a good predictor for playlist success. To evaluate a playlist's popularity, we found out that a playlist with 35,000+ followers beat 80% of playlist in terms of followers. Therefore, we use 35,000+ as a benchmark. 












    array(['A$AP Rocky', 'Otis Redding', 'Yo Gotti', 'Galantis', 'JAY Z',
           'Led Zeppelin', '21 Savage', 'Chance The Rapper', 'Rihanna',
           'Post Malone', 'Lil Wayne', 'Axwell /\\ Ingrosso', 'Young Thug',
           'Wiz Khalifa', 'Van Morrison', 'Elton John', 'Niall Horan', 'Diddy',
           'Deorro', 'Commodores', 'Radiohead', 'Adele', 'John Mayer', 'Birdy',
           'SYML', 'Ryan Adams', 'Ty Dolla $ign', 'SZA', 'Kanye West'], dtype=object)
























![png](eda_files/eda_51_0.png)


We examine the frequency of the artists that appear most often in playlists with 35,000+playlists. We select the top 30 that appear most often. Then, we average the playlist followers that includes such a top artist to examine whether playlist including such artists are indeed more popular. From the graph, we can see that 80% of these artists lead to an average playlist followers of over 40,0000, demonstrating that these artists are potentially good predictors for playlist success.














![png](eda_files/eda_55_0.png)











![png](eda_files/eda_57_0.png)


## Title Features

The last categories that were explored were the titles of the playlists. Spotify users commonly search for certain words in playlist titles, such as "Best of 2017" or "Top Pop Music." Titles were parsed to find certain substrings which were common in titles, and then categorized. For example, the titles containing "top" and "best" belong to the same category (Best). Titles containing "motivation", "exercise", or "workout" were all categorized as workout song titles. The following chart shows the mean followers for these different title categories. It is clear that the "Best" category has a high number of mean followers whereas the older songs of the 20th century have a low number of mean followers. 





    Counts for Significant Titles: 
    
                  Num_tracks  Mean_Follow  Total_Follow    Std_Follow
    Str_1970s           15.0     886801.0    10641610.0  2.251041e+06
    Str_2000s           56.0     522842.0    25096423.0  2.584003e+06
    Str_Chill           22.0     438654.0    10966340.0  1.327551e+06
    Str_Acoustic        21.0     231332.0     5089314.0  3.593668e+05
    Str_Workout         25.0     212102.0     5514662.0  3.970638e+05
    Str_Best            78.0     191977.0    12478475.0  6.122465e+05
    Str_1990s           41.0     129461.0     3754367.0  2.250944e+05
    Str_Party           16.0     115467.0     1616536.0  1.305883e+05
    Str_1960s           17.0      76001.0      912008.0  9.072836e+04
    Str_1980s           17.0      59298.0      889464.0  5.708579e+04











![png](eda_files/eda_62_0.png)


The following wordcloud visual shows the same categories, in which the image on the left represents the number of playlists in the category (frequency) and the image on the right represents the number of mean followers for playlists of the category (popularity). It is seen that many playlists are from the 2000s or are for working out. Compared to their frequencies, the party and "best" playlists have high popularity. 







```python
word_cloud()
```



![png](eda_files/eda_65_0.png)

