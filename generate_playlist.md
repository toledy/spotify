---
title: Playlist Generation & Conclusion
notebook: generate_playlist.ipynb
nav_include: 5
---

## Contents
{:.no_toc}
*  
{: toc}

## Generating Successful Playlists

### Methodology

One of the high-level project goals was to

> Use a regression model to generate new playlists according to a user-specified genre or other search filters.

To this extent, the fitted Gradient Boosting Regressor has been used. Specifically, the process of generating a playlist based on user-specified genre or other criteria is:

1. Select a subframe of the data (from master dataframe) based on the user-specified preference (e.g., genre, audio feature or artist)
2. Employ the fitted Gradient Boosting Regressor model to predict which playlist from the sub-dataframe would most likely have the most followers (using all predictors available)
3. From the predicted most popular playlist, sample a number of songs randomly (where the user can specify how many songs are desired)
4. Return song selection and provide the possibility to plot additional metrics





### Examples

For example, if you feel like listening to deep house, simply type "deep house" and the number of songs in your custom playlist desired. The function will return that number of songs from the most highly rated (in terms of followers) predicted playlist in Spotify.



```python
optimized_playlist("deep house",10)
```





    ['The Heat (I Wanna Dance With Somebody)',
     'Kiss You All Over - Single Edit',
     'Wait',
     "Runnin'",
     'Another Shot',
     'Let Me Love You',
     'Dreamer',
     'Like You Mean It',
     'Jealousy - GOLD RVSH Remix',
     'One Foot']



Similarly, to get a better understanding of all the "deep house" playlists in our data set, simply expand the function request as per the below. The output is a graph which shows the predicted logged followers of all playlists that match the genre "deep house". As is clear, the function samples songs from the most highly rated predicted playlist.



```python
optimized_playlist("deep house",5,summary=False,plot=True)
```



![png](generate_playlist_files/generate_playlist_31_0.png)


Other inputs work just as well - for example, assume you are in the mood for "2000s" songs. Simply request "Str_2000s" from the function and the number of songs requested and the output will be a sample of songs from the predicted most highly followed playlist containing "2000s" songs.



```python
optimized_playlist("Str_2000s",5)
```





    ['Through Glass', 'Hero', 'Happy?', 'Click Click Boom', 'Bodies']



As above, to get a better understanding of "2000s" playlist popularity (and to understand from which playlist the songs get sampled), simply request for the plot to display.



```python
optimized_playlist("Str_2000s",3,summary=False,plot=True)
```



![png](generate_playlist_files/generate_playlist_35_0.png)


The example genres showcased above are only a small sample of choices that the playlist generator can deal with - in essence, all predictor columns of the data set could be used (albeit with slightly tweaked inputs) instead as well. Other examples include: "violin", "Str_Workout", "pop", "Rihanna" and hundreds more.

## Conclusion & Future Work


