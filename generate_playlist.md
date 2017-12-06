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

- According to Ben Gorman, if Linear Regression were a Toyota Camry, the Gradient Boosting Regressor would easily be a UH-60 Blackhawk Helicopter
- Gradient Boosting Regressor is an ensemble machine learning procedure that fits new models consecutively to provide a more reliable estimate of the response variable. It constructs new base-learners to be correlated with the negative gradient of the loss function 
 - least square regression (ls), 
 - least absolute deviation (lad), 
 - huber (a combination of ls and lad), 
 - quantile - which allows for quantile regression
- The choice of the loss function allows for great flexibility in Gradient Boosting and the best error function is huber for our model based on trial and error / cross-validation





```python
def optimized_playlist(style, song_count):
    '''Returns playlist songs most-likely to be popular in style'''
    
    play_index = np.argmax(model.predict(x_train[x_train[style] == 1.0]))
    data_index = x_train[x_train[style] == 1.0].index.tolist()[play_index]
    playlist_id = data_master.iloc[data_index]["ID"]
    
    results = sp.user_playlist_tracks('spotify', playlist_id)
    tracks = results['items']
    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])
    
    songs_playlist = []
    for item,song in enumerate(tracks):
        song_name = tracks[item]['track']['name']
        songs_playlist.append(song_name)
    
    sample = random.sample(songs_playlist,song_count)
    return sample
```


### Examples



```python
optimized_playlist("hip hop",5)
```





    ['Bad Girls',
     'Independent Women, Pt. 1',
     'Before He Cheats',
     'White Flag',
     "Bitch I'm Madonna"]





```python

```

