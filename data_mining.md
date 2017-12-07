---
title: Data Mining & Wrangling
notebook: data_mining.ipynb
nav_include: 1
---

## Contents
{:.no_toc}
*  
{: toc}



## Data Mining

### Connecting With The Spotify API

To begin pulling playlist data from the Spotify API, first a connection with the API needs to be made. For this, both a so-called "client id" and "client secret id" were required. Once these "id's" were obtained - setting up the API connection was as simple as following the below outlined steps:



```python
# ID and Password for accessing Spotify API
client_id = "client_id"
client_secret = "client_secret_id"

# Setup the credentials
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)

# Make the connection
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
```


### Collect Spotify's Featured Playlist Data

The main idea of this project is twofold: (i) being able to infer key predictors (whether track features or artist features) which are statistically significant in determining a playlist's success in terms of number of followers; and (ii) being able to create a custom playlist that is deemed to be succesful (i.e., would obtain many followers).

To this extent, the first step in doing any further analysis was to obtain the playlists we wanted to run our predictions on. We decided to focus on Spotify's own "featured" playlists - i.e., those produced by Spotify itself given specific genres / moods / artists etc.. 

The initial step was to pull Spotify's featured playlists and obtain a number of base playlist features.



```python
# Get all spotify playlists
playlists = sp.user_playlists('spotify')

# Empty list to hold playlist information
spotify_playlists = []

# Loop to get data for each playlist
while playlists:
    
    for i, playlist in enumerate(playlists['items']):
        names = playlist['name']
        track_count = playlist['tracks']['total']
        ids = playlist['id']
        uri = playlist['uri']
        href = playlist['href']
        public = playlist['public']
        data_aggregation = names, track_count, ids, uri, href, public
        spotify_playlists.append(data_aggregation)
        
    if playlists['next']:
        playlists = sp.next(playlists)
    
    else:
        playlists = None
```


The obtained baseline playlist features were converted into a large dataframe next.



```python
# Convert list into a dataframe
data = pd.DataFrame(np.array(spotify_playlists).reshape(len(spotify_playlists),6), 
                    columns=['Name', 'No. of Tracks', 'ID', 'URI', 'HREF', 'Public'])
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
      <th>Name</th>
      <th>No. of Tracks</th>
      <th>ID</th>
      <th>URI</th>
      <th>HREF</th>
      <th>Public</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Today's Top Hits</td>
      <td>50</td>
      <td>37i9dQZF1DXcBWIGoYBM5M</td>
      <td>spotify:user:spotify:playlist:37i9dQZF1DXcBWIG...</td>
      <td>https://api.spotify.com/v1/users/spotify/playl...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>RapCaviar</td>
      <td>63</td>
      <td>37i9dQZF1DX0XUsuxWHRQd</td>
      <td>spotify:user:spotify:playlist:37i9dQZF1DX0XUsu...</td>
      <td>https://api.spotify.com/v1/users/spotify/playl...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>mint</td>
      <td>61</td>
      <td>37i9dQZF1DX4dyzvuaRJ0n</td>
      <td>spotify:user:spotify:playlist:37i9dQZF1DX4dyzv...</td>
      <td>https://api.spotify.com/v1/users/spotify/playl...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Are &amp; Be</td>
      <td>51</td>
      <td>37i9dQZF1DX4SBhb3fqCJd</td>
      <td>spotify:user:spotify:playlist:37i9dQZF1DX4SBhb...</td>
      <td>https://api.spotify.com/v1/users/spotify/playl...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Rock This</td>
      <td>64</td>
      <td>37i9dQZF1DXcF6B6QPhFDv</td>
      <td>spotify:user:spotify:playlist:37i9dQZF1DXcF6B6...</td>
      <td>https://api.spotify.com/v1/users/spotify/playl...</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



For each playlist, the number of followers was obtained - this number will be the response variable for our regression based models.



```python
# Pull the number of followers per playlist
playlist_follower = []

# Loop over playlists and get followers
for i in range(0, len(data['URI'])-1): 
    
    # If number of followers is greater than 0
    if data['No. of Tracks'][i] > 0:
        uri = data['URI'][i]
        username = uri.split(':')[2]
        playlist_id = uri.split(':')[4]
        results = sp.user_playlist(username, playlist_id)
        followers = results['followers']['total']
        playlist_follower.append(followers)
    
    # If follower count is 0, append 0   
    else: 
        followers = 0
        playlist_follower.append(followers)
```


Finally - the number of followers was concatenated to the playlist dataframe.



```python
# Add a new column for followers 
data['Followers'] = pd.DataFrame({'Followers': playlist_follower})
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
      <th>Name</th>
      <th>No. of Tracks</th>
      <th>ID</th>
      <th>URI</th>
      <th>HREF</th>
      <th>Public</th>
      <th>Followers</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Today's Top Hits</td>
      <td>50</td>
      <td>37i9dQZF1DXcBWIGoYBM5M</td>
      <td>spotify:user:spotify:playlist:37i9dQZF1DXcBWIG...</td>
      <td>https://api.spotify.com/v1/users/spotify/playl...</td>
      <td>True</td>
      <td>18247159.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>RapCaviar</td>
      <td>63</td>
      <td>37i9dQZF1DX0XUsuxWHRQd</td>
      <td>spotify:user:spotify:playlist:37i9dQZF1DX0XUsu...</td>
      <td>https://api.spotify.com/v1/users/spotify/playl...</td>
      <td>True</td>
      <td>8375355.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>mint</td>
      <td>61</td>
      <td>37i9dQZF1DX4dyzvuaRJ0n</td>
      <td>spotify:user:spotify:playlist:37i9dQZF1DX4dyzv...</td>
      <td>https://api.spotify.com/v1/users/spotify/playl...</td>
      <td>True</td>
      <td>4616753.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Are &amp; Be</td>
      <td>51</td>
      <td>37i9dQZF1DX4SBhb3fqCJd</td>
      <td>spotify:user:spotify:playlist:37i9dQZF1DX4SBhb...</td>
      <td>https://api.spotify.com/v1/users/spotify/playl...</td>
      <td>True</td>
      <td>3806312.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Rock This</td>
      <td>64</td>
      <td>37i9dQZF1DXcF6B6QPhFDv</td>
      <td>spotify:user:spotify:playlist:37i9dQZF1DXcF6B6...</td>
      <td>https://api.spotify.com/v1/users/spotify/playl...</td>
      <td>True</td>
      <td>4004115.0</td>
    </tr>
  </tbody>
</table>
</div>



Following the above outlined steps, we were able to produce a dataframe consisting of, in excess 1400, playlists with  relevant information such as playlist id, number of playlist tracks, and number of playlist followers.

### Collect Spotify Audio Features Per Track in Playlist

Using the dataframe of playlists - and specifically the playlist id column - we iterated over all tracks in every playlist and pulled relevant audio features which could potentially be helpful in predicting the success of a playlist.

To this extent, we defined a function to pull all playlists' tracks.



```python
# New function to get tracks in playlist
def get_playlist_tracks(username, playlist_id):
    results = sp.user_playlist_tracks(username, playlist_id)
    tracks = results['items']
    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])
    return tracks
```


Running the feature extraction from Spotify could take a significant amount of time and could also be prone to raise errors in the process. To avoid losing information when such error occurs, a dictionary was used in cache memory.



```python
# Subsample of data to pull
Spotify_playlists = data.iloc[0:10]

# Create playlist cache in memory
playlist_tracks = dict()
```


The playlists were prepped for audio feature extraction.


```python
# Define an example list of songs for the first 10 playlists
songs_playlist = []

for item,playlist in enumerate(playlist_tracks):
    track_len = len(playlist_tracks[playlist])
    for song_item,song in enumerate(playlist_tracks[playlist]):
        songs_playlist.append((playlist,playlist_tracks[playlist][song_item]['track']['id']))
        
print("Number of Songs in Playlists: {}".format(len(songs_playlist)))
```


    Number of Songs in Playlists: 663


Again, a dictionary in cache memory was setup for the main audio feature extraction loop.



```python
# Create audio feature dictionary and set sleeping time thresholds
songs = [item[1] for item in songs_playlist]

audio_feat = dict()
limit_songs_small = 10
limit_songs_medium = 200
```


Audio features were extracted using the below code - note running this code on all playlists takes a significant amount of time (measured in hours).



```python
# Audio feature extraction - saves information in cache
for item,song in enumerate(songs):
    if song not in audio_feat:
        try:
            audio_feat[song] = sp.audio_features(song)
        except:
            pass

        if item % limit_songs_small == 0:
            time.sleep(random.randint(0, 1))

        if item % limit_songs_medium == 0:
            time.sleep(random.randint(0, 1))

        out = np.floor(item * 1. / len(songs_playlist) * 100)
        sys.stdout.write("\r%d%%" % out)
        sys.stdout.flush()

sys.stdout.write("\r%d%%" % 100)
```

Once all the audio features were extracted, they were converted into the main audio feature dataframe and saved down as a large csv file.



```python
# Merge individual dataframes into one features dataframe
playlist_df = pd.DataFrame(songs_playlist,columns=['playlist','song'])

frame_one = [acc_df,dan_df,dur_df,ene_df,inst_df,key_df,live_df,loud_df,mode_df,spee_df,temp_df,time_df,vale_df]
features = pd.concat(frame_one,axis=1).T.groupby(level=0).first().T

frame_two = [features,playlist_df]
features_df = pd.concat(frame_two,axis=1).T.groupby(level=0).first().T.dropna()

features_df.head()
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
      <th>acousticness</th>
      <th>dance</th>
      <th>duration</th>
      <th>energy</th>
      <th>instrumentalness</th>
      <th>key</th>
      <th>liveness</th>
      <th>loudness</th>
      <th>mode</th>
      <th>playlist</th>
      <th>song</th>
      <th>speech</th>
      <th>tempo</th>
      <th>time</th>
      <th>valence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.365</td>
      <td>0.307</td>
      <td>258933</td>
      <td>0.481</td>
      <td>0</td>
      <td>3</td>
      <td>0.207</td>
      <td>-8.442</td>
      <td>0</td>
      <td>37i9dQZF1DXcBWIGoYBM5M</td>
      <td>00kkWwGsR9HblTUHb3BmdX</td>
      <td>0.128</td>
      <td>68.894</td>
      <td>3</td>
      <td>0.329</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.993</td>
      <td>0.322</td>
      <td>160897</td>
      <td>0.0121</td>
      <td>0.927</td>
      <td>5</td>
      <td>0.127</td>
      <td>-31.994</td>
      <td>1</td>
      <td>37i9dQZF1DXcBWIGoYBM5M</td>
      <td>01T3AjynqSMVfiAQCAfrKJ</td>
      <td>0.0491</td>
      <td>112.464</td>
      <td>4</td>
      <td>0.118</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.994</td>
      <td>0.375</td>
      <td>58387</td>
      <td>0.00406</td>
      <td>0.908</td>
      <td>7</td>
      <td>0.0842</td>
      <td>-31.824</td>
      <td>0</td>
      <td>37i9dQZF1DXcBWIGoYBM5M</td>
      <td>02BumRY2OTFMkMxrXSVMat</td>
      <td>0.0671</td>
      <td>139.682</td>
      <td>1</td>
      <td>0.358</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.992</td>
      <td>0.393</td>
      <td>288280</td>
      <td>0.0429</td>
      <td>0.925</td>
      <td>9</td>
      <td>0.0821</td>
      <td>-25.727</td>
      <td>0</td>
      <td>37i9dQZF1DXcBWIGoYBM5M</td>
      <td>02mkkozonPEDCenOhuWwLc</td>
      <td>0.0341</td>
      <td>135.405</td>
      <td>4</td>
      <td>0.0394</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.992</td>
      <td>0.373</td>
      <td>99867</td>
      <td>0.117</td>
      <td>0.909</td>
      <td>10</td>
      <td>0.111</td>
      <td>-25.222</td>
      <td>0</td>
      <td>37i9dQZF1DXcBWIGoYBM5M</td>
      <td>02xmGU9unopKjpblPRC67j</td>
      <td>0.0511</td>
      <td>125.288</td>
      <td>3</td>
      <td>0.189</td>
    </tr>
  </tbody>
</table>
</div>





```python
# Save as csv file
features_df.to_csv('track_features(track_indices).csv', sep=',')
```


### Collect Spotify Artist Information Per Track in Playlist

Following a similar procedure as the audio feature extraction, artist information for every track in every playlist was extracted next.

First, a function was defined to retrieve artist information given an artist name.



```python
# Subsample of data to pull
Spotify_playlists = data.iloc[0:10]

# Collect artist information per track found in Step II
playlist_tracks = dict()
```




```python
# New function to get artists in playlist
def get_artist(name):
    results = sp.search(q='artist:' + name, type='artist')
    items = results['artists']['items']
    if len(items) > 0:
        return items[0]
    else:
        return None
```


The playlists were prepped for artist feature extraction.



```python
# Define an example list of songs for the first 10 playlists
artist_list = []
song_dict = dict()
playlist_dict = dict()

for play_index,playlist in enumerate(playlist_tracks):
    songs = playlist_tracks[playlist]
    for song_index,song in enumerate(songs):
        no_artists = len(song['track']['artists'])
        for number in range(no_artists):
            name = song['track']['artists'][number]['name']
            song_id = song['track']['id']
            artist_list.append((playlist,song_id,name))
            song_dict[name] = song_id
            playlist_dict[name] = playlist
```


Again, a dictionary in cache memory was setup for the main artist feature extraction loop.



```python
# Create artist feature dictionary and set sleeping time thresholds
artists = list(set([item[2] for item in artist_list]))

artist_info = dict()
limit_artist_small = 10
limit_artist_medium = 200
```


Artist features were extracted using the below code - note running this code on all playlists takes a significant amount of time (measured in hours).



```python
# Artist feature extraction - saves information in cache
for item,artist in enumerate(artists):
    if artist not in artist_info:
        try:
            artist_info[artist] = get_artist(artist)
        except:
            pass
    
    if item % limit_artist_small == 0:
        time.sleep(random.randint(0, 1))
    
    if item % limit_artist_medium == 0:
        time.sleep(random.randint(0, 1))
        
    out = np.floor(item * 1. / len(artists) * 100)
    sys.stdout.write("\r%d%%" % out)
    sys.stdout.flush()

sys.stdout.write("\r%d%%" % 100)
```

Once all the artist features were extracted, they were converted into the main artist feature dataframe and saved down as a large csv file.



```python
# Merge individual dataframes into one features dataframe
frame_one = [follow_df,genres_df,popularity_df,song_df, playlist_df]
artist_information = pd.concat(frame_one,axis=1).T.groupby(level=0).first().T
artist_information.head()
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
      <th>artist</th>
      <th>followers</th>
      <th>genres</th>
      <th>playlist</th>
      <th>popularity</th>
      <th>song</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10 Years</td>
      <td>157035</td>
      <td>[alternative metal, nu metal, post-grunge, rap...</td>
      <td>37i9dQZF1DXcF6B6QPhFDv</td>
      <td>63</td>
      <td>0uyDAijTR0tOuH24hxDhE5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>21 Savage</td>
      <td>2323273</td>
      <td>[dwn trap, rap, trap music]</td>
      <td>37i9dQZF1DX0XUsuxWHRQd</td>
      <td>98</td>
      <td>2vaMWMPMgsWX4fwJiKmdWm</td>
    </tr>
    <tr>
      <th>2</th>
      <td>24hrs</td>
      <td>28839</td>
      <td>[dwn trap, trap music, underground hip hop]</td>
      <td>37i9dQZF1DX0XUsuxWHRQd</td>
      <td>73</td>
      <td>2c5D6B8oXAwc6easamdgVA</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3LAU</td>
      <td>175224</td>
      <td>[big room, brostep, deep big room, edm, electr...</td>
      <td>37i9dQZF1DX4JAvHpjipBk</td>
      <td>67</td>
      <td>6yxobtnNHKRAA0cvoNxJhe</td>
    </tr>
    <tr>
      <th>4</th>
      <td>50 Cent</td>
      <td>2686486</td>
      <td>[east coast hip hop, gangster rap, hip hop, po...</td>
      <td>37i9dQZF1DX0XUsuxWHRQd</td>
      <td>85</td>
      <td>32aYDW8Qdnv1ur89TUlDnm</td>
    </tr>
  </tbody>
</table>
</div>





```python
# Save as csv file
artist_information.to_csv('artists(track_indices).csv', sep=',')
```


## Data Wrangling

### Loading Data Frames

Once all data was extracted from Spotify, the next step was to combine the separate dataframes (i.e., for playlists, audio features and artists) and to perform some initial feature engineering in the hopes of creating useful data for inference and prediction of playlist success.

The first step was to load all the dataframes separately.



```python
# Load playlist dataframe
playlist_df = pd.read_csv('Playlist.csv')
playlist_df.head()
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
      <th>Name</th>
      <th>No. of Tracks</th>
      <th>ID</th>
      <th>URI</th>
      <th>HREF</th>
      <th>Public</th>
      <th>Followers</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Today's Top Hits</td>
      <td>50</td>
      <td>37i9dQZF1DXcBWIGoYBM5M</td>
      <td>spotify:user:spotify:playlist:37i9dQZF1DXcBWIG...</td>
      <td>https://api.spotify.com/v1/users/spotify/playl...</td>
      <td>True</td>
      <td>18079985.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>RapCaviar</td>
      <td>61</td>
      <td>37i9dQZF1DX0XUsuxWHRQd</td>
      <td>spotify:user:spotify:playlist:37i9dQZF1DX0XUsu...</td>
      <td>https://api.spotify.com/v1/users/spotify/playl...</td>
      <td>True</td>
      <td>8283836.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>mint</td>
      <td>61</td>
      <td>37i9dQZF1DX4dyzvuaRJ0n</td>
      <td>spotify:user:spotify:playlist:37i9dQZF1DX4dyzv...</td>
      <td>https://api.spotify.com/v1/users/spotify/playl...</td>
      <td>True</td>
      <td>4593498.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Are &amp; Be</td>
      <td>51</td>
      <td>37i9dQZF1DX4SBhb3fqCJd</td>
      <td>spotify:user:spotify:playlist:37i9dQZF1DX4SBhb...</td>
      <td>https://api.spotify.com/v1/users/spotify/playl...</td>
      <td>True</td>
      <td>3773823.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Rock This</td>
      <td>60</td>
      <td>37i9dQZF1DXcF6B6QPhFDv</td>
      <td>spotify:user:spotify:playlist:37i9dQZF1DXcF6B6...</td>
      <td>https://api.spotify.com/v1/users/spotify/playl...</td>
      <td>True</td>
      <td>3989695.0</td>
    </tr>
  </tbody>
</table>
</div>





```python
# Load track features dataframe
tracks_df = pd.read_csv('tracks_df_sub.csv').drop(['Unnamed: 0','Unnamed: 0.1'],axis=1)
tracks_df.head()
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
      <th>acousticness</th>
      <th>dance</th>
      <th>duration</th>
      <th>energy</th>
      <th>instrumentalness</th>
      <th>key</th>
      <th>liveness</th>
      <th>loudness</th>
      <th>mode</th>
      <th>playlist</th>
      <th>song</th>
      <th>speech</th>
      <th>tempo</th>
      <th>time</th>
      <th>valence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.039500</td>
      <td>0.299</td>
      <td>214973</td>
      <td>0.9210</td>
      <td>0.737000</td>
      <td>4</td>
      <td>0.5890</td>
      <td>-6.254</td>
      <td>1</td>
      <td>37i9dQZF1DXcBWIGoYBM5M</td>
      <td>0076oEQq8IToGfnzU3bTHY</td>
      <td>0.1930</td>
      <td>174.982</td>
      <td>4</td>
      <td>0.0532</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.365000</td>
      <td>0.307</td>
      <td>258933</td>
      <td>0.4810</td>
      <td>0.000000</td>
      <td>3</td>
      <td>0.2070</td>
      <td>-8.442</td>
      <td>0</td>
      <td>37i9dQZF1DXcBWIGoYBM5M</td>
      <td>00kkWwGsR9HblTUHb3BmdX</td>
      <td>0.1280</td>
      <td>68.894</td>
      <td>3</td>
      <td>0.3290</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.078700</td>
      <td>0.630</td>
      <td>261731</td>
      <td>0.6560</td>
      <td>0.000906</td>
      <td>0</td>
      <td>0.0953</td>
      <td>-6.423</td>
      <td>0</td>
      <td>37i9dQZF1DXcBWIGoYBM5M</td>
      <td>01JkrDSrakX5UO5knhpKNA</td>
      <td>0.0276</td>
      <td>133.012</td>
      <td>4</td>
      <td>0.4320</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.000192</td>
      <td>0.521</td>
      <td>188834</td>
      <td>0.8370</td>
      <td>0.051000</td>
      <td>5</td>
      <td>0.0929</td>
      <td>-4.581</td>
      <td>1</td>
      <td>37i9dQZF1DXcBWIGoYBM5M</td>
      <td>01KsbekyuQQXpVnxIfNRaC</td>
      <td>0.1220</td>
      <td>80.027</td>
      <td>4</td>
      <td>0.6230</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.993000</td>
      <td>0.322</td>
      <td>160897</td>
      <td>0.0121</td>
      <td>0.927000</td>
      <td>5</td>
      <td>0.1270</td>
      <td>-31.994</td>
      <td>1</td>
      <td>37i9dQZF1DXcBWIGoYBM5M</td>
      <td>01T3AjynqSMVfiAQCAfrKJ</td>
      <td>0.0491</td>
      <td>112.464</td>
      <td>4</td>
      <td>0.1180</td>
    </tr>
  </tbody>
</table>
</div>





```python
# Load artist information dataframe
artist_df_sub = pd.read_csv('artist_df_sub.csv').drop(['Unnamed: 0','Unnamed: 0.1'],axis=1)
artist_df_sub.head()
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
      <th>artist</th>
      <th>followers</th>
      <th>genres</th>
      <th>playlist</th>
      <th>popularity</th>
      <th>song</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>*NSYNC</td>
      <td>498511.0</td>
      <td>['boy band', 'dance pop', 'europop', 'pop', 'p...</td>
      <td>37i9dQZF1DWXDAhqlN7e6W</td>
      <td>75.0</td>
      <td>35zGjsxI020C2NPKp2fzS7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10 Years</td>
      <td>154800.0</td>
      <td>['alternative metal', 'nu metal', 'post-grunge...</td>
      <td>37i9dQZF1DWWJOmJ7nRx0C</td>
      <td>63.0</td>
      <td>4qmoz9OUEBaXUzlWQX4ZU4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2 Chainz</td>
      <td>1926728.0</td>
      <td>['dwn trap', 'pop rap', 'rap', 'southern hip h...</td>
      <td>37i9dQZF1DX7QOv5kjbU68</td>
      <td>91.0</td>
      <td>4XoP1AkbOurU9CeZ2rMEz2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>21 Savage</td>
      <td>2224587.0</td>
      <td>['dwn trap', 'rap', 'trap music']</td>
      <td>37i9dQZF1DX7QOv5kjbU68</td>
      <td>98.0</td>
      <td>4ckuS4Nj4FZ7i3Def3Br8W</td>
    </tr>
    <tr>
      <th>4</th>
      <td>24hrs</td>
      <td>27817.0</td>
      <td>['dwn trap', 'trap music', 'underground hip hop']</td>
      <td>37i9dQZF1DX0XUsuxWHRQd</td>
      <td>74.0</td>
      <td>2c5D6B8oXAwc6easamdgVA</td>
    </tr>
  </tbody>
</table>
</div>



As should be obvious from the above - artists were grouped in a list of genres by Spotify. To make sense from these lists for every artist, genres were one-hot encoded instead.



```python
# One-hot encode genre labels
mlb = MultiLabelBinarizer(sparse_output=True)
pre_data = mlb.fit_transform(artist_df_sub['genres'].str.split(','))
classes = [i.strip('[]') for i in mlb.classes_]
genre_sub = pd.DataFrame(pre_data.toarray(),columns=classes)
_, i = np.unique(genre_sub.columns, return_index=True)
genre_sub = genre_sub.iloc[:, i]

# Drop genre column from artist sub dataframe
artist_df_sub_mid = artist_df_sub.drop('genres', axis=1)

# Concatenate artist sub dataframe and genre dataframe
artist_sub_frames = [artist_df_sub_mid,genre_sub]
artist_df = pd.concat(artist_sub_frames,axis=1,join='inner')
```


Once all the genres were one-hot encoded, the dataframes were grouped by playlist to enable the below feature engineering.



```python
# Group-by function on artists
group_artists_by_playlist = artist_df.groupby('playlist') 
print("Number of playlists: ", len(group_artists_by_playlist))

# Group-by function on tracks
group_tracks_by_playlist = tracks_df.groupby('playlist')
print("Number of playlists: ", len(group_tracks_by_playlist))
```


    Number of playlists:  1546
    Number of playlists:  1465


### Feature Engineering

In terms of artists, feature engineering led to the following predictors:

* Thirty columns are the names of top 30 artists (in terms of appearing most often in popular playlists) They are categorical variablce indicating whether a playlist has a specific artist.
* Five columns representing the number of times top 50 artists (in terms of artist followers) appeared in the playlists (bucketed in 10 artists each)
* Two columns representing the mean and standard deviation of artists followers per playlist
* Two columns representing the mean and standard deviation of artists popularity per playlist
* Artist genres were one-hot encoded

First, the top 50 artists (in terms of number of Spotify followers) were listed. Final dataframe columns list the number of times these artists appear in a playlist. Second, a list of 30 artists that appear most often in playlists with 35,000+ followers was created. 


By looping over the playlists, the additional predictors were created as per the below.



```python
# Artist feature engineering
artist_feature_list=[]

for key, item in group_artists_by_playlist:
    
    category_artist_count=[]
    for ele in popular_artists:
        present=False
        for artist in item['artist']:
            if ele==artist:
                present=True
        category_artist_count.append(present*1)
    
    followers_mean=item['followers'].mean()
    followers_std=item['followers'].std()
    
    popularity_mean=item['popularity'].mean()
    popularity_std=item['popularity'].std()
    
    top_10 = item['top_0_10'].sum()
    top_10_20 = item['top_10_20'].sum()
    top_20_30 = item['top_20_30'].sum()
    top_30_40 = item['top_30_40'].sum()
    top_40_50 = item['top_40_50'].sum()
    
    tmp=[key, followers_mean,followers_std,popularity_mean,popularity_std,\
         top_10,top_10_20,top_20_30,top_30_40,top_40_50]
    for i in range(len(popular_artists)):
        tmp.append(category_artist_count[i])
    artist_feature_list.append(tuple(tmp))
    
artist_feature_names = ['followers_mean','followers_std','popularity_mean','popularity_std',
                       'top_0_10','top_10_20','top_20_30','top_30_40','top_40_50']
for i in range(len(popular_artists)):
        artist_feature_names.append(popular_artists[i])
```


All the genres in a playlist were encoded to ones in the one-hot encoded genre columns.



```python
# Splitting of genres and enumeration per playlist
genre_list = []

for key, item in group_artists_by_playlist:
    for genre in classes:
        genre_list.append(item[genre].max())
```


Finally, the main artist data frame was created below:



```python
# Reshape genres into array of proper dimensions
genre_arr = np.array(genre_list).reshape(len(artist_feature_list),len(classes))

# Create genre sub dataframe per playlist
artist_genres_df = pd.DataFrame(genre_arr)
artist_genres_df.columns = classes

#dataframe for artist grouped by playlist
artist_features_df = pd.DataFrame(artist_feature_list).set_index(0)
artist_features_df.columns = artist_feature_names

# column for number of followers
artist_features_df['Playlist_Followers'] = playlist_df[['Followers']].groupby(playlist_df['ID']).first()
artist_features_df['ID']=artist_features_df.index

artist_main_df = artist_features_df.reset_index().drop(0, axis=1)
artist_main_df.head()
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
      <th>followers_mean</th>
      <th>followers_std</th>
      <th>popularity_mean</th>
      <th>popularity_std</th>
      <th>top_0_10</th>
      <th>top_10_20</th>
      <th>top_20_30</th>
      <th>top_30_40</th>
      <th>top_40_50</th>
      <th>Playlist_Followers</th>
      <th>ID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>134413.666667</td>
      <td>3.654590e+05</td>
      <td>42.833333</td>
      <td>19.575645</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>24.0</td>
      <td>01WIu4Rst0xeZnTunWxUL7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>103320.580645</td>
      <td>3.320150e+05</td>
      <td>48.903226</td>
      <td>15.029648</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>330.0</td>
      <td>05dTMGk8MjnpQg3bKuoXcc</td>
    </tr>
    <tr>
      <th>2</th>
      <td>566814.560000</td>
      <td>1.427308e+06</td>
      <td>60.280000</td>
      <td>15.512146</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>73.0</td>
      <td>070FVPBKvfu6M5tf4I9rt2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>199831.484848</td>
      <td>2.953859e+05</td>
      <td>58.696970</td>
      <td>15.627470</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6173.0</td>
      <td>08vPKM3pmoyF6crB2EtASQ</td>
    </tr>
    <tr>
      <th>4</th>
      <td>223253.774194</td>
      <td>4.918438e+05</td>
      <td>49.516129</td>
      <td>19.489948</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>145.0</td>
      <td>08ySLuUm0jMf7lJmFwqRMu</td>
    </tr>
  </tbody>
</table>
</div>




Similar to the artist feature engineering, the playlists' audio features were engineered next. Specifically, for ever audio feature mined from Spotify, the mean and standard deviation across all playlist tracks was computed.



```python
# Feature Engineering for track df: save to feature_list 
feature_list = []

for key, item in group_tracks_by_playlist:

    acousticness_mean =item['acousticness'].mean()
    acousticness_std = item['acousticness'].std()
    
    dance_mean =item['dance'].mean()
    dance_std = item['dance'].std()
    
    duration_mean =item['dance'].mean()
    duration_std = item['dance'].std()
    
    energy_mean =item['energy'].mean()
    energy_std = item['energy'].std()
    
    instrumentalness_mean =item['instrumentalness'].mean()
    instrumentalness_std = item['instrumentalness'].std()
    
    key_mean =item['energy'].mean()
    key_std = item['energy'].std()
    
    liveness_mean =item['liveness'].mean()
    liveness_std = item['liveness'].std()
    
    loudness_mean =item['loudness'].mean()
    loudness_std = item['loudness'].std()
    
    mode_mean =item['mode'].mean()
    mode_std = item['mode'].std()
    
    speech_mean =item['speech'].mean()
    speech_std = item['speech'].std()
    
    tempo_mean =item['tempo'].mean()
    tempo_std = item['tempo'].std()
    
    time_mean =item['time'].mean()
    time_std = item['time'].std()
    
    valence_mean =item['valence'].mean()
    valence_std = item['valence'].std()
        
    feature_list.append((key, acousticness_mean, acousticness_std, dance_mean, dance_std, energy_mean, energy_std, 
                        instrumentalness_mean, instrumentalness_std, key_mean, key_std, liveness_mean, liveness_std,
                        loudness_mean, loudness_std, mode_mean, mode_std, speech_mean, speech_std, tempo_mean, tempo_std,
                        time_mean, time_std, valence_mean, valence_std))
# Save feature names
feature_names =  ['acousticness_mean','acousticness_std','dance_mean', 'dance_std', 'energy_mean', 'energy_std', 
                        'instrumentalness_mean', 'instrumentalness_std', 'key_mean', 'key_std', 'liveness_mean', 
                        'liveness_std','loudness_mean', 'loudness_std', 'mode_mean', 'mode_std', 'speech_mean', 
                        'speech_std','tempo_mean', 'tempo_std','time_mean', 'time_std', 'valence_mean', 'valence_std',
                  ]
```


The engineered audio features were converted into a dataframe as follows:



```python
features_df = pd.DataFrame(feature_list).set_index(0)
features_df.columns = feature_names

# Column for number of followers
features_df['Followers'] = playlist_df[['Followers']].groupby(playlist_df['ID']).first()
features_df['ID'] = features_df.index

features_main_df = features_df.reset_index().drop(0, axis=1)
features_main_df.head()
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
      <th>...</th>
      <th>speech_mean</th>
      <th>speech_std</th>
      <th>tempo_mean</th>
      <th>tempo_std</th>
      <th>time_mean</th>
      <th>time_std</th>
      <th>valence_mean</th>
      <th>valence_std</th>
      <th>Followers</th>
      <th>ID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.641282</td>
      <td>0.326942</td>
      <td>0.467911</td>
      <td>0.241057</td>
      <td>0.275940</td>
      <td>0.225821</td>
      <td>0.119650</td>
      <td>0.277109</td>
      <td>0.275940</td>
      <td>0.225821</td>
      <td>...</td>
      <td>0.383051</td>
      <td>0.403365</td>
      <td>101.045969</td>
      <td>51.857504</td>
      <td>3.338462</td>
      <td>1.553996</td>
      <td>0.319263</td>
      <td>0.246235</td>
      <td>24.0</td>
      <td>01WIu4Rst0xeZnTunWxUL7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.249844</td>
      <td>0.321182</td>
      <td>0.555140</td>
      <td>0.172088</td>
      <td>0.666567</td>
      <td>0.230578</td>
      <td>0.077776</td>
      <td>0.240452</td>
      <td>0.666567</td>
      <td>0.230578</td>
      <td>...</td>
      <td>0.137260</td>
      <td>0.226812</td>
      <td>130.850167</td>
      <td>30.525135</td>
      <td>4.000000</td>
      <td>0.454859</td>
      <td>0.496127</td>
      <td>0.256787</td>
      <td>6198.0</td>
      <td>056jpfChuMP5D1NMMaDXRR</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.278816</td>
      <td>0.262749</td>
      <td>0.634392</td>
      <td>0.140270</td>
      <td>0.596000</td>
      <td>0.166902</td>
      <td>0.192559</td>
      <td>0.341460</td>
      <td>0.596000</td>
      <td>0.166902</td>
      <td>...</td>
      <td>0.082210</td>
      <td>0.131105</td>
      <td>122.768255</td>
      <td>28.215783</td>
      <td>4.000000</td>
      <td>0.200000</td>
      <td>0.656235</td>
      <td>0.245299</td>
      <td>330.0</td>
      <td>05dTMGk8MjnpQg3bKuoXcc</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.228810</td>
      <td>0.251421</td>
      <td>0.600400</td>
      <td>0.178801</td>
      <td>0.612200</td>
      <td>0.192433</td>
      <td>0.179571</td>
      <td>0.336604</td>
      <td>0.612200</td>
      <td>0.192433</td>
      <td>...</td>
      <td>0.052150</td>
      <td>0.025935</td>
      <td>114.439167</td>
      <td>21.997673</td>
      <td>4.000000</td>
      <td>0.262613</td>
      <td>0.481787</td>
      <td>0.251199</td>
      <td>73.0</td>
      <td>070FVPBKvfu6M5tf4I9rt2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.394114</td>
      <td>0.362573</td>
      <td>0.599424</td>
      <td>0.151256</td>
      <td>0.541097</td>
      <td>0.289705</td>
      <td>0.203059</td>
      <td>0.332371</td>
      <td>0.541097</td>
      <td>0.289705</td>
      <td>...</td>
      <td>0.106724</td>
      <td>0.112448</td>
      <td>110.134788</td>
      <td>25.125111</td>
      <td>4.000000</td>
      <td>0.353553</td>
      <td>0.511997</td>
      <td>0.243171</td>
      <td>6173.0</td>
      <td>08vPKM3pmoyF6crB2EtASQ</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 26 columns</p>
</div>



Finally, the last step was to create the main dataframe using an inner merge on both the audio feature dataframe and artist dataframe. This inner merge meant a total of 126 playlists were lost (i.e., there was no overlap between the two dataframes across these playlists).



```python
# Concatenate the two dataframes
master_df = pd.merge(features_main_df, artist_df_groups, how='inner', on='ID')
master_df.head()
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
      <th>...</th>
      <th>'wrestling'</th>
      <th>'wrock'</th>
      <th>'ye ye'</th>
      <th>'yoik'</th>
      <th>'zapstep'</th>
      <th>'zeuhl'</th>
      <th>'zim'</th>
      <th>'zolo'</th>
      <th>'zydeco'</th>
      <th>'no_genre'</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.641282</td>
      <td>0.326942</td>
      <td>0.467911</td>
      <td>0.241057</td>
      <td>0.275940</td>
      <td>0.225821</td>
      <td>0.119650</td>
      <td>0.277109</td>
      <td>0.275940</td>
      <td>0.225821</td>
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
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.278816</td>
      <td>0.262749</td>
      <td>0.634392</td>
      <td>0.140270</td>
      <td>0.596000</td>
      <td>0.166902</td>
      <td>0.192559</td>
      <td>0.341460</td>
      <td>0.596000</td>
      <td>0.166902</td>
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
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.228810</td>
      <td>0.251421</td>
      <td>0.600400</td>
      <td>0.178801</td>
      <td>0.612200</td>
      <td>0.192433</td>
      <td>0.179571</td>
      <td>0.336604</td>
      <td>0.612200</td>
      <td>0.192433</td>
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
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.394114</td>
      <td>0.362573</td>
      <td>0.599424</td>
      <td>0.151256</td>
      <td>0.541097</td>
      <td>0.289705</td>
      <td>0.203059</td>
      <td>0.332371</td>
      <td>0.541097</td>
      <td>0.289705</td>
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
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.194509</td>
      <td>0.278470</td>
      <td>0.531067</td>
      <td>0.150001</td>
      <td>0.759400</td>
      <td>0.249805</td>
      <td>0.115499</td>
      <td>0.258020</td>
      <td>0.759400</td>
      <td>0.249805</td>
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
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 3245 columns</p>
</div>



The master dataframe was saved for EDA purposes next and final dataframe size was presented.



```python
master_df.to_csv('spotify_data_master.csv', sep=',')
```




```python
print("Number of Playlists: {}".format(master_df.shape[0]))
print("Number of Predictors: {}".format(master_df.shape[1]))
```


    Number of Playlists: 1420
    Number of Predictors: 3245


### String Parsing / Natural Language Processing

Here, we further analyze the names of the playlist based on the rationale that listeners search for key terms like 'Best', 'Hit', 'Workout' when they aim to find the relevant playlist. Due to the small size of our data, we adopted the string parsing approach for our model (which could be easily scaled with Python's NLTK package in larger models) as we don't want to exceed the dimensions of our model with too many predictors. 

- After reading in the full dataset and the playlist dataset, we perform a left join based on playlist ID and add the playlist name to the full dataset
- We search for 12 categories of specific strings that cover 'Best', 'Workout', 'Party', 'Chill', 'Acoustic', '2000s', '1990s', '1980s', '1970s', '1960s', and '1950s' using the str.contain function
- After creating these 12 boolean variables, we transform them to binary ones (0 or 1) by *1
- Lastly, we include those binary variables in the dataframe as predictor variables











```python
# Left Join by Playlist ID
new_df = pd.merge(full_df, playlist_df[['Name', 'ID']], on='ID', how='left')
new_df.shape
```





    (1420, 1494)













```python
# Search For Sub Strings
Str_Best = full_df_concise.Name.str.contains('Best|Top|Hit|best|top|hit|Hot|hot|Pick|pick')
Str_Workout = full_df_concise.Name.str.contains('Workout|workout|Motivation|motivation|Power|power|Cardio|')
Str_Party = full_df_concise.Name.str.contains('Party|party')
Str_Chill = full_df_concise.Name.str.contains('Chill|chill|Relax|relax')
Str_Acoustic = full_df_concise.Name.str.contains('Acoustic|acoustic')
Str_2000s = full_df_concise.Name.str.contains('20')
Str_1990s = full_df_concise.Name.str.contains('90|91|92|93|94|95|96|97|98|99')
Str_1980s = full_df_concise.Name.str.contains('80|81|82|83|84|85|86|87|88|89')
Str_1970s = full_df_concise.Name.str.contains('70|71|72|73|74|75|76|77|78|79')
Str_1960s = full_df_concise.Name.str.contains('60|61|62|63|64|65|66|67|68|69')
Str_1950s = full_df_concise.Name.str.contains('50s')

# Convert Boolean into Integers
Str_Best = Str_Best*1
Str_Workout = Str_Workout*1 
Str_Party = Str_Party*1
Str_Chill = Str_Chill*1
Str_Acoustic = Str_Acoustic*1
Str_2000s = Str_2000s*1
Str_1990s = Str_1990s*1
Str_1980s = Str_1980s*1
Str_1970s = Str_1970s*1
Str_1960s = Str_1960s*1
Str_1950s = Str_1950s*1

# Add to Dataframe
full_df_concise['Str_Best'] = Str_Best
full_df_concise['Str_Workout'] = Str_Workout
full_df_concise['Str_Party'] = Str_Party
full_df_concise['Str_Chill'] = Str_Chill
full_df_concise['Str_Acoustic'] = Str_Acoustic
full_df_concise['Str_2000s'] = Str_2000s
full_df_concise['Str_1990s'] = Str_1990s
full_df_concise['Str_1980s'] = Str_1980s
full_df_concise['Str_1970s'] = Str_1970s
full_df_concise['Str_1960s'] = Str_1960s
full_df_concise['Str_1950s'] = Str_1950s
```




```python
# Check New Column
full_df_concise.columns[-11:-1]
```





    Index(['Str_Best', 'Str_Workout', 'Str_Party', 'Str_Chill', 'Str_Acoustic',
           'Str_2000s', 'Str_1990s', 'Str_1980s', 'Str_1970s', 'Str_1960s'],
          dtype='object')



## Interaction Terms with Audio Features and Genre

The following section describes the process of creating interaction terms between genres and audio features. Interaction terms were considered because there may be different relationships between these features and the number of playlist followers depending on the genre. For example, different levels of energy may be more popular for rap music and acoustic music.

The first step was to bucket the genres (with a total of more than 100 specific genres) into broader categories. As seen below, some of the most common broad genres included: house, hip hop, pop, dance, r&b, acoustic, and soul. 



```python
broad_genres = ['house','hip hop','pop','dance','r&b','rap','acoustic','soul']

broad_genres = pd.DataFrame(np.zeros((full_df_concise.shape[0], len(broad_genres))), columns = broad_genres)
```




```python
for genre in broad_genres:  
    for data_col in full_df_concise.columns:
        if genre in data_col:
            indices = full_df_concise[(full_df_concise[data_col]==1)].index
            broad_genres[genre][indices] = 1

```


Next, interaction terms were made between these genre categories and certain audio features. Below are the interaction terms that were used. These features were selected through a separate analysis in which all of the genres, audio features, and all possible interactions were used as predictors to model the number of playlist followers. It was found that the interaction terms listed below were significant. 



```python
# Adding significant interaction terms from previous model
interaction_columns = ['house_acousticness_mean','hip hop_acousticness_std','pop_liveness_std','dance_liveness_std',
                      'r&b_acousticness_std','rap_energy_std','rap_key_std','acoustic_acousticness_std','acoustic_acousticness_mean',
                      'acoustic_energy_std','acoustic_key_std','soul_acousticness_std']


full_df_concise['house_acousticness_mean'] = broad_genres['house']*full_df_concise['acousticness_mean']
full_df_concise['hip hop_acousticness_std'] = broad_genres['hip hop']*full_df_concise['acousticness_std']
full_df_concise['pop_liveness_std'] = broad_genres['pop']*full_df_concise['liveness_std']
full_df_concise['dance_liveness_std'] = broad_genres['dance']*full_df_concise['liveness_std']
full_df_concise['r&b_acousticness_std'] = broad_genres['r&b']*full_df_concise['acousticness_std']
full_df_concise['rap_energy_std'] = broad_genres['rap']*full_df_concise['energy_std']
full_df_concise['rap_key_std'] = broad_genres['rap']*full_df_concise['key_std']
full_df_concise['acoustic_acousticness_std'] = broad_genres['acoustic']*full_df_concise['acousticness_std']
full_df_concise['acoustic_acousticness_mean'] = broad_genres['acoustic']*full_df_concise['acousticness_mean']
full_df_concise['acoustic_energy_std'] = broad_genres['acoustic']*full_df_concise['energy_std']
full_df_concise['acoustic_key_std'] = broad_genres['acoustic']*full_df_concise['key_std']
full_df_concise['soul_acousticness_std'] = broad_genres['soul']*full_df_concise['acousticness_std']

```




```python
full_df_concise[interaction_columns].describe()
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
      <th>count</th>
      <td>1420.000000</td>
      <td>1418.000000</td>
      <td>1418.000000</td>
      <td>1418.000000</td>
      <td>1418.000000</td>
      <td>1418.000000</td>
      <td>1418.000000</td>
      <td>1418.000000</td>
      <td>1420.000000</td>
      <td>1418.000000</td>
      <td>1418.000000</td>
      <td>1418.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.224109</td>
      <td>0.235339</td>
      <td>0.156279</td>
      <td>0.137165</td>
      <td>0.239961</td>
      <td>0.210305</td>
      <td>0.210305</td>
      <td>0.102606</td>
      <td>0.115892</td>
      <td>0.080324</td>
      <td>0.080324</td>
      <td>0.173310</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.212280</td>
      <td>0.144852</td>
      <td>0.056181</td>
      <td>0.073726</td>
      <td>0.143718</td>
      <td>0.094412</td>
      <td>0.094412</td>
      <td>0.150939</td>
      <td>0.190786</td>
      <td>0.117964</td>
      <td>0.117964</td>
      <td>0.162756</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.129984</td>
      <td>0.111896</td>
      <td>0.160846</td>
      <td>0.204460</td>
      <td>0.204460</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.221718</td>
      <td>0.302918</td>
      <td>0.155873</td>
      <td>0.149616</td>
      <td>0.306497</td>
      <td>0.238572</td>
      <td>0.238572</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.240984</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.366849</td>
      <td>0.341949</td>
      <td>0.185451</td>
      <td>0.180391</td>
      <td>0.344083</td>
      <td>0.267752</td>
      <td>0.267752</td>
      <td>0.285148</td>
      <td>0.228570</td>
      <td>0.220703</td>
      <td>0.220703</td>
      <td>0.332228</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.961000</td>
      <td>0.428986</td>
      <td>0.351859</td>
      <td>0.351859</td>
      <td>0.444861</td>
      <td>0.371096</td>
      <td>0.371096</td>
      <td>0.444861</td>
      <td>0.961000</td>
      <td>0.347747</td>
      <td>0.347747</td>
      <td>0.420705</td>
    </tr>
  </tbody>
</table>
</div>



The final dataframe has been created, and the next step is to model with the features created. 
