---
title: Spotify Playlist Data Analysis
---

Data Science I | Harvard University | Fall 2017


## Introduction

Spotify is a music, podcast, and video streaming service. It provides digital rights management - protected content from record labels and media companies. One of Spotify’s primary products is Playlists, collections of tracks that users (and / or Spotify) can build for any mood or event. With over 40 million songs available, the company attempts to direct the most relevant songs to users based on their preferences.
These Playlists are compiled in a complex manner, involving both human-led and computer-led processes. What stands is that algorithmically curated discovery playlists, and their effectiveness, remain an important business interest for the company. 

The overarching goal of this project is to understand how these algorithms can be evaluated and improved with machine learning techniques.

## Problem Statement and Motivation

Spotify’s business model is, to a significant extent, centered around providing its users with relevant songs based on user inputs and historical preferences. Being able to recommend appropriate playlists to its users is hence of vital importance. With this motivation in mind, the two problem statements are:

**1. What predictors and what model can be used to determine the success of a Spotify playlist (i.e., number of followers) more accurately out-of-sample than a simple baseline model and how well do these predictors match with expectations gained from exploratory data analysis?**

**2. Using this improved model, generate playlists according to user-specified filters such that the resultant playlists are deemed to have a high probability of being successful.**

## Introduction and Description of Data

**High-level Data Statistics**

| Metric            | Statistic  |
|:-----------------:|:----------:|
| Unique Playlists  | 1,420      |
| Unique Tracks     | 72,789     |
| Unique Artists    | 28,915     |
| Unique Predictors | >3,000     |

Please refer to [Data Wrangling](https://toledy.github.io/spotify/data_mining_and_wrangling.html) for more information on the data mining and wrangling procedures employed.

Please refer to [EDA](https://toledy.github.io/spotify/eda.html) for more in-depth insights into the exploratory data analysis.

## Modeling Approach and Project Trajectory

Please refer to [Baseline Models](https://toledy.github.io/spotify/baseline_models.html) for more information on the baseline models employed.

Please refer to [Ensemble Models](https://toledy.github.io/spotify/ensemble_methods.html) for more information on the advanced models employed.


## Results, Conclusions, and Future Work

To come

## Literature Review / Related Work / Sources

We would be remiss not to mention both the Spotify API:

[Spotify Developer API](https://developer.spotify.com)

As well as the useful Python API extension:

[Spotipy Python Library](https://github.com/plamere/spotipy)

Finally, of course:

[Harvard Data Science I](https://cs109.github.io/a-2017/)