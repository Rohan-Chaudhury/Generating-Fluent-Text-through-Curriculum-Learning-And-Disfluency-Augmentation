# README

# Datasets

## Spotify

The Spotify Podcasts Dataset can be obtained by applying at [this link](https://podcastsdataset.byspotify.com/), as access to the datset is controlled by Spotify. 

`spotify` contains the transcripts and their transformations.

## Wikipedia

The WikiSplit Dataset can be obtained by going to [this link](https://github.com/google-research-datasets/wiki-split). We use the 0th column of the test set for our experiments.

`wikipedia` contains the sentences and their transformations.

# Running this code
1. Go to utils_general.py and edit the PATHS at the top of the file. 
2. Create the spotify and wikipedia datasets by running their notebooks (Spotify.ipynb and Wikipedia.ipynb).