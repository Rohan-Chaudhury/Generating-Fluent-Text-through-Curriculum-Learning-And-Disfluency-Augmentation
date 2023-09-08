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

# Data Transformations

There are 2 types of transformations that we do on this data: 
1. **Synthetic Disfluency Augmentation (0 to 10 transformations)**: We run our repeats, interjections, and false starts transformations on the podcast transcripts.
2. **Disfluency Annotation Model (-1 transformation)**: We run Jamshid Lou and Johnson's (2020) model on the 0 podcast transcripts and remove words marked as disfluent. 
    * The repository for Jamshid Lou and Johnson (2020) is [here](https://github.com/pariajm/joint-disfluency-detector-and-parser). On this page, they note [here](https://github.com/pariajm/joint-disfluency-detector-and-parser#using-the-trained-models-for-disfluency-tagging) to use [this](https://github.com/pariajm/english-fisher-annotations) repository for "us\[ing\] the trained models to disfluency label your own data."
    * Steps to run it: 
        1. Use [these instructions](https://github.com/pariajm/english-fisher-annotations#using-the-model-to-annotate-fisher) to (1) clone the repo, (2) download the model and necessary resources for the model, and (3) unzip the model into this directory, `disfluency-spotify-wikipedia`.
        2. Create a virtual environment which meets [these requirements](https://github.com/pariajm/english-fisher-annotations#software-requirements) to run the script.
        3. Run the script [using these instructions](https://github.com/pariajm/english-fisher-annotations#using-the-model-to-annotate-fisher) on these files:
            * `mkdir spotify/annotated`
            * `$ python main.py --input-path ./spotify/0 --output-path ./spotify/annotated --model ./model/swbd_fisher_bert_Edev.0.9078.pt`
            * then run the script which takes annotated -> puts it in the -1 folder all cleaned up nicely!!!


# References
Paria Jamshid Lou and Mark Johnson. 2020. Improving disfluency detection by self-training a self-attentive model. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 3754–3763, Online. Association for Computational Linguistics.
