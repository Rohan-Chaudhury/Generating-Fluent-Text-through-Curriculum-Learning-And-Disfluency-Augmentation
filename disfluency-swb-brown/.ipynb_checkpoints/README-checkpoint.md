# README

# Datasets

## Swb

The Threebank-3 Dataset can be obtained by purchasing at [this link](https://catalog.ldc.upenn.edu/LDC99T42), as access to the datset is controlled by the Linguistic Data Consortium (LDC) at the University of Pennsylvania. We utilize the tree files in: treebank_3/parsed/mrg/swbd. 

`swb` contains the transcripts (derived from the tree files) and their transformations (according to our synthetic data transformations).


## Brown

The Brown Dataset can be obtained by going to [this link](https://github.com/google-research-datasets/wiki-split). We utilize the tree files in: treebank_3/parsed/mrg/brown. 

`brown` contains the transcripts (derived from the tree files) and their transformations (according to our synthetic data transformations).

# Running this code
1. Go to tb.py and edit the PTB_base_dir variable, so it contains the absolute path to your copy of the treebank_3 datset. 
2. Go to utils_general.py and edit the PATHS at the top of the file. 
3. Create the swb and brown datasets by running their notebooks (Swb.ipynb and Brown.ipynb).
