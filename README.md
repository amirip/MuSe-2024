# MuSe-2024 Baseline Model: GRU Regressor


[Homepage](https://www.muse-challenge.org) || [Baseline Paper](https://www.researchgate.net/publication/380664467_The_MuSe_2024_Multimodal_Sentiment_Analysis_Challenge_Social_Perception_and_Humor_Recognition)


## Sub-challenges and Results 
For details, please see the [Baseline Paper](https://www.researchgate.net/publication/380664467_The_MuSe_2024_Multimodal_Sentiment_Analysis_Challenge_Social_Perception_and_Humor_Recognition). If you want to sign up for the challenge, please fill out the form 
[here](https://www.muse-challenge.org/challenge/participate).

* MuSe-Perception: predicting 16 different dimensions of social perception (e.g. Assertiveness, Likability, Warmth,...). 
 *Official baseline*: **.3573** mean Pearson's correlation over all 16 classes.

* MuSe-Humor: predicting the presence/absence of humor in cross-cultural (German/English) football press conference recordings. 
*Official baseline*: **.8682** AUC.


## Installation
It is highly recommended to run everything in a Python virtual environment. Please make sure to install the packages listed 
in ``requirements.txt`` and adjust the paths in `config.py` (especially ``BASE_PATH`` and ``HUMOR_PATH`` and/or ``PERCEPTION_PATH``, respectively). 

You can then, e.g., run the unimodal baseline reproduction calls in the ``*_full.sh`` file provided for each sub-challenge.

## Settings
The ``main.py`` script is used for training and evaluating models.  Most important options:
* ``--task``: choose either `perception` or `humor` 
* ``--feature``: choose a feature set provided in the data (in the ``PATH_TO_FEATURES`` defined in ``config.py``). Adding 
``--normalize`` ensures normalization of features (recommended for ``eGeMAPS`` features).
* Options defining the model architecture: ``d_rnn``, ``rnn_n_layers``, ``rnn_bi``, ``d_fc_out``
* Options for the training process: ``--epochs``, ``--lr``, ``--seed``,  ``--n_seeds``, ``--early_stopping_patience``,
``--reduce_lr_patience``,   ``--rnn_dropout``, ``--linear_dropout``
* In order to use a GPU, please add the flag ``--use_gpu``
* Predict labels for the test set: ``--predict``
* Specific parameter for MuSe-Perception: ``label_dim`` (one of the 16 labels, cf. ``config.py``), ``win_len`` and ``hop_len`` for segmentation.

For more details, please see the ``parse_args()`` method in ``main.py``.

## Reproducing the baselines 
Please note that exact reproducibility can not be expected due to dependence on hardware. 
### Unimodal models
For every challenge, a ``*_full.sh`` file is provided with the respective call (and, thus, configuration) for each of the precomputed features.
Moreover, you can directly load one of the checkpoints corresponding to the results in the baseline paper. Note that 
the checkpoints are only available to registered participants. 

A checkpoint model can be loaded and evaluated as follows:

`` main.py --task humor --feature faus --eval_model /your/checkpoint/directory/humor_faus/model_102.pth`` 


### Late Fusion
We utilize a simple late fusion approach, which averages different models' predictions. 
First, predictions for development and test set have to be created using the ``--predict`` option in ``main.py``. 
This will create prediction folders under the folder specified as the prediction directory in ``config.py``.

Then, ``late_fusion.py`` merges these predictions:
* ``--task``: choose either `humor` or `perception` 
* ``--label_dim``: for MuSe-Perception, cf. ``PERCEPTION_LABELS`` in ``config.py``
* ``--model_ids``: list of model IDs, whose predictions are to be merged. These predictions must first be created (``--predict`` in ``main.py`` or ``personalisation.py``). 
  The model id is a folder under the ``{config.PREDICTION_DIR}/humor`` for humor and ``{config.PREDICTION_DIR}/perception/{label_dim}`` for perception. 
  It is the parent folder of the folders named after the seeds (e.g. ``101``). These contain the files ``predictions_devel.csv`` and ``predictoins_test.csv``
* ``--seeds``: seeds for the respective model IDs.  

### Model Checkpoints
Checkpoints for the [Perception Sub-Challenge](https://mediastore.rz.uni-augsburg.de/get/Bm2Ds0KUNd/)

Checkpoints for the [Humor Sub-Challenge](https://mediastore.rz.uni-augsburg.de/get/_Xvipe7oPO/)


##  Citation:

The MuSe2024 baseline paper is only available in a preliminary version as of now: [https://www.researchgate.net/publication/380664467_The_MuSe_2024_Multimodal_Sentiment_Analysis_Challenge_Social_Perception_and_Humor_Recognition](https://www.researchgate.net/publication/380664467_The_MuSe_2024_Multimodal_Sentiment_Analysis_Challenge_Social_Perception_and_Humor_Recognition)

