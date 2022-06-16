# Generalization properties of learning algorithms and Sharpness Aware Minimization based on Minimum Sharpness
### CS-439: Optimization for machine learning
**Description**:
One of the biggest challenges in deep learning is an understanding generalization. Sharpness is one of the indicators of generalization properties that perform well in practice. Moreover, sharpness aware minimization (SAM) is a new state-of-the-art technique based on simultaneously minimizing both loss and sharpness. In this work, we investigate the recently introduced notion of sharpness, known as minimum sharpness. We investigate its correlation with the generalization gap, by considering many different optimizers and SAM. Finally, we tackle the question of adaptivity of learning algorithms as that also has an impact on generalization, and investigate how the choice of optimizer influences sharpness. 

### Project structure
- The folder `/checkpoints` contains results from current runs that you do within the repository once you fork it.
- The folder `/checkpoints_test` contains some precomputed checkpoints for the model illustrated in the `TrainingSample.ipynb` notebook. The structure of the checkpoints folder is:
  - `/DATASET` (_FashionMNIST_/_CIFAR10_)
    - `/MODEL_ARCHITECTURE` (_SimpleBatch_/_MiddleBatch_/_ComplexBatch_)
      - `/epochX` (_50_/_100_/_150_/_200_): We train all models up to 200 epochs and save the checkpoints every 50 epochs.
      - `/converged`: Whenever the model converges (loss is lower than tolerance set), we save again the checkpoints.
- The folder `/data` should be empty by default and will be populated with data when training the models.
- The folder `/results` contains .csv files with results for each of the datasets.
- The folder `/optimizers` contains implementation of AdaBound, AdaShift and SAM in torch, collected from external sources.
- The folder `/sharpness` contains the approximate calculation of the Hessian and sharpness
- Notebooks `TrainingSample.ipynb`, `DataAnalysis.ipynb` illustrate our work and are presented below.
- The files within the repository represent:
<br />`models.py` - Contains the architecture of the models we considered.
<br />`main.py` - Able to run the trainings and computation for a given configuration. A configuration is given by *dataset*, *model architecture*, *optimizer*
<br />`helpers.py` - Various utils used for training, testing, computation, data preprocessing.
### Running the code

We require installation of *Python*. The needed libraries are stated in `requirements.txt`, to install them run: `pip install -r requirements.txt`,
or `pip3 install -r requirements.txt` (Python 3).
1. To explore our work, we encourage you to look through our notebooks:
- `TrainingSample.ipynb` allows you to train a model and compute sharpness for a given dataset, architecture and optimizer
- `DataAnalysis.ipynb` loads all the results from trainings and prepares the plots. If results are missing, it requires you to download all the existing checkpoints from training to extract the results, which might take a longer time, or alternatively retrain the models.
All the existing checkpoints are available at: https://drive.google.com/drive/folders/10LuJDXzP6P_xH-z66Kh4KaWPfR1s0-t9?usp=sharing
However, due to limited size on Github, we have not added them here (>30GB).

2. For running a model for a given configuration, we also offer a runnable Python file:
- `python main.py train $dataset $model $optimizer $use_sam $load_existing` allows you to train a model
  - _dataset_ should be `CIFAR10` or `FashionMNIST`
  - _model_ should be `SimpleBatch`, `MiddleBatch` or `ComplexBatch`
  - _optimizer_ should be `SGD`, `PHB`, `Adagrad`, `Adam`, `AdaShift`, `AdaBound`
  - _use_sam_ should be 0 (do not use) or 1 (use)
  - _load_existing_ is not used here, it can be 0 or 1.
- `python main.py compute_sharpness $dataset $model $optimizer $use_sam $load_existing` allows you to compute sharpness for the given model. All params stay the same, except for:
  - _load_existing_ should be 1 if you trained the model already and would like to load from file, 0 otherwise
- `python main.py plot $dataset $model $optimizer $use_sam $load_existing` allows you to visualize the computations

3. To automatically run all the configuration (dataset, optimizer, arhitecture), we offer you some shell scripts which can be run as:
- To train all models, run: `chmod +x train_all` and `./train_all`
- To compute sharpness for all trained models, run: `chmod +x compute_sharpness` and `./compute_sharpness_all.sh`

### Authors

- Jana Vuckovic: jana.vuckovic@epfl.ch
- Miguel-Angel Sanchez Ndoye: miguel-angel.sanchezndoye@epfl.ch
- Irina Bejan: irina.bejan@epfl.ch 