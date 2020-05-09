# Image-Classifier

In this project, I build a Python application that can train an image classifier on a dataset, then predict new images using the trained model.

## Train.py
Prints out training loss, validation loss, and validation accuracy as the network trains

### Options:

* Basic usage: `python train.py data_directory`


* Set directory to save checkpoints: `python train.py data_dir --save_dir save_directory`

* Choose architecture: `python train.py data_dir --arch "vgg13"`

* Set hyperparameters: `python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20`

* Use GPU for training: `python train.py data_dir --gpu`

## Predict.py
Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

### Options:

* Basic usage: `python predict.py /path/to/image checkpoint`

* Return top KK most likely classes: `python predict.py input checkpoint --top_k 3`

* Use a mapping of categories to real names: `python predict.py input checkpoint --category_names cat_to_name.json`

* Use GPU for inference: `python predict.py input checkpoint --gpu`
