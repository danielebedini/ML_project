# Machine Learning project
Exam project for the university course of Machine Learning from the Master Degree in Artificial Intelligence. 
Prof. Alessio Micheli.

## Important!
Before executing the code, read carefully the file "requirements.txt". We used the most famous libraries such as NumPy and MatPlotLib, but for the right version we reccomend to check the requirements file.

All the files must be executed from the most external path, like in the command written below.

## How to execute the model
For saving space, we did not include the dataset that we used. If you want to use a specific dataset, you can add them in the folder "data/monk" or "data/cup" and then execute the selected model.

Once you have selected the model and the dataset, you can run the following command:
```
python [folder of the model]/model_name.py
```
monk network:
```
python monkModels/monk_1_net.py
```
monk kfold:
```
python monkModels/monk_1_kfold.py
```
cup blind test:
```
python cupModels/cup_model_blind_test.py
```
In the *utilities.py* file you can find all the other utility methods such as one hot encoding, methods for the plots and a method to print the progress bar.

## Other notes
In general, for training usually it takes less than 10 seconds.

For the final cup model, it takes about 1 minute, because it is an ensemble of 10 models, based on the selected one.

These numbers refers to an Apple Silicon M1 and M2, with a NumPy version that exploits GPU accelleration.
