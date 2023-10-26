import os
from ML_Pipeline.data import get_train_data_gen, get_validation_data_gen
from ML_Pipeline.model import fit_model, hyperparam_tuning
from configparser import ConfigParser
from ML_Pipeline.plots import plot_metrics
from ML_Pipeline.evaluate_model import evaluate_model
import datetime
import ast

# Print the current working directory
print("Working directory: {}".format(os.getcwd()))

# Reading configuration from a config.ini file
config = ConfigParser()
config.read('config.ini')
print("Sections: ", config.sections())

# Extracting training parameters from the configuration
train_params = {}
train_params['architecture'] = config.get('train_params', 'architecture')
train_params['train_all_layers'] = config.getboolean('train_params', 'train_all_layers')
train_params['learning_rate'] = config.getfloat('train_params', 'learning_rate')
train_params['optimizer'] = config.get('train_params', 'optimizer')
train_params['dropout_rate'] = config.getfloat('train_params', 'dropout_rate')
train_params['earlystop_patience'] = config.getint('train_params', 'earlystop_patience')
train_params['epochs'] = config.getint('train_params', 'epochs')
print("######## Train Params ########")
print(train_params)

# Data paths
home_dir = config.get('paths', 'home_dir')
train_data_path = config.get('paths', 'train_data_path')
train_data_path = os.path.join(home_dir, train_data_path)
validation_data_path = config.get('paths', 'validation_data_path')
validation_data_path = os.path.join(home_dir, validation_data_path)
test_data_path = config.get('paths', 'test_data_path')
test_data_path = os.path.join(home_dir, test_data_path)
print('Train data path: {}'.format(train_data_path))
print('Validation data path: {}'.format(validation_data_path))
print('Test data path: {}'.format(test_data_path))

# Data generators for loading training, validation, and test data
train_datagen = get_train_data_gen(train_data_path)
validation_datagen = get_validation_data_gen(validation_data_path)
test_datagen = get_validation_data_gen(test_data_path)

# Initiate and train the model
save_model_ind = config.getboolean('train_params', 'save_model')

if config.getboolean('hp_tune_params', 'hp_tune') == False:
    print("\nInitiating model training...\n")
    history, model = fit_model(train_data=train_datagen, validation_data=validation_datagen,
                               save_model=save_model_ind, kwargs=train_params)
    print("\nModel Training complete\n")

    # Save plot and history
    plot_metrics(history.history, metrics=['loss', 'accuracy'])

    # Evaluate the model on test data
    print("\nInitiating model evaluation...\n")
    evaluate_model(model, test_datagen)
    print("\nModel Evaluation complete\n")
    print("Complete at {}".format(datetime.datetime.now()))

if config.getboolean('hp_tune_params', 'hp_tune') == True:
    print("\nInitiating Hyperparameter tuning...\n")
    hp_grid = config.get('hp_tune_params', 'param_grid')
    hp_grid = ast.literal_eval(hp_grid)
    
    hyperparam_tuning(hp_grid, train_data=train_datagen, validation_data=validation_datagen,
                      save_model=save_model_ind, train_params=train_params)
    print("Hyperparameter tuning complete at {}".format(datetime.datetime.now()))
