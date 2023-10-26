import tensorflow as tf
import tensorflow.keras.models as models
from tensorflow.keras import layers
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import EarlyStopping

def get_pre_trained_model_inception(train_all_layers=False):
    # Initialize a pretrained InceptionV3 model with custom settings
    model = InceptionV3(include_top=False, input_shape=(150, 150, 3), weights='imagenet')

    # Freeze layers if not training all layers
    if not train_all_layers:
        for layer in model.layers:
            layer.trainable = False

    last_layer = model.get_layer("mixed7")
    last_output = last_layer.output

    return model, last_output

def get_pre_trained_model_resnet(train_all_layers=False):
    # Initialize a pretrained ResNet50 model with custom settings
    model = ResNet50(include_top=False, weights='imagenet', input_shape=(150, 150, 3))

    # Freeze layers if not training all layers
    if not train_all_layers:
        for layer in model.layers:
            layer.trainable = False

    output = model.layers[-1].output

    return model, output

def get_optimizer(optimizer_name="adam", lr=0.0001):
    # Initialize the optimizer (Adam or RMSprop) with a learning rate
    assert optimizer_name == "adam" or optimizer_name == "rmsprop"
    if optimizer_name == "adam":
        optimizer = Adam(learning_rate=lr)
    elif optimizer_name == "rmsprop":
        optimizer = RMSprop(learning_rate=lr)
    return optimizer

def define_model(model="inception_v3", train_all_layers=False, optimizer='rmsprop', learning_rate=0.0001, dropout=0.2):
    # Define and compile a custom model based on a pretrained model

    if model == "inception_v3":
        model, last_output = get_pre_trained_model_inception(train_all_layers=train_all_layers)
    elif model == "resnet":
        model, last_output = get_pre_trained_model_resnet(train_all_layers=train_all_layers)

    # Add custom layers on top of the pretrained model
    x = layers.Flatten()(last_output)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(model.input, x)

    optimizer = get_optimizer(optimizer_name=optimizer, lr=learning_rate)

    # Compile the model
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model

def get_callbacks(patience=5):
    # Return early stopping callback with specified patience
    es_callback = EarlyStopping(monitor='val_loss', patience=patience)
    return [es_callback]

def fit_model(train_data, validation_data, save_model=False, **kwargs):
    # Fit and train the model

    train_params = kwargs['kwargs']
    print(train_params)

    model = define_model(
        model=train_params['architecture'], train_all_layers=train_params['train_all_layers'],
        optimizer=train_params['optimizer'], learning_rate=train_params['learning_rate'],
        dropout=train_params['dropout_rate'])

    callbacks = get_callbacks(train_params['earlystop_patience'])

    history = model.fit(train_data, validation_data=validation_data,
                        epochs=train_params['epochs'], steps_per_epoch=None,
                        validation_steps=None, verbose=1, callbacks=callbacks)

    if save_model:
        model.save("../output/models/{}.h5".format(train_params['architecture']))

    return history, model

def hyperparam_tuning(param_grid, train_data, validation_data, save_model, train_params):
    # Hyperparameter tuning by trying different values in the param_grid
    print(param_grid)
    for param in param_grid.keys():
        print("Tuning {}".format(param))
        if param == 'learning_rate':
            for lr in param_grid[param]:
                print("\nTrying with {0}={1}".format(param, lr))
                train_params[param] = lr
                fit_model(train_data=train_data, validation_data=validation_data, save_model=save_model, kwargs=train_params)
