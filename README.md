# Quick Setup
Everything is done directly under the RobustQML directory

Activate the environment
```
conda activate robust
```
Run this to add the scripts in this directory to your python path
```
export PYTHONPATH=$PYTHONPATH:$PWD
```
Choose which model to run from the configs in `model/configs` e.g. a standard AutoEncoder Model
```
export Model=baseline
```
Prepare the data. This step downloads the data from [hugging face](https://huggingface.co/datasets/fastmachinelearning/collide-1m). It will take a while. Hugging face will cache the downloads so the next time this command is run it will be faster. Currently this will generate a 'trigger' like set of inputs, 5 jets (pT, eta, phi), 4 electrons (pT, eta, phi) and 4 muons (pT, eta, phi). This can be changed in data/dataset.py L68. Currently the only feature transformation performed is to take the top 5,4,4 jets,electrons,muons in pT or pad the input vector if there are fewer than these values. Later a 0-1 feature scaling is performed before feeding the inputs to the model for training. 
```
python data/process_entire_set.py
```
Train the model
```
python train/train.py -y model/configs/$Model.yaml -o output/$Model
```
Make some basic validation plots
```
python plot/test.py python plot/test.py -o output/$Model
```

# Adding a new model

To add a new model class to the repo there are a number of steps needed: First create a file MyModel.py in the model directory This python script must contain a uniquely named model class that inherits from ADModel and is registered with the model factory:
```
@ADModelFactory.register('MyModel')
class MyModel(ADModel):
    """MyModel class

    Args:
        ADModel (_type_): Base class of a ADModel
    """
This ADModelFactory.register('MyModel') allows you to generate your model class directly from the yaml config.

The rest of the MyModel.py is up to you but you must include the following methods with the following arguments:

build_model(input_shape,output_shape)
# Shape of the input and output of the model, derived from the X_train.shape[1:]  and y_train.shape[1:] (avoiding the batch size dimension)
compile_model()
fit( X_train)
# Each passed as a numpy array
@ADModel.save_decorator
save(out_dir)
# The save decorator is required.
@ADModel.load_decorator
load(out_dir)
# The load decorator is required.
```

Secondly, add your model to the model/__init__.py as from model.MyModel import MyModel

Finally, add your config yaml to model/configs

The config must follow in style to the others present and contains your model hyperparameters. The hyperparameters are automatically loaded when the model is created either from the original yaml or from a saved model folder. They are accessed in your model class with, for example, self.model_config['name']. The internal hyperparameters in each dictionary are for you to decide and access when building and compiling your model. Below is the minimum requirements for your yaml config:

model: MyModel #! This must be the same as the name you registered you model with in the JetModelFactory.register
run_config :
  verbose : 2
  debug : True

model_config :
  name : baseline

training_config :
  validation_split : 0.1

To train your new model just specify the new yaml when training e.g.

python train/train.py -y model/configs/mymodel.yaml -o output/mymodel
