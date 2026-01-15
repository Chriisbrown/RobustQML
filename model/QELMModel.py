"""AutoEncoder model child class

Written 23/12/2025 cebrown@cern.ch
"""

import os

import numpy as np
import numpy.typing as npt

from model.AnomalyDetectionModel import ADModelFactory, ADModel
import strawberryfields as sf
from strawberryfields.ops   import *
from random                 import uniform      as r 


from sklearn.metrics import mean_absolute_error
# Register the model in the factory with the string name corresponding to what is in the yaml config
@ADModelFactory.register('AutoEncoderModel')
class AutoEncoderModel(ADModel):

    """AutoEncoderModel class

    Args:
        AutoEncoderModel (_type_): Base class of a AutoEncoderModel
    """

    def build_model(self, inputs_shape: tuple):
        """build model override, makes the model layer by layer

        Args:
            inputs_shape (tuple): Shape of the input
        """
        pass
        
       
        
    def substrate(self, n_modes, feats):
        
        """
        QELM quantum substrate
        
        NB: seed must be inside the definition to make sure the substrate is always 
            constant.
        """
        
        random.seed(42)


        eng     = sf.Engine('gaussian')#, backend_options={'cutoff_dim': cutoff})
        circ    = sf.Program(n_modes)
        
        with circ.context as q:
            
            ### Data Embedding        
            for i in range(n_modes):
                # Dgate(2*feats[i]) | q[i]
                Dgate(feats[i]) | q[i]
            
            ### Substrate
            for i in range(n_modes - 1):
                CXgate(r(0, 1)) | (q[i], q[i+1])
            CXgate(r(0, 1)) | (q[-1], q[0])
            
        if eng.run_progs:
            eng.reset()
            
        state = eng.run(circ).state
        probs = []
        for i in range(n_modes):
            mPhoton = state.mean_photon(mode=i)
            probs.append(mPhoton[0])
            probs.append(mPhoton[1])
            del mPhoton


        del(eng)
        del(circ)
        
        return np.array(probs)



    def compile_model(self):
        """compile the model generating callbacks and loss function
        Args:
        """
        

    def fit(
        self,
        train: npt.NDArray[np.float64],
    ):
        """Fit the model to the training dataset

        Args:
            X_train (npt.NDArray[np.float64]): X train dataset
            
        """
        
        model_inputs = []


        for dat in train:
            probs = self.substrate(10, dat)
            model_inputs.append(np.concatenate((probs, dat)))
            del(probs)
            
        model_inputs = np.array(model_inputs)

        # Train the model using hyperparameters in yaml config
        keras.config.disable_traceback_filtering()
        
        history = self.AD_model.fit(
            train.to_numpy(),
            train.to_numpy(),
            epochs=self.training_config['epochs'],
            batch_size=self.training_config['batch_size'],
            verbose=self.run_config['verbose'],
            validation_split=self.training_config['validation_split'],
            shuffle=True,
            callbacks=self.callbacks,
        )
        
        self.history = history.history
        
        
    def predict(self, X_test) -> float:
        """Predict method for model

        Args:
            X_test (npt.NDArray[np.float64]): Input X test

        Returns:
            float: model prediction
        """
        model_outputs = self.AD_model.predict(X_test)
        ad_scores = tf.keras.losses.mae(model_outputs, X_test)
        ad_scores = ad_scores._numpy()
        return ad_scores

    # Decorated with save decorator for added functionality
    @ADModel.save_decorator
    def save(self, out_dir: str = "None"):
        """Save the model file

        Args:
            out_dir (str, optional): Where to save it if not in the output_directory. Defaults to "None".
        """
        # Export the model
        os.makedirs(os.path.join(out_dir, 'model'), exist_ok=True)
        # Use keras save format !NOT .h5! due to depreciation
        export_path = os.path.join(out_dir, "model/saved_model.keras")
        self.AD_model.save(export_path)
        print(f"Model saved to {export_path}")

    @ADModel.load_decorator
    def load(self, out_dir: str = "None"):
        """Load the model file

        Args:
            out_dir (str, optional): Where to load it if not in the output_directory. Defaults to "None".
        """
        # Load the model
        self.AD_model = load_model(f"{out_dir}/model/saved_model.keras")