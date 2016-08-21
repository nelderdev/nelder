from patsy import dmatrices, dmatrix
import pandas as pd
import tensorflow as tf
import numpy as np

from .. import model as model

from .families import Poisson

class GLM(model.Model):
    """
    Implements GLM based models; inherits from parent Model class which contains inference methods
    """
    def __init__(self, formula, data, family):

        self.model_name = "GLM"
        self.supported_methods = ["MLE", "PML", "Laplace", "M-H", "BBVI"]
        self.default_method = "MLE"
        self.multivariate_model = False

        # Format the data
        self.is_pandas = True # This is compulsory for this model type
        self.data_original = data.copy()
        self.formula = formula
        self.y, self.X = dmatrices(formula, data)
        self.y_name = self.y.design_info.describe()
        self.X_names = self.X.design_info.describe().split(" + ")
        self.z_no = self.X.shape[1]
        self.data_name = self.y_name
        self.y = np.array([self.y.ravel()]).T
        self.data = self.y.copy()
        self.X = np.array([self.X])[0]
        self.index = data.index

        # Create latent variables
        self._create_latent_variables()

        # Obtain family negative loglikelihood
        self.family = family
        try:
            self.neg_loglikelihood = family.neg_loglikelihood
            self.z_no += family.additional_variables
        except ValueError:
            print("Could not find negative loglikelihood method and/or additional variable count for this family!")

    def _create_latent_variables(self):
        """
        Create the latent variables to be used in the model
        """
        pass

    def _model(self):
        """
        Creates model matrices for the GLM
        
        Returns
        -------
        An tf.InteractiveSession
        """
        X = tf.placeholder(tf.float32, [None, self.X.shape[1]])
        Y = tf.placeholder(tf.float32, [None, 1])
        beta = tf.Variable(self.family.get_initial_values(len(self.X_names)))
        theta = tf.matmul(X, tf.gather(beta, list(range(self.X.shape[1]))))

        # Check for dispersion latent variables
        if self.family.scale is False:
            scale = tf.constant(0.0)
        else:
            scale = tf.exp(tf.gather(beta, [self.z_no-1])) # change to something else

        # Check for dispersion latent variables
        if self.family.dof is False:
            dof = tf.constant(0.0)
        else:
            dof = tf.exp(tf.gather(beta, [self.z_no-2])) # change to something else

        return beta, X, Y, self.neg_loglikelihood(Y, theta, scale, dof)