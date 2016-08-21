from patsy import dmatrices, dmatrix
import pandas as pd
import tensorflow as tf
import numpy as np

class Model(object):

    @staticmethod
    def initialize_session():
        """
        Initializes a TensorFlow Interactive Session (if one does not exist)
        
        Returns
        -------
        An tf.InteractiveSession
        """
        global NELDER_INTERACTIVE_SESSION

        if tf.get_default_session() is None:
            NELDER_INTERACTIVE_SESSION = tf.InteractiveSession()

        return NELDER_INTERACTIVE_SESSION

    def fit(self, mode, **kwargs):
        """
        Fits latent variables for the model
        """
        
        # Fitting optional arguments
        batch_size = kwargs.get('batch_size', 10)

        if mode == "BGD":
            self.batch_mle()

        elif mode == "SGD":
            self.stochastic_mle(batch_size=batch_size)

        elif mode == "OGD":
            self.online_mle()

    def batch_mle(self):
        """
        This finds the MLE estimate of latent variables using batch gradients - that is it uses every datapoint to compute the gradient.
        It uses the ADAM optimizer to perform stochastic optimization.
        """
        beta, X, Y, neg_loglikelihood = self._model()

        optimizer = tf.train.AdamOptimizer().minimize(neg_loglikelihood)

        # Initializing the variables
        init = tf.initialize_all_variables()

        # Launch the graph
        sess = self.initialize_session()
        sess.run(init)
        for iteration in range(500):
            c = sess.run([optimizer, neg_loglikelihood], feed_dict={X: self.X, Y: self.y})
        print(sess.run(beta))

    def online_mle(self):
        """
        This finds the MLE estimate of latent variables using online stochastic gradients - that is the number of iterations is equal to the
        number of datapoints, which it iterates through sequentially to obtain a latent variable estimate at each timepoint.
        """        
        beta, X, Y, neg_loglikelihood = self._model()

        optimizer = tf.train.AdamOptimizer().minimize(neg_loglikelihood)

        # Initializing the variables
        init = tf.initialize_all_variables()

        # Launch the graph
        sess = self.initialize_session()
        sess.run(init)

        for iteration in range(1,self.X.shape[0]):
            c = sess.run([optimizer, neg_loglikelihood], feed_dict={X: self.X[iteration-1:iteration,:], Y: self.y[iteration-1:iteration]})
        print(sess.run(beta))

    def stochastic_mle(self, batch_size=10):
        """
        This finds the MLE estimate of latent variables using stochastic gradients - that is it uses a subset of datapoints at each iteration to
        compute the gradient. It uses the ADAM optimizer to perform stochastic optimization.
        """
        beta, X, Y, neg_loglikelihood = self._model()

        optimizer = tf.train.AdamOptimizer().minimize(neg_loglikelihood)

        # Initializing the variables
        init = tf.initialize_all_variables()

        # Launch the graph
        sess = self.initialize_session()
        sess.run(init)

        for iteration in range(500):
            batch_indices = np.random.randint(self.X.shape[0], size=batch_size)
            c = sess.run([optimizer, neg_loglikelihood], feed_dict={X: self.X[batch_indices,:], Y: self.y[batch_indices]})
        print(sess.run(beta))