import pandas as pd
import tensorflow as tf
import numpy as np

from .. import model as model

class Normal(object):

	def __init__(self):
		"""
		Contains information on whether the family has certain kinds of latent variables - e.g. a dispersion (scale) latent variable -
		as well as information on how many additional distributional variables are required for this distribution
		"""
		self.scale = True
		self.dof = False
		self.additional_variables = 1

	@staticmethod
	def neg_loglikelihood(Y, theta, scale, dof):
		"""
		Creates the negative loglikelihood for the model

		Parameters
        -------
        Y - tf.placeholder
        	The data for the target variable

        theta - tf.placeholder
        	the location parameter for the distribution

        scale - tf.placeholder
        	the dispersion parameter for the distribution

        dof - tf.placeholder
        	the degrees of freedom parameter for the distribution

		Returns
        -------
        - An tf.placeholder with the negative loglikelihood 
		"""
		return -tf.contrib.distributions.Normal(theta, scale).log_likelihood(Y)

class Poisson(object):

	def __init__(self):
		"""
		Contains information on whether the family has certain kinds of latent variables - e.g. a dispersion (scale) latent variable -
		as well as information on how many additional distributional variables are required for this distribution
		"""
		self.scale = False
		self.dof = False
		self.additional_variables = 0

	@staticmethod
	def neg_loglikelihood(Y, theta, scale, dof):
		"""
		Creates the negative loglikelihood for the model

		Parameters
        -------
        Y - tf.placeholder
        	The data for the target variable

        theta - tf.placeholder
        	the location parameter for the distribution

        scale - tf.placeholder
        	the dispersion parameter for the distribution

        dof - tf.placeholder
        	the degrees of freedom parameter for the distribution

		Returns
        -------
        - An tf.placeholder with the negative loglikelihood 
		"""

		return tf.reduce_mean(-tf.reduce_sum(Y*theta - tf.exp(theta)))


class StudentT(object):

	def __init__(self):
		"""
		Contains information on whether the family has certain kinds of latent variables - e.g. a dispersion (scale) latent variable -
		as well as information on how many additional distributional variables are required for this distribution
		"""
		self.scale = True
		self.dof = True
		self.additional_variables = 2

	@staticmethod
	def neg_loglikelihood(Y, theta, scale, dof):
		"""
		Creates the negative loglikelihood for the model

		Parameters
        -------
        Y - tf.placeholder
        	The data for the target variable

        theta - tf.placeholder
        	the location parameter for the distribution

        scale - tf.placeholder
        	the dispersion parameter for the distribution

        dof - tf.placeholder
        	the degrees of freedom parameter for the distribution

		Returns
        -------
        - An tf.placeholder with the negative loglikelihood 
		"""
		return -tf.contrib.distributions.StudentT(dof, theta, scale).log_likelihood(Y)


