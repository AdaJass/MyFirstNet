# Copyright (c) 2015 ev0
#
# Permission to use, copy, modify, and distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

import numpy as np
from convae import *

def testMnist():
	"""	Test convolutional autoencoder on MNIST dataset.
	"""
	print("Loading MNIST images...")
	data = np.load('data/mnist.npz')
	train_data = data['train_data'][0:1000].reshape(1000, 28, 28, 1)
	valid_data = data['valid_data'][0:1000].reshape(1000, 28, 28, 1)
	train_data = np.concatenate((train_data, valid_data))
	test_data = data['test_data'][0:1000].reshape(1000, 28, 28, 1)

	print("Creating network...")

	layers = [
				PoolLayer((2, 2), 'max'),
				ConvLayer(6, 1, (7, 7), stride=3)
			]

	hyperparams = {

		'layer_wise': False,
		'no_images': 12,
		'conv':[
			{
				'epochs': 20,
				'batch_size': 500,
				'view_kernels': False,
				'view_recon': True,
				'no_images': 5,
				'pert_prob': 0.5,
				'eps_w': 0.0007,
				'eps_b': 0.0007,
				'eps_decay': 9,
				'eps_intvl': 30,
				'eps_satr': 'inf',
				'mu': 0.7,
				'l2': 0.95,
				'RMSProp': True,
				'RMSProp_decay': 0.9,
				'minsq_RMSProp': 0,
			}
		]
	}

	ae = ConvAE(layers)
	ae.train(train_data, test_data, hyperparams)


def testTorontoFaces():
	"""
	Test autoencoder on Toronto Faces dataset.
	"""

	print("Loading Toronto Facial images...")
	data = np.load('data/faces.npz')
	train_data = np.transpose(data['train_data'], (2, 0, 1)).reshape(2925, 32, 32, 1)
	test_data = np.transpose(data['test_data'], (2, 0, 1)).reshape(418, 32, 32, 1)

	print("Creating network...")

	layers = [
				ConvLayer(6, 1, (3, 3), outputType='linear') #do PCA
			]

	hyperparams = {

		'layer_wise': False,
		'no_images': 12,
		'conv': [
			{
				'epochs': 50,
				'batch_size': 500,
				'view_kernels': False,
				'view_recon': True,
				'no_images': 12,
				'eps_w': 0.005,
				'eps_b': 0.005,
				'eps_decay': 9,
				'eps_intvl': 10,
				'eps_satr': 'inf',
				'mu': 0.7,
				'l2': 0.95,
				'RMSProp': True,
				'RMSProp_decay': 0.9,
				'minsq_RMSProp': 0.01,
			}
		]
	}

	ae = ConvAE()
	ae.train(train_data, test_data, layers, hyperparams)


if __name__ == '__main__':

	testMnist()
	testTorontoFaces()