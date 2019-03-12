from music21 import converter, instrument, note, chord
import glob
import pickle

from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def get_notes():

    notes = []

    """parsing files"""

    for file in glob.glob("type0/*.midi"):
        midi = converter.parse(file)
        print("Parsing %s" % file)

        notes_to_parse = None

        parts = instrument.partitionByInstrument(midi)

        if parts: #file has instrument parts
            notes_to_parse = parts.parts[0].recurse()
        else: #file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note): #if it is a note
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord): # if it is a chord
                notes.append('.'.join(str(n) for n in element.normalOrder))

    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes

def prepare_sequences(notes, n_vocab):

    """mapping note strings to numbers"""

    seq_len = 100

    #get all pitch names
    pitchnames = sorted(set(item for item in notes))

    #create a dictionary to map pitches to integers
    note_to_int = dict((note,number) for number,note in enumerate(pitchnames))
    net_in = []
    net_out = []

    #create inputs and outputs
    for i in range(0, len(notes) - seq_len, 1):
        seq_in = notes[i:i + seq_len]
        seq_out = notes[i + seq_len]
        net_in.append([note_to_int[char] for char in seq_in])
        net_out.append(note_to_int[seq_out])

    n_patterns = len(net_in)

    #reshape input into LSTM-compatible format
    net_in = numpy.reshape(net_in, (n_patterns, seq_len, 1))
    #normalize input
    net_in = net_in/float(n_vocab)

    net_out = np_utils.to_categorical(net_out)

    return (net_in, net_out)



def train():
    notes = get_notes()
    n_vocab = len(set(notes))
    net_in, net_out = prepare_sequences(notes, n_vocab)

    #building models
    (x_train, y_train), (x_test, y_test) = (net_in, net_in), (net_out, net_out)

    image_size = x_train.shape[1]
    original_dim = image_size * image_size
    x_train = np.reshape(x_train, [-1, original_dim])
    x_test = np.reshape(x_test, [-1, original_dim])
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # network parameters
    input_shape = (original_dim, )
    intermediate_dim = 512
    batch_size = 128
    latent_dim = 2
    epochs = 50
    # VAE model = encoder + decoder
    # build encoder model
    inputs = Input(shape=input_shape, name='encoder_input')
    x = Dense(intermediate_dim, activation='relu')(inputs)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    encoder.summary()
    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(intermediate_dim, activation='relu')(latent_inputs)
    outputs = Dense(original_dim, activation='sigmoid')(x)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae_mlp')


    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Use mse loss instead of binary cross entropy (default)"
    parser.add_argument("-m",
                        "--mse",
                        help=help_, action='store_true')
    args = parser.parse_args()
    models = (encoder, decoder)
    data = (x_test, x_test)

    # VAE loss = mse_loss or xent_loss + kl_loss
    if args.mse:
        reconstruction_loss = mse(inputs, outputs)
    else:
        reconstruction_loss = binary_crossentropy(inputs,
                                                  outputs)

    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    vae.summary()

    if args.weights:
        vae.load_weights(args.weights)
    else:
        # train the autoencoder
        vae.fit(x_train, x_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_test, None))
        vae.save_weights('vae_mlp_mnist.h5')

if __name__ == "__main__":
    train()
