import pretty_midi
import glob
import pickle
import pypianoroll

from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

from mido import MidiFile, MidiTrack, Message as MidiMessage

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
import librosa

NPITCH = 128
# output_t_size = 500 = 5 * 100 bc. 100 is default freq pretty_midi
def export_piano_rolls(roll, filename, output_t_size=500):
    print("exporting ", filename)

    pitch, ntime = piano_rolls[0].shape
    if pitch != NPITCH:
        print("Mismatch number of pitches")

    for st in range(0, ntime, output_t_size):
        roll_img = roll[:,st:st+output_t_size]
        if st + output_t_size > ntime:
            break
        plt.imsave('rolls_imgs/{}-st-{}.png'.format(filename, st), roll_img, cmap=cm.gray)
    print("Exported the song {}".format(filename))

def get_filename(filepath):
    base = os.path.basename(filepath)
    return os.path.splitext(base)[0]


def get_piano_rolls():
    pr = []
    done = False
    for file in glob.glob("type0/*.midi"):
        pm = pretty_midi.PrettyMIDI(file) #takes a midi file and converts to pretty_midi
        #pm.remove_invalid_notes()
        pianoroll = pm.get_piano_roll()
        # instr = pm.instruments #splits into list of instruments
        # for instrument in instr:
        #     name = pretty_midi.program_to_instrument_name(instrument.program)
        #     if name == "Acoustic Grand Piano": #only take the piano track
        #         piano_roll = instrument.get_piano_roll() #get the piano roll, which is a np.ndarray
        #         #print(piano_roll.check_pianoroll())
        #         pr.append(piano_roll)
        #         roll = (piano_roll[:,:]>0).astype(int)
        #         filename = get_filename(file)
        #         pianoroll_to_midi(piano_roll, filename)
        #         #export_piano_rolls(roll, filename)
        #         if len(pr) == 1:
        #             done = True
        #             break
        # if done:
        #     break
        filename = get_filename(file)
        piano_roll_to_pretty_midi(pianoroll, filename)
        pr.append(pianoroll)
    return pr #list of piano rolls (np.ndarray matrices)

def piano_roll_to_pretty_midi(piano_roll, filename, fs=50, program=2):
    '''Convert a Piano Roll array into a PrettyMidi object
     with a single instrument.
    Parameters
    ----------
    piano_roll : np.ndarray, shape=(128,frames), dtype=int
        Piano roll of one instrument
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    program : int
        The program number of the instrument.
    Returns
    -------
    midi_object : pretty_midi.PrettyMIDI
        A pretty_midi.PrettyMIDI class instance describing
        the piano roll.
    '''
    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    pm.write("re-{}.midi".format(filename))
    return pm



def prepare_sequences(pr):



    return (net_in, net_out)


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






def train():
    pr = get_piano_rolls()
    net_in, net_out = prepare_sequences(pr)

    #building models
    (x_train, y_train), (x_test, y_test) = (net_in, net_in), (net_out, net_out)#?????????
    #xtrain and ytrain are the same thing, xtest and ytest are the same thing
    #different "cuts" of the sequences, train and test should overlap by a bit

    image_size = x_train.shape[1] #how to change this

    #not sure if we need these four lines/how to alter for audio vs mnist data?
    x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
    x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # network parameters
    input_shape = 5#change
    batch_size = 128
    kernel_size = 3
    filters = 16
    latent_dim = 2
    epochs = 30

    # VAE model = encoder + decoder
    # build encoder model
    inputs = Input(shape=input_shape, name='encoder_input')
    x = inputs
    for i in range(2):
        filters *= 2
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   activation='relu',
                   strides=2,
                   padding='same')(x)

    # shape info needed to build decoder model
    shape = K.int_shape(x)

    # generate latent vector Q(z|X)
    x = Flatten()(x)
    x = Dense(16, activation='relu')(x)
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
    x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
    x = Reshape((shape[1], shape[2], shape[3]))(x)

    for i in range(2):
        x = Conv2DTranspose(filters=filters,
                            kernel_size=kernel_size,
                            activation='relu',
                            strides=2,
                            padding='same')(x)
        filters //= 2

    outputs = Conv2DTranspose(filters=1,
                              kernel_size=kernel_size,
                              activation='sigmoid',
                              padding='same',
                              name='decoder_output')(x)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae')

    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Use mse loss instead of binary cross entropy (default)"
    parser.add_argument("-m", "--mse", help=help_, action='store_true')
    args = parser.parse_args()
    models = (encoder, decoder)
    data = (x_test, y_test)



    # VAE loss = mse_loss or xent_loss + kl_loss
    if args.mse:
        reconstruction_loss = mse(K.flatten(inputs), K.flatten(outputs))
    else:
        reconstruction_loss = binary_crossentropy(K.flatten(inputs),
                                                  K.flatten(outputs))

    reconstruction_loss *= image_size * image_size
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='rmsprop')
    vae.summary()

    if args.weights:
        vae.load_weights(args.weights)
    else:
        # train the autoencoder
        vae.fit(x_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_test, None))
        vae.save_weights('vae_cnn_mnist.h5')


if __name__ == "__main__":
    get_piano_rolls()
