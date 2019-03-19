# melody-vae
training a vae that generates a melody

Dataset (type0 folder): 320 single-track (Type 0) piano MIDI files from various classical composers, manually downloaded from http://www.piano-midi.de/midicoll.htm

changing from using vae -> cvae, vae with cnn


## CVAE Notebook
- Example from [pytorch-vae Github](https://github.com/sksq96/pytorch-vae) 
### Install Dependencies
- `python3 -m pip install imageio, torchsummary`


Need libsndfile and audiolab for playback.
- `brew install libsndfile`
- `pip install scikits.audiolab`

## Useful Readings:
- [Pianoroll Dataset Blog](https://salu133445.github.io/lakh-pianoroll-dataset/representation.html)
