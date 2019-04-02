# melody-vae
training a vae that generates a melody

Dataset (type0 folder): 320 single-track (Type 0) piano MIDI files from various classical composers, manually downloaded from http://www.piano-midi.de/midicoll.htm

changing from using vae -> cvae, vae with cnn


## CVAE Notebook
- Example from [pytorch-vae Github](https://github.com/sksq96/pytorch-vae) 
### Install Dependencies
- `pip install imageio, torchsummary, scikit-image`
- `pip install --upgrade scikit-image --user`

Need libsndfile and audiolab for playback.
- `brew install libsndfile`
- `pip install scikits.audiolab`
- `brew install https://raw.githubusercontent.com/Homebrew/homebrew-core/34dcd1ff65a56c3191fa57d3dd23e7fffd55fae8/Formula/fluid-synth.rb`

## Fix
- `AttributeError: dlsym(0x7fd2dbaf2010, fluid_synth_get_channel_info): symbol not found` during `import fluidsynth`

[Github pyfluidsynth Issue Sol](https://github.com/nwhitehead/pyfluidsynth/issues/19)
```
brew uninstall fluidsynth
brew install pkg-config
git clone https://github.com/FluidSynth/fluidsynth.git
cd fluidsynth
git checkout 1.1.x
mkdir build
cd build
cmake ..
sudo make install
fluidsynth --version
```

## Useful Readings:
- [Pianoroll Dataset Blog](https://salu133445.github.io/lakh-pianoroll-dataset/representation.html)
