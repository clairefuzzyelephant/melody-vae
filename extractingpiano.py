import glob
import pretty_midi

for file in glob.glob("type0/*.mid"):
    pm = pretty_midi.PrettyMIDI(file) #takes a midi file and converts to pretty_midi
    instr = pm.instruments #splits into list of instruments
    for instrument in instr:
        name = pretty_midi.program_to_instrument_name(instrument.program)
        #print(name)
        if name == "Acoustic Grand Piano": #only take the piano track
            piano_roll = instrument.get_piano_roll() #get the piano roll, which is a np.ndarray
            print(piano_roll)
