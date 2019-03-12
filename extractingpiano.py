from music21 import converter
import glob

for file in glob.glob("type0/*.midi"):
    midi = converter.parse(file)
    print(file)
    parts = midi.parts.stream()
    for p in parts:
        print(p.partName)
