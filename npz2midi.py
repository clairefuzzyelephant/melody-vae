import glob
import pypianoroll


for file in glob.glob("flattened/*.npz"):
    print(file)
    new_obj = pypianoroll.load(file) #loads npz into a pypianoroll multitrack matrix
    new_obj.write(file) #converts to a midi 
