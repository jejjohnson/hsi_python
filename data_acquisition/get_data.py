import urllib.request
import scipy.io as sio
import pickle
import os

def load_sparc():

    filename = 'SPARC.mat'

    url = "https://github.com/IPL-UV/simpleR/raw/master/DATA/SPARC.mat"

    with urllib.request.urlopen(url) as f:
        raw_matfile = f.read()
    
    with open(filename, 'wb') as save:
        save.write(raw_matfile)

    data = sio.loadmat('SPARC.mat')

    x = data['X']
    y = data['Y']
    numcontrol = data['NumControl']
    wavelength = data['WaveLength']

    return x, y, numcontrol, wavelength