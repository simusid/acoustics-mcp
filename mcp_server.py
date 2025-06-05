from mcp.server import FastMCP
from typing import List, Dict, Any
import sys
import json
from glob import glob
import librosa
import numpy as np
import scipy.io.wavfile
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

mcp = FastMCP("app")

@mcp.tool()
def list_files(root_path):
	"""Passed a path in the filesystem, it will recursively 
	search for all wav or mp3 files.  
	Parameters:
	root_path : starting folder path

	Returns:
	List of files
	"""
	return glob(f"{root_path}/*mp3") + glob(f"{root_path}/*wav")

@mcp.tool()
def resample(fname, sr):
	"""This function changes the sampling rate of a time series.
	Parameters:
	fname - the name of the file
	sr - the sampling rate of the file
	
	Returns:
	name of the new resampled file"""
	x,sr = librosa.load(fname, sr=sr)
	new_fname=Path(fname).stem + "_resample.wav"
	scipy.io.wavfile.write(new_fname, sr, x)
	return new_fname

@mcp.tool()
def showSpectrogram(fname):
	"""This function is passed a file name of an acoustic time series.
	It loads the file then generates and displays the spectrogram
	Parameters:
	fname - the name of the file with the time series"""
	x,sr = librosa.load(fname, sr=None)
	y = librosa.stft(y=x)
	y = np.log(np.abs(y))
	y = y-y.min()
	y = y/y.max()*255.
	img = Image.fromarray(y)
	img.show()
	return True

@mcp.tool()
def split_file(fname):
	"""This function is passed a file name of a time series file.  It splits it into 
	two pieces and saves both.
	
	Paramters:
	fname - name of the input file
	
	Returns:  
	names of the two output files, first half and second half."""
	x, sr = librosa.load(fname, sr=None)
	idx = x.shape[0]//2
	scipy.io.wavfile.write("/tmp/first_half.wav", sr, x[:idx])
	scipy.io.wavfile.write("/tmp/second_half.wav", sr, x[idx:])
	return "/tmp/first_half.wav", "/tmp/second_half.wav"

# run the server
if __name__ == "__main__":
	print('starting server\n', file=sys.stderr)
	mcp.run()
