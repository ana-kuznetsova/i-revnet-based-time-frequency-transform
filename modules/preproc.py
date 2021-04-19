import os
import numpy as np
import librosa
from pydub import AudioSegment
import pandas as pd
from tqdm import tqdm


def segment(aud, ms=40, overlap=10):
    segments = []
    
    prev_count = 0
    count=0
    while count < len(aud):
        count+=ms
        aud_seg = aud[prev_count:count]
        aud_seg = np.asarray(aud_seg.get_array_of_samples(), dtype = np.int32)
        max_amplitude = max(aud_seg)
        aud_seg = aud_seg/max_amplitude
        segments.append(aud_seg)
        prev_count=count-overlap
    return segments

def extract_feats(fnames, scores):
    for aud in tqdm(fnames):
        aud = AudioSegment.from_wav(aud)
