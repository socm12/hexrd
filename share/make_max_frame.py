#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 12:14:38 2019

@author: bernier2
"""
import os
import numpy as np
from hexrd import imageseries

data_dir = os.getcwd()
image_dir = os.path.join(os.getcwd(), 'imageseries')
samp_name = "Ruby1_hydra"
scan_number = 0

print("Making requested max frame...")
max_frames_output_name = os.path.join(
    data_dir,
    "%s_%d-maxframes.hdf5" % (samp_name, scan_number)
)

if os.path.exists(max_frames_output_name):
    os.remove(max_frames_output_name)

max_frames = dict.fromkeys(cfg.instrument.hedm.detectors)
for det_key in max_frames.iterkeys():
    fc_file = os.path.join(
            image_dir,
            "%s_%06d-fc_%%s.npz" % (samp_name, scan_number))
    ims = imageseries.open(fc_file % det_key, 'frame-cache')
    max_frames[det_key] = imageseries.stats.max(ims)

ims_out = imageseries.open(
        None, 'array',
        data=np.array([max_frames[i] for i in max_frames]),
        meta={'panels': max_frames.keys()}
    )
imageseries.write(
        ims_out, max_frames_output_name,
        'hdf5', path='/imageseries'
    )
