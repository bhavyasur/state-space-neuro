from elephant import gpfa
import numpy as np
import scipy.io as sio

import sys
from pathlib import Path

ext =  Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ext))

from old.lorenz.latent_attractor import AttractorDataset



