import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import Stream
from pylsl import StreamInlet, resolve_streams
from collections import deque
from scipy.signal import butter, lfilter, sosfilt, sosfiltfilt
from scipy.integrate import trapezoid

import time

NUM_CHANNELS = 4
SAMPLE_RATE = 256  # Muse 2 default
BUFFER_LENGTH = 5  # seconds of data to show
PLOT_INTERVAL = 0.05  # update every 50 ms
BATCH_SIZE = 10  # Pull 10 samples at a time
FILTER_ORDER = 4
frame_count = 0
t0 = None  # For relative time

bands = {
    'alpha': (8,  12),
    'beta':  (12, 30),
    'theta': (4,   8),
    'delta': (0.5, 4),
    'gamma': (30, 100),
}

# 2) Pre-design SOS filters and initial states:
sos_filters   = {}
filter_states = {}
for name, (low, high) in bands.items():
    sos_filters[name]   = name+"this"

print(sos_filters)

for name in bands:
    print(bands[name])


import numpy as np
print(np.__version__)

print(trapezoid([1,2,3], [0,1,2]))
