"""
    CIS 6800 Final Project 

    Object to mask the unseen
    occupants
"""

import numpy as np

class OccupnacyMasker:

    def __init__(self):
        pass

    def mask(self, img : np.ndarray) -> np.ndarray:
        # mask behind objects
