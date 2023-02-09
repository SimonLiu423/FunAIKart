import logging
import os
import PAIA
import cv2
import torch
from dqn_model import *

class MLPlay:
    def __init__(self):
        pass

    def decision(self, state: PAIA.State) -> PAIA.Action:

        return action