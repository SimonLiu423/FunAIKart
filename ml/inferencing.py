import logging
import os
import PAIA
import cv2
import torch


class MLPlay:
    def __init__(self):
        pass

    def decision(self, state: PAIA.State) -> PAIA.Action:
        action = PAIA.create_action_object(acceleration=True, brake=False, steering=0.0)
        if state.event == PAIA.Event.EVENT_NONE:
            action = PAIA.create_action_object(acceleration=True, brake=False, steering=0.0)
        elif state.event != PAIA.Event.EVENT_NONE:
            action = PAIA.create_action_object(command=PAIA.Command.COMMAND_FINISH)
            logging.info('Progress: %.3f' %state.observation.progress )
        return action