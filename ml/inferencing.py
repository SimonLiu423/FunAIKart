import logging
import os
import PAIA
import cv2
import torch
from dqn_model import *

class MLPlay:
    def __init__(self):
        self.action_space = [
            (False, False, 0.0),
            (True, False, 0.0),
            (True, False, -1.0),
            (True, False, 1.0),
            (False, True, 0.0),
            (False, True, -1.0),
            (False, True, 1.0),
        ]
        self.n_actions = len(self.action_space)
        self.img_size = (32, 64)
        self.state_info_size = 28
        self.step = 0
        self.action_id = 1
        self.k = 4
        self.state_front_stack = []
        self.state_back_stack = []
        self.device = 'cpu'

        self.qnet = QNet((self.k * 2, *self.img_size), self.n_actions, self.state_info_size)
        fname = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_model.dat")
        self.qnet.load_state_dict(torch.load(fname))
        self.qnet.eval()

    def decision(self, state: PAIA.State) -> PAIA.Action:

        if not state.observation.images.front.data:
            return PAIA.create_action_object(*self.action_space[1])

        self.step += 1
        state_front_img, state_back_img = self.preprocess(state)
        state_front_img = state_front_img[np.newaxis, ...]  # (width, height) => (1, width, height)
        state_back_img = state_back_img[np.newaxis, ...]
        self.state_front_stack.append(state_front_img)
        self.state_back_stack.append(state_back_img)

        state_info = self.get_state_info(state)

        if self.step % self.k == 0:
            stacked_state = self.state_front_stack + self.state_back_stack
            stacked_img = np.concatenate(stacked_state, axis=0)

            self.state_front_stack = []
            self.state_back_stack = []

            self.action_id = self.choose_action(stacked_img, state_info, 0)

        action = PAIA.create_action_object(True, False, 0.0)

        if state.event == PAIA.Event.EVENT_NONE:
            action = PAIA.create_action_object(*self.action_space[self.action_id])
        elif state.event != PAIA.Event.EVENT_NONE:
            action = PAIA.create_action_object(command=PAIA.Command.COMMAND_FINISH)
            logging.info('Progress: %.3f' % state.observation.progress)

        return action

    def choose_action(self, state, state_info, epsilon=0.0):
        if np.random.random() < epsilon:
            act_v = np.random.randint(self.n_actions)
        else:
            state_v = torch.tensor([state]).to(self.device)
            state_info_v = torch.tensor([state_info], dtype=torch.float32).to(self.device)
            q_vals_v = self.qnet(state_v.float(), state_info_v)
            _, act_v = torch.max(q_vals_v, dim=1)
        action = int(act_v)
        return action

    def preprocess(self, state):
        front_img_array = PAIA.image_to_array(state.observation.images.front.data)  # img_array.shape = (112, 252, 3)
        back_img_array = PAIA.image_to_array(state.observation.images.back.data)  # img_array.shape = (112, 252, 3)
        # TODO Image Preprocessing ****************#
        # Hint:
        #      GrayScale: img  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #      Resize:    img  = cv2.resize(img, (width, height))
        resized_front_img = cv2.resize(front_img_array, (self.img_size[1], self.img_size[0]))
        gray_front_img = cv2.cvtColor(resized_front_img, cv2.COLOR_BGR2GRAY)

        resized_back_img = cv2.resize(back_img_array, (self.img_size[1], self.img_size[0]))
        gray_back_img = cv2.cvtColor(resized_back_img, cv2.COLOR_BGR2GRAY)
        return gray_front_img, gray_back_img

    def get_state_info(self, state):
        return np.array([
            state.observation.rays.F.hit,
            state.observation.rays.F.distance,
            state.observation.rays.B.hit,
            state.observation.rays.B.distance,
            state.observation.rays.R.hit,
            state.observation.rays.R.distance,
            state.observation.rays.L.hit,
            state.observation.rays.L.distance,
            state.observation.rays.FR.hit,
            state.observation.rays.FR.distance,
            state.observation.rays.RF.hit,
            state.observation.rays.RF.distance,
            state.observation.rays.FL.hit,
            state.observation.rays.FL.distance,
            state.observation.rays.LF.hit,
            state.observation.rays.LF.distance,
            state.observation.rays.BR.hit,
            state.observation.rays.BR.distance,
            state.observation.rays.BL.hit,
            state.observation.rays.BL.distance,
            state.observation.progress,
            state.observation.usedtime,
            state.observation.velocity,
            state.observation.refills.wheel.value,
            state.observation.refills.gas.value,
            state.observation.effects.nitro.number,
            state.observation.effects.turtle.number,
            state.observation.effects.banana.number,
        ])
