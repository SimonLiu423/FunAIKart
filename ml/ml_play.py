import logging
import os  # you can use functions in logging: debug, info, warning, error, critical, log
from config import ENV
import PAIA
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import collections
from dqn_model import *
from datetime import datetime

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
            np.array(dones, dtype=np.uint8), np.array(next_states)


class DeepQNetwork():
    def __init__(
            self,
            n_actions,
            input_shape,
            qnet,
            device,
            learning_rate=2e-4,
            reward_decay=0.99,
            replace_target_iter=1000,
            memory_size=10000,
            batch_size=32,
    ):
        # initialize parameters
        self.n_actions = n_actions
        self.input_shape = input_shape
        self.lr = learning_rate
        self.gamma = reward_decay
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.device = device
        self.learn_step_counter = 0
        self.exp_buffer = ExperienceBuffer(memory_size)

        # Network
        self.net = qnet(self.input_shape, self.n_actions).to(self.device)
        self.tgt_net = qnet(self.input_shape, self.n_actions).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

    def calc_loss(self):
        states, actions, rewards, dones, next_states = self.exp_buffer.sample(self.batch_size)

        states_v = torch.tensor(np.array(states, copy=False)).to(self.device)
        next_states_v = torch.tensor(np.array(next_states, copy=False)).to(self.device)
        actions_v = torch.tensor(actions).to(self.device)
        rewards_v = torch.tensor(rewards).to(self.device)
        done_mask = torch.BoolTensor(dones).to(self.device)

        state_action_values = self.net(states_v.float()).gather(1, actions_v.unsqueeze(-1).type(torch.int64)).squeeze(
            -1)
        with torch.no_grad():
            next_state_values = self.tgt_net(next_states_v.float()).max(1)[0]
            next_state_values[done_mask] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * self.gamma + rewards_v
        return nn.MSELoss()(state_action_values, expected_state_action_values)

    def choose_action(self, state, epsilon=0.0):
        if np.random.random() < epsilon:
            act_v = np.random.randint(self.n_actions)
        else:
            state_v = torch.tensor([state]).to(self.device)
            q_vals_v = self.net(state_v.float())
            _, act_v = torch.max(q_vals_v, dim=1)
        action = int(act_v)
        return action

    def learn(self):
        # check to replace target parameters
        if len(self.exp_buffer) >= self.batch_size:
            if self.learn_step_counter % self.replace_target_iter == 0:
                self.tgt_net.load_state_dict(self.net.state_dict())
            self.optimizer.zero_grad()
            loss_t = self.calc_loss()
            loss_t.backward()
            self.optimizer.step()
        self.learn_step_counter += 1

    def store_transition(self, s, a, r, d, s_):
        exp = Experience(s, a, r, d, s_)
        self.exp_buffer.append(exp)

    def save_model(self):
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_model.dat")
        torch.save(self.net.state_dict(), model_path)


def epsilon_compute(frame_id, epsilon_max=1.0, epsilon_min=0.02, epsilon_decay=10000):
    return max(epsilon_min, epsilon_max - frame_id / epsilon_decay)


class MLPlay:
    def __init__(self):
        # self.demo = Demo.create_demo() # create a replay buffer
        self.episode_number = 1
        self.epsilon = 1.0
        self.progress = 0
        self.total_rewards = []
        self.best_mean = -100000
        # TODO create any variables you need **********************************************************************#
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
        self.prev_progress = 0
        self.episode_reward = 0
        self.step = 0
        self.frame_idx = 0
        self.cnt = 0
        self.k = 4
        self.action_id = 1
        self.curr_state = None
        self.state_front_stack = []
        self.state_back_stack = []
        self.effects = {'n': 0, 't': 0, 'b': 0, 'w': 1.0, 'g': 1.0}

        self.agent = DeepQNetwork(self.n_actions, (self.k * 2, *self.img_size), QNet, device='cpu')
        # self.agent.net.load_state_dict(torch.load('./saves/best_0.pt'))
        # **********************************************************************************************************#

    def effect_diff(self, state):
        new_effects = {
            'n': state.observation.effects.nitro.number,
            't': state.observation.effects.turtle.number,
            'b': state.observation.effects.banana.number,
            'w': state.observation.refills.wheel.value,
            'g': state.observation.refills.gas.value,
        }
        # logging.info('wheel: {}, gas: {}'.format(state.observation.refills.wheel.value, state.observation.refills.gas.value))
        diff = {
            'n': new_effects['n'] - self.effects['n'],
            't': new_effects['t'] - self.effects['t'],
            'b': new_effects['b'] - self.effects['b'],
            'w': new_effects['w'] - self.effects['w'],
            'g': new_effects['g'] - self.effects['g'],
        }
        self.effects = new_effects
        return diff

    def preprocess(self, state):
        front_img_array = PAIA.image_to_array(state.observation.images.front.data)  # img_array.shape = (112, 252, 3)
        back_img_array = PAIA.image_to_array(state.observation.images.back.data)  # img_array.shape = (112, 252, 3)
        # TODO Image Preprocessing ****************#
        # Hint:
        #      GrayScale: img  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #      Resize:    img  = cv2.resize(img, (width, height))
        resized_front_img = cv2.resize(front_img_array, (self.img_size[1], self.img_size[0]))
        gray_front_img = cv2.cvtColor(resized_front_img, cv2.COLOR_BGR2GRAY)

        resized_back_img = cv2.resize(back_img_array,  (self.img_size[1], self.img_size[0]))
        gray_back_img = cv2.cvtColor(resized_back_img, cv2.COLOR_BGR2GRAY)
        return gray_front_img, gray_back_img

    def get_reward(self, state):
        progress = state.observation.progress
        ret = 0
        if progress > self.prev_progress:
            ret += 1.0
        elif progress == self.prev_progress:
            ret += -1.0
        else:
            ret += -2.0
        self.prev_progress = progress

        if state.observation.rays.F.distance <= 0.05 or \
                state.observation.rays.B.distance <= 0.05 or \
                state.observation.rays.R.distance <= 0.05 or \
                state.observation.rays.L.distance <= 0.05 or \
                state.observation.rays.FR.distance <= 0.05 or \
                state.observation.rays.RF.distance <= 0.05 or \
                state.observation.rays.FL.distance <= 0.05 or \
                state.observation.rays.LF.distance <= 0.05 or \
                state.observation.rays.BR.distance <= 0.05 or \
                state.observation.rays.BL.distance <= 0.05:
            ret += -3

        # TODO get item reward
        effect_diff = self.effect_diff(state)

        if effect_diff['t'] > 0 or effect_diff['b'] > 0:
            ret += -5 + 3

        if effect_diff['w'] > 0:
            ret += effect_diff['w'] + 3

        if effect_diff['g'] > 0:
            ret += effect_diff['g'] + 3

        if effect_diff['n'] > 0:
            ret += 3

        if state.event in [PAIA.Event.EVENT_TIMEOUT, PAIA.Event.EVENT_UNDRIVABLE]:
            ret += -5
        elif state.event == PAIA.Event.EVENT_WIN:
            ret += 10

        # logging.info('reward: {}'.format(ret))
        return ret

    def decision(self, state: PAIA.State) -> PAIA.Action:
        '''
        Implement yor main algorithm here.
        Given a state input and make a decision to output an action
        '''
        # Implement Your Algorithm
        # Note: You can use PAIA.image_to_array() to convert
        #       state.observation.images.front.data and 
        #       state.observation.images.back.data to numpy array (range from 0 to 1)
        #       For example: img_array = PAIA.image_to_array(state.observation.images.front.data)

        # TODO Reinforcement Learning Algorithm *******************************************************************#
        # 1. Preprocess
        # 2. Design state, action, reward, next_state by yourself
        # 3. Store the datas into ReplayedBuffer
        # 4. Update Epsilon value
        # 5. Train Q-Network
        MAX_EPISODES = int(ENV.get('MAX_EPISODES') or -1)

        if state.event == PAIA.Event.EVENT_RESTART:
            logging.info('EVENT_RESTART')

            self.episode_number += 1
            self.episode_reward = 0

            self.prev_progress = 0
            self.step = 0
            self.cnt = 0
            self.action_id = 1
            self.state_front_stack = []
            self.state_back_stack = []
            self.effects = {'n': 0, 't': 0, 'b': 0, 'w': 1.0, 'g': 1.0}

        self.step += 1

        if not state.observation.images.front.data:
            return PAIA.create_action_object(*self.action_space[1])

        state_front_img, state_back_img = self.preprocess(state)
        state_front_img = state_front_img[np.newaxis, ...]      # (width, height) => (1, width, height)
        state_back_img = state_back_img[np.newaxis, ...]
        self.state_front_stack.append(state_front_img)
        self.state_back_stack.append(state_back_img)

        self.agent.learn()
        # logging.info('Step: {}'.format(self.step))

        if self.step >= 75 and self.step % self.k == 0:
            delta_progress = (state.observation.progress - self.prev_progress) * 1000
            if delta_progress < 0.1:
                self.cnt += 1
            else:
                self.cnt = 0

            r = self.get_reward(state)
            done = (state.event in [PAIA.Event.EVENT_RESTART, PAIA.Event.EVENT_FINISH])

            self.episode_reward += r

            stacked_state = self.state_front_stack + self.state_back_stack
            # [(1, width, height), (1, width, height), ...] => (k, width, height)
            stacked_img = np.concatenate(stacked_state, axis=0)
            self.agent.store_transition(self.curr_state, self.action_id, r, done, stacked_img)

        if self.step % self.k == 0:
            stacked_state = self.state_front_stack + self.state_back_stack
            stacked_img = np.concatenate(stacked_state, axis=0)     # (k*2, width, height)
            self.curr_state = stacked_img
            self.state_back_stack = []
            self.state_front_stack = []

            self.frame_idx += 1     # nth state
            self.epsilon = epsilon_compute(frame_id=self.frame_idx)
            # self.epsilon = 0
            self.action_id = self.agent.choose_action(self.curr_state, self.epsilon)

        # *********************************************************************************************************#

        action = PAIA.create_action_object(True, False, 0.0)
        # if (MAX_EPISODES < 0 or self.episode_number < MAX_EPISODES) and self.cnt > 10:
        #     self.total_rewards.append(self.episode_reward)
        #     logging.info('Epispde: ' + str(self.episode_number) + ', Epsilon: ' + str(
        #         self.epsilon) + ', Progress: %.3f' % state.observation.progress + ', Reward: ' + str(self.episode_reward))
        #     mean_reward = np.mean(self.total_rewards[-30:])
        #     if self.best_mean < mean_reward:
        #         print("Best mean reward updated %.3f -> %.3f, model saved" % (self.best_mean, mean_reward))
        #         self.best_mean = mean_reward
        #         self.agent.save_model()
        #
        #     action = PAIA.create_action_object(command=PAIA.Command.COMMAND_RESTART)
        #     #
        #     # self.episode_number += 1
        #     # self.episode_reward = 0
        #     #
        #     # self.prev_progress = 0
        #     # self.step = 0
        #     # self.cnt = 0
        #     # self.action_id = 1
        #     # self.state_front_stack = []
        #     # self.state_back_stack = []
        if state.event == PAIA.Event.EVENT_NONE:
            # Continue the game

            # TODO You can decide your own action (change the following action to yours) *****************************#
            action = PAIA.create_action_object(*self.action_space[self.action_id])
            # *********************************************************************************************************#

            # You can save the step to the replay buffer (self.demo)
            # self.demo.create_step(state=state, action=action)
        elif state.event == PAIA.Event.EVENT_RESTART:
            # You can do something when the game restarts by someone
            # You can decide your own action (change the following action to yours)

            # TODO Do anything you want when the game reset *********************************************************#
            pass
            #
            # self.total_rewards.append(self.episode_reward)
            # logging.info('Epispde: ' + str(self.episode_number) + ', Epsilon: ' + str(
            #     self.epsilon) + ', Progress: %.3f' % state.observation.progress + ', Reward: ' + str(
            #     self.episode_reward))
            # mean_reward = np.mean(self.total_rewards[-30:])
            # if self.best_mean < mean_reward:
            #     print("Best mean reward updated %.3f -> %.3f, model saved" % (self.best_mean, mean_reward))
            #     self.best_mean = mean_reward
            #     self.agent.save_model()
            # *********************************************************************************************************#

            # You can start a new episode and save the step to the replay buffer (self.demo)
            # self.demo.create_episode()
            # self.demo.create_step(state=state, action=action)
        elif state.event != PAIA.Event.EVENT_NONE:
            # You can do something when the game (episode) ends
            want_to_restart = True  # Uncomment if you want to restart
            # want_to_restart = False # Uncomment if you want to finish
            if (MAX_EPISODES < 0 or self.episode_number < MAX_EPISODES) and want_to_restart:
                # Do something when restart
                action = PAIA.create_action_object(command=PAIA.Command.COMMAND_RESTART)
                # You can save the step to the replay buffer (self.demo)
                # self.demo.create_step(state=state, action=action)
            else:
                # Do something when finish
                action = PAIA.create_action_object(command=PAIA.Command.COMMAND_FINISH)
                # You can save the step to the replay buffer (self.demo)
                # self.demo.create_step(state=state, action=action)
                # You can export your replay buffer
                # self.demo.export('kart.paia')
            self.total_rewards.append(self.episode_reward)
            logging.info('Epispde: ' + str(self.episode_number) + ', Epsilon: ' + str(
                self.epsilon) + ', Progress: %.3f' % state.observation.progress + ', Reward: ' + str(self.episode_reward))
            mean_reward = np.mean(self.total_rewards[-30:])
            if self.best_mean < mean_reward:
                print("Best mean reward updated %.3f -> %.3f, model saved" % (self.best_mean, mean_reward))
                self.best_mean = mean_reward
                self.agent.save_model()

            # self.episode_number += 1
            # self.episode_reward = 0
            #
            # self.prev_progress = 0
            # self.step = 0
            # self.cnt = 0
            # self.action_id = 1
            # self.state_front_stack = []
            # self.state_back_stack = []


        ##logging.debug(PAIA.action_info(action))
        return action

    def autosave(self):
        # now = datetime.now()
        # fname = 'Episode_{}_{}{}_{}{}.pt'.format(str(self.episode_number).zfill(3), str(now.month).zfill(2), str(now.day).zfill(2), str(now.hour).zfill(2), str(now.minute).zfill(2))
        # model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'autosave', fname)
        # torch.save(self.agent.net.state_dict(), model_path)
        '''
        self.autosave() will be called when the game restarts,
        You can save some important information in case that accidents happen.
        '''
        pass
