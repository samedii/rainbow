from collections import deque
import random
import atari_py
import torch
import cv2  # Note that importing cv2 before torch may cause segfaults?
import pickle
import numpy as np
import pandas as pd


class Env():
  def __init__(self, args):
    self.device = args.device
    self.actions = {
        'out': 0,
        'in': 1
    }
    self.position = 0
    self.training = True  # Consistent with model training mode
    self.threshold = 0.01 # courtage and spread

    data = get_data()
    self.Xs, self.ys, self.lengths = prepare_data(data)
    self.current_stock = 0
    self.current_day = 0

  def _get_state(self):
    return torch.Tensor(list(self.Xs[self.current_stock][self.current_day]) + [self.position]).to(self.device)

  def reset(self):
    self.current_stock += 1
    self.current_day = 0
    self.position = 0
    return self._get_state()

  def step(self, action):
    reward = self.ys[self.current_stock][self.current_day] if action == 1 else 0

    if self.position == 0 and action == 1:
      reward += np.log(1 - 2*self.threshold)

    done = (self.current_day >= self.lengths[self.current_stock])
    # Return state, reward, done
    return self._get_state(), reward, done

  def train(self):
    self.training = True

  def eval(self):
    self.training = False

  def action_space(self):
    return len(self.actions)

  def render(self):
    pass

  def close(self):
    pass


def get_data():

  database = 'SSE'

  file_object = open('prepared_data_'+database,'rb') 
  data = pickle.load(file_object)
  file_object.close()

  return data

def prepare_data(data):

  X = []
  y = []
  lengths = []
  for stock in data:
      last = stock['Last'].values
      log_rdiff_last = np.log(last[1:]/last[:-1])
      log_rdiff_last = np.expand_dims(log_rdiff_last, axis=1)

      lag1 = log_rdiff_last[4:-1]
      lag5 = log_rdiff_last[:-5]

      X_stock = np.concatenate((lag1, lag5), axis=1) # TODO: do something more useful
      y_stock = log_rdiff_last[5:]

      is_nan = np.logical_or(np.isnan(X_stock).any(axis=1), np.isnan(y_stock).any(axis=1))
      X_stock = X_stock[~is_nan]
      y_stock = y_stock[~is_nan]

      X.append(X_stock)
      y.append(y_stock)
      lengths.append(y_stock.shape[0])

  return X, y, lengths