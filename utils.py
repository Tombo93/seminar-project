import os
import numpy as np
import matplotlib.pyplot as plt

from rlbench.backend.observation import Observation


def plot_img(img: np.ndarray) -> None:
  plt.imshow(img)
  plt.show()


def save_img(fname: str, img: np.ndarray, ) -> None:
  plt.imsave(fname, img, format="png")


def save_observation_imgs(obs: Observation, save_folder: str = "assets") -> None:
  for k, v in obs.__dict__.items():
    if isinstance(v, np.ndarray):
      if (v.dtype == np.uint8):
        v = v.astype(float) / 225
      fname = os.path.join(save_folder, f"{k}.png")
      save_img(fname, v)
