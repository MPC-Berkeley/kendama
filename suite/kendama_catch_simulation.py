# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Based on dm_control Ball-in-Cup Domain."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers
import numpy as np

_DEFAULT_TIME_LIMIT = 20  # (seconds)
_CONTROL_TIMESTEP = .01   # (seconds)
np.random.seed(42)

SUITE = containers.TaggedTasks()


def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""
  return common.read_model('kendama_catch_simulation.xml'), common.ASSETS


@SUITE.add('benchmarking', 'easy')
def catch(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Kendama-Ball-in-Cup task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = KendamaBallInCup(random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)


class Physics(mujoco.Physics):
  """Physics with additional features for the Ball-in-Cup domain."""

  def ball_to_cup(self):
    """Returns the position vector from the ball to the target."""
    cup = self.named.data.xpos['cup', ['x', 'z']]
    ball = self.named.data.xpos['ball', ['x', 'z']]
    return cup - ball

  def vel_ball_to_cup(self):
    """Returns the velocity vector from the ball to the cup."""
    cup = np.array([self.named.data.qvel['cup_x'],self.named.data.qvel['cup_z']])
    ball = np.array([self.named.data.qvel['ball_x'],self.named.data.qvel['ball_z']]) 
    return cup - ball

  def ball_to_target(self):
    """Returns the vector from the ball to the target."""
    target = self.named.data.site_xpos['target', ['x', 'z']]
    ball = self.named.data.xpos['ball', ['x', 'z']]
    return target - ball

  def in_target(self):
    """Returns 1 if the ball is in the target, 0 otherwise."""
    ball_to_target = abs(self.ball_to_target())
    target_size = self.named.model.site_size['target', [0, 2]]
    ball_size = self.named.model.geom_size['ball', 0]
    return float(all(ball_to_target < target_size - ball_size))


class KendamaBallInCup(base.Task):
  """The Ball-in-Cup task. Put the ball in the cup."""

  def initialize_episode(self, physics, random_seed = 42):
    """Sets the state of the environment at the start of each episode.

    Args:
      physics: An instance of `Physics`.

    """
    # Find a collision-free random initial position of the ball.
    penetrating = True
    np.random.seed(random_seed)
    while penetrating:
      # Assign a random ball position.
      physics.named.data.qpos['cup_x'] = 0.0 
      physics.named.data.qpos['cup_z'] = 0.0 
      physics.named.data.qpos['ball_x'] = np.random.uniform(0.3158, 0.3492)
      physics.named.data.qpos['ball_z'] = np.random.uniform(0.2095, 0.2457)

      # Assign a random ball velocity
      physics.named.data.qvel['ball_x'] = np.random.uniform(-1.4, -0.5)
      physics.named.data.qvel['ball_z'] = np.random.uniform(0.5, 1.4)
      physics.after_reset()
      penetrating = physics.data.ncon > 0

  def get_observation(self, physics):
    """Returns an observation of the state."""
    obs = collections.OrderedDict()
    obs['position'] = physics.position()
    obs['velocity'] = physics.velocity() 
    return obs

  def get_reward(self, physics):
    """Returns a sparse reward."""
    return physics.in_target()

  def get_termination(self, physics):
    """Returns terminal condition."""
    if physics.ball_to_target()[1]>0.3:
      return True
    else:
      return None


