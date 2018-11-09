import numpy as np
import pygame
from rllab.envs.box2d.parser import find_body

from rllab.core.serializable import Serializable
from rllab.envs.box2d.box2d_env import Box2DEnv
from rllab.misc import autoargs
from rllab.misc.overrides import overrides

import rllab.envs.mujoco.seeding as seeding
from rllab.envs.base import Env, Step

class MountainCarEnv(Box2DEnv, Serializable):
    NAME= "MountainCarEnv"

    @autoargs.inherit(Box2DEnv.__init__)
    @autoargs.arg("height_bonus_coeff", type=float,
                  help="Height bonus added to each step's reward")
    @autoargs.arg("goal_cart_pos", type=float,
                  help="Goal horizontal position")
    def __init__(self,
                 height_bonus=1.,
                 goal_cart_pos=0.6,
                 *args, **kwargs):
        super(MountainCarEnv, self).__init__(
            self.model_path("mountain_car.xml.mako"),
            *args, **kwargs
        )
        self.max_cart_pos = 2
        self.goal_cart_pos = goal_cart_pos
        self.height_bonus = height_bonus
        self.cart = find_body(self.world, "cart")
        self.steps = 0
        Serializable.quick_init(self, locals())

    @overrides
    def compute_reward(self, action):
        yield
        yield self.is_current_done()

    @overrides
    def is_current_done(self):
        return self.cart.position[0] >= self.goal_cart_pos \
            or abs(self.cart.position[0]) >= self.max_cart_pos


    @overrides
    def step(self, action):
        """
        Note: override this method with great care, as it post-processes the
        observations, etc.
        """
        reward_computer = self.compute_reward(action)
        # forward the state
        action = self._inject_action_noise(action)
        for _ in range(self.frame_skip):
            self.forward_dynamics(action)
        # notifies that we have stepped the world
        next(reward_computer)
        # actually get the reward
        reward = next(reward_computer)
        self._invalidate_state_caches()
        self.steps += 1
        done = self.is_current_done() or self.steps >=400
        next_obs = self.get_current_obs()
        

        return Step(observation=next_obs, reward=reward, done=done)


    @overrides
    def reset(self):
        self.steps = 0
        self._set_state(self.initial_state)
        self._invalidate_state_caches()
        bounds = np.array([
            [-1],
            [1],
        ])
        low, high = bounds
        xvel = np.random.uniform(low, high)
        self.cart.linearVelocity = (float(xvel), self.cart.linearVelocity[1])
        return self.get_current_obs()

    @overrides
    def action_from_keys(self, keys):
        if keys[pygame.K_LEFT]:
            return np.asarray([-1])
        elif keys[pygame.K_RIGHT]:
            return np.asarray([+1])
        else:
            return np.asarray([0])



    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)