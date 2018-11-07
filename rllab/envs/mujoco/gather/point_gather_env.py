from rllab.envs.mujoco.gather.gather_env import GatherEnv
from rllab.envs.mujoco.point_env import PointEnv


class PointGatherEnv(GatherEnv):

    NAME= "PointGatherEnv"
    MODEL_CLASS = PointEnv
    ORI_IND = 2
