import gym
import numpy as np

from mujoco_py import load_model_from_path, MjSim, MjViewer
import mujoco_py

from gym import ObservationWrapper, spaces

DEFAULT_SIZE = 300

class FrankaEnv(gym.Env):
    def __init__(self):
        self.model_path = 'model/panda.xml'
        self.frame_skip = 1
        self.model = load_model_from_path(self.model_path)
        self.sim = mujoco_py.MjSim(self.model)
         
        self.data = self.sim.data
        self.viewer = None
        self._viewers = {}
        
        self.observation_space = self._construct_observation_space()
        self.action_space = self._construct_action_space()
        self.metadata = {
            "render.modes": ["human", "rgb_array", "depth_array"],
            "video.frames_per_second": int(np.round(1.0 / self.dt)),
        }
        
        self.sim_state = self.sim.get_state()
        
        
    def step(self, action, render=False):
        
        print('step')
        sum_reward = 0
        done = False
        ac_torque_1 = action[:5]*80
        ac_torque_2 = action[5:]*12
        
        ac_torque = np.concatenate((ac_torque_1,ac_torque_2),axis=-1)
        
        self.do_simulation(ac_torque, self.frame_skip)
        obs =self.get_observation()
          
        print(obs)
        
        if render:
            self.render()
        
        
        return obs, sum_reward, done, dict()
    
    
        
    def _construct_action_space(self):
        action_low = -1 * np.ones(7)
        action_high = 1 * np.ones(7)
        return gym.spaces.Box(low=action_low, high=action_high, dtype=np.float32)
    
    
    def _construct_observation_space(self):
        obs_low = -1 * np.ones(17)
        obs_high = 1 * np.ones(17)
        return gym.spaces.Box(obs_low, obs_high)
    
    
    
    def get_observation(self):
        self.sim_state = self.sim.get_state()
        
        joint_position = self.sim_state.qpos
        joint_velocity = self.sim_state.qvel
        return np.concatenate(
        (
            joint_position, #7
            joint_velocity, #7
            self.get_body_com("endEffector"),# 3
        )
        )
        
    def reset(self):
        qpos = np.array([0,0,0,0,0,0,0])
        qvel = np.array([0,0,0,0,0,0,0])
        self.set_state(qpos, qvel)
        
        return self.get_observation()
    
    
    
    def set_state(self, qpos, qvel):
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(
            old_state.time, qpos, qvel, old_state.act, old_state.udd_state
        )
        self.sim.set_state(new_state)
        self.sim.forward()
    
    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames):
        self.sim.data.ctrl[:] = ctrl
        self.sim.step()
        # for _ in range(n_frames):
        #     self.sim.step()
    
    
    def render(
        self,
        mode="human",
        width=DEFAULT_SIZE,
        height=DEFAULT_SIZE,
        camera_id=None,
        camera_name=None,
    ):
        if mode == "rgb_array" or mode == "depth_array":
            if camera_id is not None and camera_name is not None:
                raise ValueError(
                    "Both `camera_id` and `camera_name` cannot be"
                    " specified at the same time."
                )

            no_camera_specified = camera_name is None and camera_id is None
            if no_camera_specified:
                camera_name = "track"

            if camera_id is None and camera_name in self.model._camera_name2id:
                camera_id = self.model.camera_name2id(camera_name)

            self._get_viewer(mode).render(width, height, camera_id=camera_id)

        if mode == "rgb_array":
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == "depth_array":
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            # Extract depth part of the read_pixels() tuple
            data = self._get_viewer(mode).read_pixels(width, height, depth=True)[1]
            # original image is upside-down, so flip it
            return data[::-1, :]
        elif mode == "human":
            self._get_viewer(mode).render()
            
            
            
    def viewer_setup(self):
        """
        This method is called when the viewer is initialized.
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        self.viewer.cam.trackbodyid = 1   #id of the body to track()
        self.viewer.cam.distance = self.model.stat.extent * 1 #how much zoom in
        self.viewer.cam.lookat[0] -= 0.5 #offset x
        self.viewer.cam.lookat[1] -= 0.5 #offset y
        self.viewer.cam.lookat[2] += 0.1 #offset z
        self.viewer.cam.elevation = -40   #cam rotation around the axis in the plane going throug the frame origin

        pass

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == "human":
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == "rgb_array" or mode == "depth_array":
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)

            self.viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    def get_body_com(self, body_name):
        return self.data.get_body_xpos(body_name)

    def state_vector(self):
        return np.concatenate([self.sim.data.qpos.flat, self.sim.data.qvel.flat])