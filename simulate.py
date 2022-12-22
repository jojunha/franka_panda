#!/usr/bin/env python3
import os
import numpy as np

# import mujoco_py
from mujoco_py import load_model_from_path, MjSim, MjViewer
import controller


class franka_panda:
    def __init__(self) -> None:
        self.model_path = 'model/panda.xml'
        self.model = load_model_from_path(self.model_path)
        self.sim = MjSim(self.model)
        self.viewer = MjViewer(self.sim)
        
        self.controller = controller.CController()
        self._torque = np.zeros(7, dtype=np.float64)
        
    def run(self) -> None:
        while True:
            self.sim_state = self.sim.get_state()

            self.controller.read(self.sim_state.time, self.sim_state.qpos, self.sim_state.qvel)
            self.controller.control_mujoco()
            self._torque = self.controller.write()
    
            for i in range(7):
                self.sim.data.ctrl[i] = self._torque[i]
    
    
            self.sim.forward()
            self.sim.step()

            self.viewer.render()
    
            if os.getenv('TESTING') is not None:
                break

def main():
    panda = franka_panda()
    panda.run()
    
if __name__ == "__main__":
    main()
        
        