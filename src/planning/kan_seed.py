"""Frozen KAN kinematic proposal for a single trajectory optimizer.

This is a planning warm start, not execution. Its simulated FIFO is local to
the proposal and MUST NEVER replace the executed-action history in control.
No collision correction, servo, search, shortcutting or retiming is hidden here.
"""
import time
import numpy as np
import torch
from src.data.dagger_buffer import ActionHistory
from src.geometry.rotations import pose9,pose_error6,rotation_6d_to_matrix


class KANSeed:
    def __init__(self, actor, geometry, device='cuda:0'):
        self.actor=actor;self.geometry=geometry;self.device=device
        self.base=geometry.index[geometry.meta['base_path']]
        self.tool=geometry.index[geometry.meta['tool_path']]
        prefix='/World/Base_02/zbot_8s_new_02'
        self.other=[geometry.index[prefix+f'/a{i}'] for i in range(1,9)]+[geometry.index[prefix+'/b8']]

    def tensor(self,x):return torch.as_tensor(np.asarray(x),device=self.device,dtype=torch.float32).unsqueeze(0)

    def geometry_state(self,q):
        transforms,_=self.geometry.forward(q)
        local=np.linalg.inv(transforms[self.base])@transforms
        T=local[self.tool]
        return pose9(self.tensor(T[:3,3]),self.tensor(T[:3,:3])),self.tensor(local[self.other,:3,3])

    def observation(self,start,goal,q,xstart,xgoal,history):
        current,other=self.geometry_state(q)
        actions,mask=history.padded()
        return dict(q_start=self.tensor(start),q_current=self.tensor(q),q_goal=self.tensor(goal),
                    delta_q_goal=self.tensor((goal-q+np.pi)%(2*np.pi)-np.pi),x_start=xstart,x_current=current,x_goal=xgoal,
                    delta_x_goal=pose_error6(current[:,:3],rotation_6d_to_matrix(current[:,3:]),xgoal[:,:3],rotation_6d_to_matrix(xgoal[:,3:])),
                    other_arm_geometry=other,history_action=self.tensor(actions),history_mask=torch.tensor(mask,device=self.device)[None])

    @torch.inference_mode()
    def generate(self,start,goal,max_steps=180,velocity=1.5,dt=1/30,history=None):
        begun=time.perf_counter()
        # All angle-valued inputs follow the training [-pi,pi) convention;
        # output path is unwrapped continuously for spline optimization.
        start=(np.asarray(start)+np.pi)%(2*np.pi)-np.pi
        goal=(np.asarray(goal)+np.pi)%(2*np.pi)-np.pi
        q=start.copy();path=[q.copy()]
        xstart,_=self.geometry_state(start);xgoal,_=self.geometry_state(goal)
        virtual=ActionHistory(self.actor.config.history_length)
        if history is not None:
            actions,mask=history.padded()
            for a in actions[mask]:virtual.append(a)
        near=False
        for _ in range(max_steps):
            wrapped=(q+np.pi)%(2*np.pi)-np.pi
            action=self.actor(self.observation(start,goal,wrapped,xstart,xgoal,virtual))['action'][0].cpu().numpy()
            command=np.clip(action,-velocity*dt,velocity*dt)
            q=q+command;path.append(q.copy());virtual.append(command.astype(np.float32))
            near=bool(np.max(abs((goal-q+np.pi)%(2*np.pi)-np.pi))<.1)
            if near:break
        # Endpoint anchoring is a seed condition, not an executed final jump.
        endpoint=q+(goal-q+np.pi)%(2*np.pi)-np.pi
        path.append(endpoint)
        return np.array(path),dict(proposal_steps=len(path)-2,near_goal=near,wall_s=time.perf_counter()-begun,
                                   history='local virtual kinematic rollout, never execution history',endpoint_anchored=True)
