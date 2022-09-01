import torch
import lietorch
import numpy as np
import torch.nn.functional as F

from lietorch import SE3
from factor_graph import FactorGraph
from droid_slam.data_readers.mvsnet_dataloader import mvs_loader


class DroidFrontend:
    def __init__(self, net, video, mvsnet, args):
        self.video = video
        self.update_op = net.update
        self.graph = FactorGraph(video, net.update, max_factors=48)
        self.mvsnet = mvsnet
        self.args = args

        # local optimization window
        self.t0 = 0
        self.t1 = 0

        # frontent variables
        self.is_initialized = False
        self.count = 0

        self.max_age = 25
        self.iters1 = 4
        self.iters2 = 2

        self.warmup = args.warmup
        self.beta = args.beta
        self.frontend_nms = args.frontend_nms
        self.keyframe_thresh = args.keyframe_thresh
        self.frontend_window = args.frontend_window
        self.frontend_thresh = args.frontend_thresh
        self.frontend_radius = args.frontend_radius

    def __update(self):
        """ add edges, perform update """

        self.count += 1
        self.t1 += 1

        if self.graph.corr is not None:
            self.graph.rm_factors(self.graph.age > self.max_age, store=True)

        self.graph.add_proximity_factors(self.t1-5, max(self.t1-self.frontend_window, 0), 
            rad=self.frontend_radius, nms=self.frontend_nms, thresh=self.frontend_thresh, beta=self.beta, remove=True)

        self.video.disps[self.t1-1] = torch.where(self.video.disps_sens[self.t1-1] > 0, 
           self.video.disps_sens[self.t1-1], self.video.disps[self.t1-1])

        for itr in range(self.iters1):
            self.graph.update(None, None, use_inactive=True)

        # set initial pose for next frame
        poses = SE3(self.video.poses)
        d = self.video.distance([self.t1-3], [self.t1-2], beta=self.beta, bidirectional=True)

        if d.item() < self.keyframe_thresh:
            self.graph.rm_keyframe(self.t1 - 2)
            
            with self.video.get_lock():
                self.video.counter.value -= 1
                self.t1 -= 1

        else:
            for itr in range(self.iters2):
                self.graph.update(None, None, use_inactive=True)

        # refine depths
        i = 3
        if self.mvsnet is not None:
            ref_id, src_ids = self.t1 - i, [self.t1-i-2, self.t1-i-1, self.t1-i+1, self.t1-i+2]
            img_ids = [ref_id] + src_ids
            poses = SE3(self.video.poses[img_ids]).matrix()
            tstamps = self.video.tstamp[img_ids]
            images, proj_matrices, depth_values = mvs_loader(self.args, tstamps, poses, self.video.disps[ref_id])
            with torch.no_grad():
                mvs_outputs = self.mvsnet(images, proj_matrices, depth_values, temperature=0.01)
                final_depth = mvs_outputs["refined_depth"]
                mask = torch.ones_like(final_depth) > 0.0
                for stage, thresh_conf in zip(["stage1", "stage2", "stage3"], [0.1, 0.2, 0.3]):
                    conf_stage = F.interpolate(mvs_outputs[stage]["photometric_confidence"].unsqueeze(1),
                                               (mask.size(1), mask.size(2))).squeeze(1)
                    mask = mask & (conf_stage > thresh_conf)
                final_depth[~mask] = 1e-6
            # disp_up = 1 / (final_depth + 1e-6)
            self.video.disps_up[ref_id] = final_depth.squeeze(0) #disp_up.squeeze(0).clamp(min=0.001)
            self.video.ref_image[0] = images[0, 0]
            # self.video.disps[ref_id] = F.interpolate(disp_up.unsqueeze(0), scale_factor=0.125).squeeze(0).squeeze(0)
            # print(self.t1, tstamps[0])
            # if self.t1 == 10:
            #     import matplotlib.pyplot as plt
            #     plt.imshow(final_depth.squeeze(0).cpu().numpy())
            #     plt.colorbar()
            #     plt.show()
        # set pose for next itration
        self.video.poses[self.t1] = self.video.poses[self.t1-1]
        self.video.disps[self.t1] = self.video.disps[self.t1-1].mean()

        # update visualization
        # self.video.dirty[self.graph.ii.min():self.t1] = True
        self.video.dirty[max(self.graph.ii.min(), self.t1-i-3):(self.t1-i+1)] = True

    def __initialize(self):
        """ initialize the SLAM system """

        self.t0 = 0
        self.t1 = self.video.counter.value

        self.graph.add_neighborhood_factors(self.t0, self.t1, r=3)

        for itr in range(8):
            self.graph.update(1, use_inactive=True)

        self.graph.add_proximity_factors(0, 0, rad=2, nms=2, thresh=self.frontend_thresh, remove=False)

        for itr in range(8):
            self.graph.update(1, use_inactive=True)


        # self.video.normalize()
        self.video.poses[self.t1] = self.video.poses[self.t1-1].clone()
        self.video.disps[self.t1] = self.video.disps[self.t1-4:self.t1].mean()

        # initialization complete
        self.is_initialized = True
        self.last_pose = self.video.poses[self.t1-1].clone()
        self.last_disp = self.video.disps[self.t1-1].clone()
        self.last_time = self.video.tstamp[self.t1-1].clone()

        with self.video.get_lock():
            self.video.ready.value = 1
            self.video.dirty[:self.t1] = True

        self.graph.rm_factors(self.graph.ii < self.warmup-4, store=True)

    def __call__(self):
        """ main update """
        # do initialization
        if not self.is_initialized and self.video.counter.value == self.warmup:
            self.__initialize()
            
        # do update
        elif self.is_initialized and self.t1 < self.video.counter.value:
            self.__update()

        
