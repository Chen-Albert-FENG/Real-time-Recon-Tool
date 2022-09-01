import torch
import torch.nn.functional as F
import cv2
import lietorch
import droid_backends
import time
import argparse
import numpy as np
import open3d as o3d

from lietorch import SE3
import geom.projective_ops as pops
import depth_fusion

CAM_POINTS = np.array([
        [ 0,   0,   0],
        [-1,  -1, 1.5],
        [ 1,  -1, 1.5],
        [ 1,   1, 1.5],
        [-1,   1, 1.5],
        [-0.5, 1, 1.5],
        [ 0.5, 1, 1.5],
        [ 0, 1.2, 1.5]]) / 3

CAM_LINES = np.array([
    [1,2], [2,3], [3,4], [4,1], [1,0], [0,2], [3,0], [0,4], [5,7], [7,6]])


def white_balance(img):
    # from https://stackoverflow.com/questions/46390779/automatic-white-balancing-with-grayworld-assumption
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result


def create_camera_actor(g, scale=0.05):
    """ build open3d camera polydata """
    camera_actor = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(scale * CAM_POINTS),
        lines=o3d.utility.Vector2iVector(CAM_LINES))

    color = (g * 1.0, 0.5 * (1-g), 0.9 * (1-g))
    camera_actor.paint_uniform_color(color)
    return camera_actor


def create_point_actor(points, colors):
    """ open3d point cloud from numpy array """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    return point_cloud


def droid_visualization(video, device="cuda:0"):
    """ DROID visualization frontend """

    torch.cuda.set_device(device)
    droid_visualization.video = video
    droid_visualization.cameras = {}
    droid_visualization.points = {}
    droid_visualization.warmup = 8
    droid_visualization.scale = 1.0
    droid_visualization.ix = 0

    droid_visualization.filter_thresh = 0.01
    droid_visualization.thresh_view = 3
    droid_visualization.thresh_disp = 1.0

    intr_params = video.intrinsics[0] * 16
    fx, fy, cx, cy = intr_params.cpu().numpy()
    intr_mat = torch.tensor([[fx, 0, cx, 0], [0, fy, cy, 0],
                            [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32, device=device)

    def increase_filter(vis):
        droid_visualization.filter_thresh *= 2
        with droid_visualization.video.get_lock():
            droid_visualization.video.dirty[:droid_visualization.video.counter.value] = True

    def decrease_filter(vis):
        droid_visualization.filter_thresh *= 0.5
        with droid_visualization.video.get_lock():
            droid_visualization.video.dirty[:droid_visualization.video.counter.value] = True

    def animation_callback(vis):
        cam = vis.get_view_control().convert_to_pinhole_camera_parameters()

        with torch.no_grad():

            with video.get_lock():
                t = video.counter.value 
                dirty_index, = torch.where(video.dirty.clone())
                dirty_index = dirty_index
            if (len(dirty_index) == 0) or (intr_mat is None):
                return
            video.dirty[dirty_index] = False
            # convert poses to 4x4 matrix
            poses = torch.index_select(video.poses, 0, dirty_index)
            poses = SE3(poses).matrix() # [N, 4, 4]
            # disps = torch.index_select(video.disps, 0, dirty_index)
            """disps = torch.index_select(video.disps_up, 0, dirty_index).clone()
            Ps = SE3(poses).inv().matrix().cpu().numpy()

            images = torch.index_select(video.images, 0, dirty_index).clone()
            # images = images.cpu()[:,[2,1,0],3::8,3::8].permute(0,2,3,1) / 255.0
            # images = F.interpolate(images, scale_factor=0.5)
            images = images.cpu()[:, [2, 1, 0]].permute(0, 2, 3, 1) / 255.0
            points = droid_backends.iproj(SE3(poses).inv().data, disps, video.intrinsics[0]*8).cpu()

            thresh = droid_visualization.filter_thresh * torch.ones_like(disps.mean(dim=[1,2]))
            
            count = droid_backends.depth_filter(
                video.poses, video.disps_up, video.intrinsics[0]*8, dirty_index, thresh)
            count = count.cpu()
            disps = disps.cpu()
            masks = ((count >= 2) & (disps > .5*disps.mean(dim=[1,2], keepdim=True)))"""
            depths = torch.index_select(video.disps_up, 0, dirty_index).unsqueeze(1).unsqueeze(0)
            intr_matrices = intr_mat.unsqueeze(0).repeat(poses.size(0), 1, 1)
            proj_matrices = torch.stack((poses, intr_matrices), dim=1).unsqueeze(0) # [N, 2, 4, 4]
            ref_depth, src_depths = depths[:, -1, ...], depths[:, :-1, ...]
            ref_cam, src_cams = proj_matrices[:, -1, ...], proj_matrices[:, :-1, ...]
            reproj_xyd, in_range = depth_fusion.get_reproj(ref_depth, src_depths, ref_cam, src_cams)
            vis_masks, vis_mask = depth_fusion.vis_filter(ref_depth, reproj_xyd, in_range,
                                                          droid_visualization.thresh_disp,
                                                          droid_visualization.filter_thresh, droid_visualization.thresh_view)
            ref_depth_avg = depth_fusion.ave_fusion(ref_depth, reproj_xyd, vis_masks)
            # ref_image = video.images[dirty_index[-1]]
            ref_image = video.ref_image[0].permute(1, 2, 0) #/ 255.0
            idx_img = depth_fusion.get_pixel_grids(*ref_depth_avg.size()[-2:]).unsqueeze(0)
            idx_cam = depth_fusion.idx_img2cam(idx_img, ref_depth_avg, ref_cam)
            points = depth_fusion.idx_cam2world(idx_cam, ref_cam)[..., :3, 0]
            ref_pose = torch.inverse(poses[-1]).cpu().numpy()
            ix = dirty_index[-1].item()
            if len(droid_visualization.cameras.keys()) > 1:
                if ix in droid_visualization.cameras:
                    vis.remove_geometry(droid_visualization.cameras[ix])
                    del droid_visualization.cameras[ix]

                if ix in droid_visualization.points:
                    vis.remove_geometry(droid_visualization.points[ix])
                    del droid_visualization.points[ix]

            ### add camera actor ###
            cam_actor = create_camera_actor(True)
            cam_actor.transform(ref_pose)
            vis.add_geometry(cam_actor)
            droid_visualization.cameras[ix] = cam_actor

            mask = vis_mask.reshape(-1)
            pts = points.reshape(-1, 3)[mask].cpu().numpy()
            clr = ref_image.reshape(-1, 3)[mask].cpu().numpy()

            ## add point actor ###
            point_actor = create_point_actor(pts, clr)
            vis.add_geometry(point_actor)
            droid_visualization.points[ix] = point_actor

            # for i in range(len(dirty_index)):
            #     pose = Ps[i]
            #     ix = dirty_index[i].item()
            #
            #     if ix in droid_visualization.cameras:
            #         vis.remove_geometry(droid_visualization.cameras[ix])
            #         del droid_visualization.cameras[ix]
            #
            #     if ix in droid_visualization.points:
            #         vis.remove_geometry(droid_visualization.points[ix])
            #         del droid_visualization.points[ix]
            #
            #     ### add camera actor ###
            #     cam_actor = create_camera_actor(True)
            #     cam_actor.transform(pose)
            #     vis.add_geometry(cam_actor)
            #     droid_visualization.cameras[ix] = cam_actor
            #
            #     mask = masks[i].reshape(-1)
            #     pts = points[i].reshape(-1, 3)[mask].numpy()
            #     clr = images[i].reshape(-1, 3)[mask].numpy()
            #
            #     ## add point actor ###
            #     point_actor = create_point_actor(pts, clr)
            #     vis.add_geometry(point_actor)
            #     droid_visualization.points[ix] = point_actor

            # hack to allow interacting with vizualization during inference
            if len(droid_visualization.cameras) >= droid_visualization.warmup:
                cam = vis.get_view_control().convert_from_pinhole_camera_parameters(cam)

            droid_visualization.ix += 1
            vis.poll_events()
            vis.update_renderer()

    ### create Open3D visualization ###
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.register_animation_callback(animation_callback)
    vis.register_key_callback(ord("S"), increase_filter)
    vis.register_key_callback(ord("A"), decrease_filter)

    vis.create_window(height=540, width=960)
    vis.get_render_option().load_from_json("misc/renderoption.json")

    vis.run()
    vis.destroy_window()
