import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def load_blender_data(basedir, half_res=False, testskip=1):
    metas = {}
    frame_no = 0

    with open(os.path.join(basedir,'frame_{}.json'.format(frame_no)),'r') as fp:
        metas[frame_no] = json.load(fp)


    # i_split = [np.arange(3),np.array([]),np.arange(3,4)]
    #train-3  test,val-1

    i_split = [np.arange(5), np.array([]), np.array([])]

    # imgs=[]
    # poses=[]
    # for frame in metas[frame_no]['frames'][::1]:
    #     fname = os.path.join(basedir,frame['file_path'] + '.npy')
    #     imgs.append(np.load(fname))
    #     poses.append(np.array(frame['transform_matrix']))
    #
    # imgs = (np.array(imgs)/255.).astype(np.float32)
    # #(4,3,84,84)
    # poses = np.array(poses).astype(np.float32)
    # #(4,4,4)
    # imgs = np.transpose(imgs, (0, 2, 3, 1))

    imgs=[]
    poses=[]
    for frame in metas[frame_no]['frames'][::1]:
        fname = os.path.join(basedir,frame['file_path'] + '.png')
        imgs.append(imageio.imread(fname))
        poses.append(np.array(frame['transform_matrix']))

    imgs = (np.array(imgs)/255.).astype(np.float32)
    #(5,84,84,3)
    poses = np.array(poses).astype(np.float32)
    #(5,4,4)



    H, W = imgs[0].shape[:2]
    #84,84

    camera_angle_x = float(metas[frame_no]['camera_angle_x'])
    # 0.7853981633974483

    focal = .5 * W / np.tan(.5 * camera_angle_x)
    #1111.1110311937682
    
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)

    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs_0 = tf.image.resize_area(imgs_0, [400, 400]).numpy()

        
    return imgs, poses, render_poses, [H, W, focal], i_split


