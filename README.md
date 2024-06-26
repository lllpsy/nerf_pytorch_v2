# modify the data configuration

add a new configuration file:

in configs/robo_test.txt



```
expname = blender_test_robo_only_train
basedir = ./logs
datadir = /home/melody/nerf_pytorch_v2/data/nerf_synthetic/robo
dataset_type = blender

no_batching = True

use_viewdirs = True
white_bkgd = False
lrate_decay = 500



i_img = 100000
i_weights = 500
i_testset = 100000
i_video = 100000
i_print = 10

N_rand = 256
N_samples = 32
N_importance = 64

precrop_iters = 500
precrop_frac = 0.5

half_res = False
```





things I’ve changed

> white_bkgd: false
>
> half_res:false
>
> i_img = 50->100000
>
> i_testset = 2500->100000
>
> i_video = 2500->100000
>
> lrate= 5e-4
>
> i_weights:10000->500
>
> i_print:100->10
>
> N_rand = 1024->256 N_samples = 64->32 N_importance = 128->64
>
> precrop_iters = 500->50





# Dealing with new data formats

under data/nerf_synthetic/robo folder:

there is a frame_0.json file

and a imgs_0 folder, which contains 5 .png format images



# modify the data preprocessing code

in load_blender_small_data.py:

```python
metas = {}
frame_no = 0

with open(os.path.join(basedir,'frame_{}.json'.format(frame_no)),'r') as fp:
    metas[frame_no] = json.load(fp)
```



```python
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
```



```python
camera_angle_x = float(metas[frame_no]['camera_angle_x'])

# 0.7853981633974483
```

# result
after 10000 iterations:

half_res = False

N_rand = 1024->256

N_samples = 64->32

N_importance = 128->64

precrop_iters = 500->50

N_iters = 200000+1->10000 + 1


![image](https://github.com/lllpsy/rl-lab/assets/59329407/7eb7bf71-b610-4e0f-9ed2-34e27dc50bcb)


[TRAIN] Iter: 10000 Loss: 0.001231816248036921  PSNR: 31.56230926513672   


after 100000 iterations:

N_rand = 1024->256

half_res = True

N_iters = 200000+1->100000 + 1
![image](https://github.com/lllpsy/nerf_pytorch_v2/assets/59329407/c4303114-43cd-4fb0-b092-3bb3a4f5f29b)


[TRAIN] Iter: 100000 Loss: 6.640172614424955e-06  PSNR: 54.33806610107422


# temp
![image](https://github.com/lllpsy/nerf_pytorch_v2/assets/59329407/30f00e57-a570-44a3-bc62-2f1f14673e3d)

