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


