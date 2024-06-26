# modify the data configuration

add a new configuration file:

in configs/robo_test.txt

(this is for 100k iterations)



```
expname = blender_test_robo_only_train
basedir = ./logs
datadir = /home/melody/nerf_pytorch_v2/data/nerf_synthetic/robo
dataset_type = blender

no_batching = True

use_viewdirs = True
white_bkgd = False
lrate_decay = 500



i_img = 400000
i_weights = 10000
i_testset = 400000
i_video = 100000
i_print = 100

N_rand = 256
N_samples = 64
N_importance = 128

precrop_iters = 500
precrop_frac = 0.5

half_res = True
```





things I’ve changed

> white_bkgd: false
>
> i_img = 500->400k
>
> i_testset = 50k->400k
>
> i_video = 50k->400k
>
> N_rand = 1024->256
>





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

this is the result for new dataset without test


white_bkgd: false


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


# if we add a test image?

continue to modify the code to pad the image and generate the correct test image


in configs/robo_test.txt

```
white_bkgd = True
```



in load_blender_small_data.py

```python
def pad_image(image, macro_block_size=16):
    height, width = image.shape[:2]
    new_height = ((height + macro_block_size - 1) // macro_block_size) * macro_block_size
    new_width = ((width + macro_block_size - 1) // macro_block_size) * macro_block_size
    padded_image = np.zeros((new_height, new_width, 3), dtype=image.dtype)
    padded_image[:height, :width] = image
    alpha_channel = np.zeros((new_height, new_width, 1), dtype=image.dtype)
    alpha_channel[:height, :width] = 255
    padded_image_rgba = np.concatenate((padded_image, alpha_channel), axis=-1)
    return padded_image_rgba

```



in def load_blender_data(basedir, half_res=False, testskip=1):

```python
    imgs=[]    
    
    for frame in metas[frame_no]['frames'][::1]:
        fname = os.path.join(basedir,frame['file_path'] + '.png')
        img = imageio.imread(fname)
        img = pad_image(img)
        imgs.append(img)
        poses.append(np.array(frame['transform_matrix']))

 # imgs:(5,96,96,4)
```





in run_nerf_helpers.py

```python
def to8b(x):
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    x = np.clip(x, 0, 1)
    return (255 * x).astype(np.uint8)

```



when iteration is 200k:

![image](https://github.com/lllpsy/nerf_pytorch_v2/assets/59329407/30f00e57-a570-44a3-bc62-2f1f14673e3d)


however, unfortunately, it can't generate a clear test image

