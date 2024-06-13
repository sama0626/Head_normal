## Introduction
  This is the normal estimation code provided in "SeIF: Semantic-constrained Deep Implicit Function for Single-image 3D Head Reconstruction". 
   Note that all of the code and results can be only used for research purposes.

## Demos
  First, The pre-trained normal estimation model can be downloaded from
  ```
https://pan.baidu.com/s/1ay8g8qRWZ0Uw_i2kXOAWWg
```
  Access code: SeIF
  
  Then, Create a checkpoints directory and place the downloaded model in this directory.

  At this time, run the following code:
```
 python demo.py
```
  
  This code inputs the picture in the image folder and outputs the normal information from this picture. The corresponding results are saved in a image folder. Such as input: image/1.png, output: image/1_normal.png.

！！！Note that the reconstruction results are better with higher resolution and positive pose of the input image.

  <div align=center>
<img src="https://github.com/starVisionTeam/SeIF/blob/master/demo/2.png"  />
</div>
<p align="center">Figure 1: Input.</p>

<div align=center>
<img src="https://github.com/starVisionTeam/SeIF/blob/master/demo/2_normal.png"  />
</div>
<p align="center">Figure 2: Output.</p>
