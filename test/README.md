If you want to compute the psnr and ssim reported in the paper, please download datasets from the following links and place them in this directory. Your directory tree should look like this

`derain` <br/>
  `├──`[input]([Derain](https://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html))  <br/>
  `└──`target <br/>

`dehaze` <br/>
  `├──`[input](https://sites.google.com/view/reside-dehaze-datasets/reside-v0)  <br/>
  `└──`target <br/>

For denoise dir, you could just put the clean images (e.g, BSD68 or Urban100) into the corresponding directory. In the paper, we test our AirNet on BSD68 and Urban100.

For deraining, we test our AirNet on Rain100L. 

For dehazing, we test our AirNet on SOTS outdoor. 

If you do not want to get the qualitative results only, please put your test images into test/, and run demo.py to compute the qualitative results only.