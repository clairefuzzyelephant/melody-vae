# melody-vae
training a vae that generates a melody

Dataset (type0 folder): 320 single-track (Type 0) piano MIDI files from various classical composers, manually downloaded from http://www.piano-midi.de/midicoll.htm

changing from using vae -> cvae, vae with cnn


## CVAE Notebook
- Example from [pytorch-vae Github](https://github.com/sksq96/pytorch-vae) 
### Install Dependencies
- `pip install imageio, torchsummary, scikit-image`
- `pip install --upgrade scikit-image --user`

Need libsndfile and audiolab for playback.
- `brew install libsndfile`
- `pip install scikits.audiolab`
- `brew install https://raw.githubusercontent.com/Homebrew/homebrew-core/34dcd1ff65a56c3191fa57d3dd23e7fffd55fae8/Formula/fluid-synth.rb`

## Fix
- `AttributeError: dlsym(0x7fd2dbaf2010, fluid_synth_get_channel_info): symbol not found` during `import fluidsynth`

[Github pyfluidsynth Issue Sol](https://github.com/nwhitehead/pyfluidsynth/issues/19)
```
brew uninstall fluidsynth
brew install pkg-config
git clone https://github.com/FluidSynth/fluidsynth.git
cd fluidsynth
git checkout 1.1.x
mkdir build
cd build
cmake ..
sudo make install
fluidsynth --version
```

## Useful Readings:
- [Pianoroll Dataset Blog](https://salu133445.github.io/lakh-pianoroll-dataset/representation.html)


## Trainings results
```
Epoch[1/50] Loss: 1323.707 1322.040 1.667 tensor(1323.7068) tensor(1.6667) 1
Epoch[2/50] Loss: 1238.448 1237.332 1.116 tensor(1238.4482) tensor(1.1158) 1
Epoch[3/50] Loss: 1155.286 1153.052 2.234 tensor(1155.2858) tensor(2.2342) 1
Epoch[4/50] Loss: 1385.116 1383.161 1.955 tensor(1385.1163) tensor(1.9550) 1
Epoch[5/50] Loss: 1606.427 1604.125 2.302 tensor(1606.4270) tensor(2.3019) 1
Epoch[6/50] Loss: 696.221 693.394 2.827 tensor(696.2215) tensor(2.8272) 1
Epoch[7/50] Loss: 1300.368 1298.059 2.308 tensor(1300.3676) tensor(2.3082) 1
Epoch[8/50] Loss: 621.003 617.363 3.639 tensor(621.0027) tensor(3.6394) 1
Epoch[9/50] Loss: 932.638 928.759 3.879 tensor(932.6377) tensor(3.8791) 1
Epoch[10/50] Loss: 1272.932 1270.671 2.261 tensor(1272.9320) tensor(2.2612) 1
Epoch[11/50] Loss: 506.531 503.439 3.092 tensor(506.5305) tensor(3.0916) 1
Epoch[12/50] Loss: 764.249 760.731 3.518 tensor(764.2493) tensor(3.5184) 1
Epoch[13/50] Loss: 746.649 743.370 3.280 tensor(746.6494) tensor(3.2797) 1
Epoch[14/50] Loss: 852.725 849.800 2.925 tensor(852.7245) tensor(2.9247) 1
Epoch[15/50] Loss: 26.956 23.881 3.074 tensor(26.9558) tensor(3.0745) 1
Epoch[16/50] Loss: 1249.462 1246.455 3.007 tensor(1249.4624) tensor(3.0074) 1
Epoch[17/50] Loss: 238.564 234.801 3.763 tensor(238.5639) tensor(3.7631) 1
Epoch[18/50] Loss: 635.482 632.427 3.055 tensor(635.4818) tensor(3.0548) 1
Epoch[19/50] Loss: 732.926 729.239 3.688 tensor(732.9264) tensor(3.6876) 1
Epoch[20/50] Loss: 688.646 684.659 3.987 tensor(688.6463) tensor(3.9869) 1
Epoch[21/50] Loss: 619.761 616.126 3.635 tensor(619.7611) tensor(3.6350) 1
Epoch[22/50] Loss: 94.686 90.471 4.215 tensor(94.6856) tensor(4.2148) 1
Epoch[23/50] Loss: 943.898 940.307 3.591 tensor(943.8983) tensor(3.5912) 1
Epoch[24/50] Loss: 291.560 288.461 3.099 tensor(291.5598) tensor(3.0987) 1
Epoch[25/50] Loss: 177.507 172.539 4.968 tensor(177.5071) tensor(4.9684) 1
Epoch[26/50] Loss: 628.294 624.818 3.476 tensor(628.2936) tensor(3.4755) 1
Epoch[27/50] Loss: 526.630 523.218 3.412 tensor(526.6296) tensor(3.4116) 1
Epoch[28/50] Loss: 486.030 482.661 3.369 tensor(486.0302) tensor(3.3695) 1
Epoch[29/50] Loss: 190.518 186.003 4.515 tensor(190.5177) tensor(4.5148) 1
Epoch[30/50] Loss: 264.189 261.195 2.994 tensor(264.1891) tensor(2.9943) 1
Epoch[31/50] Loss: 382.654 378.038 4.616 tensor(382.6541) tensor(4.6160) 1
Epoch[32/50] Loss: 195.452 192.280 3.172 tensor(195.4518) tensor(3.1721) 1
Epoch[33/50] Loss: 459.840 455.853 3.987 tensor(459.8400) tensor(3.9868) 1
Epoch[34/50] Loss: 326.989 324.011 2.978 tensor(326.9892) tensor(2.9778) 1
Epoch[35/50] Loss: 132.950 128.427 4.523 tensor(132.9502) tensor(4.5231) 1
Epoch[36/50] Loss: 140.350 136.698 3.652 tensor(140.3501) tensor(3.6518) 1
Epoch[37/50] Loss: 327.773 323.197 4.575 tensor(327.7726) tensor(4.5754) 1
Epoch[38/50] Loss: 73.060 69.361 3.698 tensor(73.0599) tensor(3.6984) 1
Epoch[39/50] Loss: 191.330 186.770 4.559 tensor(191.3298) tensor(4.5594) 1
Epoch[40/50] Loss: 490.440 486.355 4.086 tensor(490.4403) tensor(4.0857) 1
Epoch[41/50] Loss: 332.505 329.190 3.316 tensor(332.5053) tensor(3.3158) 1
Epoch[42/50] Loss: 277.392 273.471 3.921 tensor(277.3920) tensor(3.9214) 1
Epoch[43/50] Loss: 98.593 95.013 3.580 tensor(98.5932) tensor(3.5802) 1
Epoch[44/50] Loss: 251.590 247.713 3.877 tensor(251.5903) tensor(3.8769) 1
Epoch[45/50] Loss: 78.620 73.099 5.521 tensor(78.6201) tensor(5.5206) 1
Epoch[46/50] Loss: 180.351 175.629 4.722 tensor(180.3507) tensor(4.7217) 1
Epoch[47/50] Loss: 143.110 139.206 3.904 tensor(143.1101) tensor(3.9044) 1
Epoch[48/50] Loss: 339.833 335.981 3.852 tensor(339.8329) tensor(3.8517) 1
Epoch[49/50] Loss: 343.103 339.104 3.999 tensor(343.1026) tensor(3.9988) 1
Epoch[50/50] Loss: 82.478 78.221 4.257 tensor(82.4782) tensor(4.2574) 1
Notifing Epoch[50/50] Loss: 82.478 78.221 4.257 Just because I have to put something. <Response [400]>
```


- batch 16
```
Epoch[1/100] Loss: 1022.473 1022.422 0.050 tensor(32719.1230) tensor(1.6123) 32
Epoch[2/100] Loss: 981.394 981.290 0.104 tensor(31404.6074) tensor(3.3354) 32
Epoch[3/100] Loss: 867.052 866.868 0.183 tensor(27745.6562) tensor(5.8669) 32
Epoch[4/100] Loss: 1002.440 1002.264 0.175 tensor(32078.0684) tensor(5.6124) 32
Epoch[5/100] Loss: 894.700 894.469 0.232 tensor(28630.4082) tensor(7.4089) 32
Epoch[6/100] Loss: 958.333 958.117 0.216 tensor(30666.6582) tensor(6.9247) 32
Epoch[7/100] Loss: 774.296 774.063 0.233 tensor(24777.4707) tensor(7.4503) 32
Epoch[8/100] Loss: 741.276 741.032 0.243 tensor(23720.8223) tensor(7.7861) 32
Epoch[9/100] Loss: 731.365 731.101 0.264 tensor(23403.6699) tensor(8.4423) 32
Epoch[10/100] Loss: 709.043 708.793 0.250 tensor(22689.3750) tensor(8.0074) 32
Epoch[11/100] Loss: 665.015 664.748 0.267 tensor(21280.4668) tensor(8.5294) 32
Epoch[12/100] Loss: 609.271 609.038 0.232 tensor(19496.6582) tensor(7.4290) 32
Epoch[13/100] Loss: 531.731 531.469 0.262 tensor(17015.3848) tensor(8.3860) 32
Epoch[14/100] Loss: 626.180 625.901 0.279 tensor(20037.7715) tensor(8.9236) 32
Epoch[15/100] Loss: 515.033 514.736 0.297 tensor(16481.0547) tensor(9.4946) 32
Epoch[16/100] Loss: 475.266 474.990 0.276 tensor(15208.4961) tensor(8.8199) 32
Epoch[17/100] Loss: 458.548 458.210 0.338 tensor(14673.5498) tensor(10.8207) 32
Epoch[18/100] Loss: 451.229 450.911 0.318 tensor(14439.3291) tensor(10.1888) 32
Epoch[19/100] Loss: 440.151 439.817 0.334 tensor(14084.8389) tensor(10.7004) 32
Epoch[20/100] Loss: 470.045 469.681 0.365 tensor(15041.4551) tensor(11.6649) 32
Epoch[21/100] Loss: 425.007 424.687 0.320 tensor(13600.2354) tensor(10.2556) 32
Epoch[22/100] Loss: 351.273 350.893 0.380 tensor(11240.7266) tensor(12.1515) 32
Epoch[23/100] Loss: 337.725 337.330 0.395 tensor(10807.1992) tensor(12.6396) 32
Epoch[24/100] Loss: 303.201 302.774 0.427 tensor(9702.4336) tensor(13.6630) 32
Epoch[25/100] Loss: 202.109 201.578 0.531 tensor(6467.4912) tensor(17.0051) 32
Epoch[26/100] Loss: 289.052 288.631 0.420 tensor(9249.6543) tensor(13.4474) 32
Epoch[27/100] Loss: 220.268 219.776 0.491 tensor(7048.5728) tensor(15.7266) 32
Epoch[28/100] Loss: 209.482 208.994 0.488 tensor(6703.4277) tensor(15.6124) 32
Epoch[29/100] Loss: 180.330 179.852 0.478 tensor(5770.5640) tensor(15.3084) 32
Epoch[30/100] Loss: 224.042 223.571 0.470 tensor(7169.3359) tensor(15.0535) 32
Epoch[31/100] Loss: 204.807 204.302 0.505 tensor(6553.8174) tensor(16.1526) 32
Epoch[32/100] Loss: 194.608 194.075 0.533 tensor(6227.4604) tensor(17.0543) 32
Epoch[33/100] Loss: 185.073 184.511 0.562 tensor(5922.3501) tensor(18.0000) 32
Epoch[34/100] Loss: 162.628 162.053 0.576 tensor(5204.1050) tensor(18.4210) 32
Epoch[35/100] Loss: 145.588 145.057 0.530 tensor(4658.8037) tensor(16.9732) 32
Epoch[36/100] Loss: 113.608 112.981 0.627 tensor(3635.4619) tensor(20.0666) 32
Epoch[37/100] Loss: 130.991 130.476 0.515 tensor(4191.7114) tensor(16.4682) 32
Epoch[38/100] Loss: 110.716 110.131 0.585 tensor(3542.9143) tensor(18.7225) 32
Epoch[39/100] Loss: 118.706 118.155 0.551 tensor(3798.6006) tensor(17.6257) 32
Epoch[40/100] Loss: 104.342 103.807 0.535 tensor(3338.9446) tensor(17.1180) 32
Epoch[41/100] Loss: 125.945 125.384 0.560 tensor(4030.2253) tensor(17.9306) 32
Epoch[42/100] Loss: 101.821 101.256 0.565 tensor(3258.2651) tensor(18.0835) 32
Epoch[43/100] Loss: 100.361 99.830 0.532 tensor(3211.5569) tensor(17.0110) 32
Epoch[44/100] Loss: 100.835 100.311 0.523 tensor(3226.7048) tensor(16.7495) 32
Epoch[45/100] Loss: 99.008 98.432 0.576 tensor(3168.2595) tensor(18.4204) 32
Epoch[46/100] Loss: 92.914 92.396 0.518 tensor(2973.2454) tensor(16.5791) 32
Epoch[47/100] Loss: 99.185 98.679 0.506 tensor(3173.9351) tensor(16.2006) 32
Epoch[48/100] Loss: 87.625 87.051 0.574 tensor(2804.0037) tensor(18.3714) 32
Epoch[49/100] Loss: 120.083 119.658 0.425 tensor(3842.6523) tensor(13.6032) 32
Epoch[50/100] Loss: 82.940 82.506 0.434 tensor(2654.0933) tensor(13.8950) 32
Epoch[51/100] Loss: 75.928 75.408 0.519 tensor(2429.6902) tensor(16.6231) 32
Epoch[52/100] Loss: 81.048 80.614 0.434 tensor(2593.5266) tensor(13.8907) 32
Epoch[53/100] Loss: 72.597 72.067 0.530 tensor(2323.1030) tensor(16.9534) 32
Epoch[54/100] Loss: 64.995 64.530 0.465 tensor(2079.8291) tensor(14.8801) 32
Epoch[55/100] Loss: 53.235 52.754 0.480 tensor(1703.5049) tensor(15.3744) 32
Epoch[56/100] Loss: 69.512 69.047 0.465 tensor(2224.3748) tensor(14.8658) 32
Epoch[57/100] Loss: 61.353 60.781 0.572 tensor(1963.2964) tensor(18.3115) 32
Epoch[58/100] Loss: 46.436 45.948 0.488 tensor(1485.9427) tensor(15.6116) 32
Epoch[59/100] Loss: 58.141 57.693 0.449 tensor(1860.5164) tensor(14.3521) 32
Epoch[60/100] Loss: 64.824 64.371 0.454 tensor(2074.3765) tensor(14.5180) 32
Epoch[61/100] Loss: 51.889 51.455 0.434 tensor(1660.4485) tensor(13.8977) 32
Epoch[62/100] Loss: 60.101 59.656 0.445 tensor(1923.2397) tensor(14.2336) 32
Epoch[63/100] Loss: 65.835 65.413 0.422 tensor(2106.7258) tensor(13.5031) 32
Epoch[64/100] Loss: 63.463 63.036 0.427 tensor(2030.8120) tensor(13.6488) 32
Epoch[65/100] Loss: 49.551 49.125 0.426 tensor(1585.6432) tensor(13.6408) 32
Epoch[66/100] Loss: 53.832 53.434 0.398 tensor(1722.6190) tensor(12.7214) 32
Epoch[67/100] Loss: 49.889 49.519 0.370 tensor(1596.4552) tensor(11.8502) 32
Epoch[68/100] Loss: 67.331 66.942 0.389 tensor(2154.5857) tensor(12.4436) 32
Epoch[69/100] Loss: 59.948 59.517 0.431 tensor(1918.3308) tensor(13.7878) 32
Epoch[70/100] Loss: 77.529 77.121 0.408 tensor(2480.9304) tensor(13.0424) 32
Epoch[71/100] Loss: 59.683 59.313 0.370 tensor(1909.8550) tensor(11.8362) 32
Epoch[72/100] Loss: 51.356 50.959 0.397 tensor(1643.4056) tensor(12.7074) 32
Epoch[73/100] Loss: 39.866 39.472 0.394 tensor(1275.7135) tensor(12.6123) 32
Epoch[74/100] Loss: 45.177 44.821 0.357 tensor(1445.6711) tensor(11.4130) 32
Epoch[75/100] Loss: 51.050 50.674 0.376 tensor(1633.5885) tensor(12.0357) 32
Epoch[76/100] Loss: 72.697 72.377 0.320 tensor(2326.2898) tensor(10.2357) 32
Epoch[77/100] Loss: 61.694 61.363 0.331 tensor(1974.2017) tensor(10.5925) 32
Epoch[78/100] Loss: 59.917 59.589 0.328 tensor(1917.3311) tensor(10.4812) 32
Epoch[79/100] Loss: 46.735 46.403 0.331 tensor(1495.5040) tensor(10.5954) 32
Epoch[80/100] Loss: 24.212 23.841 0.371 tensor(774.7726) tensor(11.8600) 32
Epoch[81/100] Loss: 29.538 29.151 0.387 tensor(945.2115) tensor(12.3832) 32
Epoch[82/100] Loss: 34.068 33.648 0.420 tensor(1090.1836) tensor(13.4556) 32
Epoch[83/100] Loss: 35.178 34.790 0.388 tensor(1125.6996) tensor(12.4206) 32
Epoch[84/100] Loss: 25.432 25.022 0.411 tensor(813.8305) tensor(13.1405) 32
Epoch[85/100] Loss: 29.545 29.158 0.387 tensor(945.4327) tensor(12.3923) 32
Epoch[86/100] Loss: 33.086 32.717 0.369 tensor(1058.7628) tensor(11.8151) 32
Epoch[87/100] Loss: 39.971 39.599 0.372 tensor(1279.0695) tensor(11.8981) 32
Epoch[88/100] Loss: 52.792 52.505 0.287 tensor(1689.3473) tensor(9.1772) 32
Epoch[89/100] Loss: 51.091 50.815 0.276 tensor(1634.9216) tensor(8.8350) 32
Epoch[90/100] Loss: 57.966 57.667 0.299 tensor(1854.9176) tensor(9.5733) 32
Epoch[91/100] Loss: 42.478 42.170 0.307 tensor(1359.2908) tensor(9.8354) 32
Epoch[92/100] Loss: 52.672 52.391 0.281 tensor(1685.5137) tensor(9.0055) 32
Epoch[93/100] Loss: 43.447 43.159 0.288 tensor(1390.2915) tensor(9.2185) 32
Epoch[94/100] Loss: 42.037 41.752 0.285 tensor(1345.1852) tensor(9.1240) 32
Epoch[95/100] Loss: 44.425 44.139 0.286 tensor(1421.6115) tensor(9.1665) 32
Epoch[96/100] Loss: 34.721 34.410 0.311 tensor(1111.0582) tensor(9.9446) 32
Epoch[97/100] Loss: 26.972 26.635 0.338 tensor(863.1173) tensor(10.8002) 32
Epoch[98/100] Loss: 27.940 27.657 0.283 tensor(894.0865) tensor(9.0513) 32
Epoch[99/100] Loss: 22.050 21.700 0.349 tensor(705.5853) tensor(11.1721) 32
Epoch[100/100] Loss: 23.225 22.887 0.339 tensor(743.2123) tensor(10.8354) 32
Notifing Epoch[100/100] Loss: 23.225 22.887 0.339 Just because I have to put something. <Response [400]>
```

- batchnorm2d + tanh
```
Epoch[1/100] Loss: 8405.504 8405.366 0.138 tensor(268976.1250) tensor(4.4091) 32
Epoch[2/100] Loss: 7841.206 7841.023 0.183 tensor(250918.5781) tensor(5.8483) 32
```

- batchnorm2d + tanh, [-1, 1] image
~ 7400 when use -1, 1 image

- batchnorm2D + tanh 
```
Epoch[1/300] Loss: 8367.783 8367.661 0.122 tensor(267769.0625) tensor(3.9035) 32
Epoch[2/300] Loss: 7883.275 7883.100 0.175 tensor(252264.7969) tensor(5.6032) 32
Epoch[3/300] Loss: 7428.754 7428.537 0.217 tensor(237720.1250) tensor(6.9362) 32
Epoch[4/300] Loss: 7032.993 7032.749 0.244 tensor(225055.7812) tensor(7.8100) 32
Epoch[5/300] Loss: 6628.666 6628.385 0.281 tensor(212117.3125) tensor(8.9950) 32
Epoch[6/300] Loss: 6290.156 6289.831 0.325 tensor(201285.) tensor(10.4032) 32
Epoch[7/300] Loss: 6010.559 6010.223 0.336 tensor(192337.8906) tensor(10.7518) 32
Epoch[8/300] Loss: 5675.315 5674.965 0.351 tensor(181610.0938) tensor(11.2245) 32
Epoch[9/300] Loss: 5372.935 5372.548 0.387 tensor(171933.9062) tensor(12.3762) 32
Epoch[10/300] Loss: 5156.219 5155.802 0.417 tensor(164999.) tensor(13.3473) 32
Epoch[11/300] Loss: 4845.917 4845.499 0.418 tensor(155069.3438) tensor(13.3866) 32
Epoch[12/300] Loss: 4530.906 4530.472 0.434 tensor(144988.9844) tensor(13.8902) 32
Epoch[13/300] Loss: 4457.497 4457.073 0.424 tensor(142639.9062) tensor(13.5653) 32
Epoch[14/300] Loss: 4179.242 4178.800 0.441 tensor(133735.7344) tensor(14.1178) 32
Epoch[15/300] Loss: 4039.120 4038.646 0.475 tensor(129251.8516) tensor(15.1958) 32
Epoch[16/300] Loss: 3805.753 3805.255 0.498 tensor(121784.1016) tensor(15.9421) 32
Epoch[17/300] Loss: 3619.314 3618.764 0.550 tensor(115818.0391) tensor(17.5953) 32
Epoch[18/300] Loss: 3426.547 3426.016 0.532 tensor(109649.5156) tensor(17.0156) 32
Epoch[19/300] Loss: 3227.848 3227.312 0.537 tensor(103291.1484) tensor(17.1700) 32
Epoch[20/300] Loss: 3184.563 3184.029 0.534 tensor(101906.0156) tensor(17.0875) 32
Epoch[21/300] Loss: 3011.204 3010.642 0.562 tensor(96358.5312) tensor(17.9956) 32
Epoch[22/300] Loss: 2897.254 2896.683 0.571 tensor(92712.1250) tensor(18.2625) 32
Epoch[23/300] Loss: 2796.849 2796.278 0.572 tensor(89499.1797) tensor(18.2890) 32
Epoch[24/300] Loss: 2577.289 2576.713 0.576 tensor(82473.2500) tensor(18.4399) 32
Epoch[25/300] Loss: 2472.892 2472.303 0.589 tensor(79132.5312) tensor(18.8445) 32
Epoch[26/300] Loss: 2428.278 2427.661 0.617 tensor(77704.9062) tensor(19.7523) 32
Epoch[27/300] Loss: 2296.262 2295.685 0.578 tensor(73480.3906) tensor(18.4861) 32
Epoch[28/300] Loss: 2204.026 2203.439 0.587 tensor(70528.8203) tensor(18.7798) 32
Epoch[29/300] Loss: 2097.986 2097.379 0.607 tensor(67135.5391) tensor(19.4255) 32
Epoch[30/300] Loss: 2058.738 2058.111 0.627 tensor(65879.6250) tensor(20.0795) 32
Epoch[31/300] Loss: 1998.374 1997.735 0.639 tensor(63947.9766) tensor(20.4418) 32
Epoch[32/300] Loss: 1859.543 1858.875 0.669 tensor(59505.3906) tensor(21.4036) 32
Epoch[33/300] Loss: 1805.919 1805.252 0.667 tensor(57789.4102) tensor(21.3417) 32
Epoch[34/300] Loss: 1706.316 1705.682 0.634 tensor(54602.1211) tensor(20.2799) 32
Epoch[35/300] Loss: 1727.627 1726.926 0.702 tensor(55284.0664) tensor(22.4483) 32
Epoch[36/300] Loss: 1673.236 1672.550 0.686 tensor(53543.5508) tensor(21.9573) 32
Epoch[37/300] Loss: 1600.327 1599.651 0.676 tensor(51210.4648) tensor(21.6341) 32
Epoch[38/300] Loss: 1577.615 1576.933 0.682 tensor(50483.6914) tensor(21.8204) 32
Epoch[39/300] Loss: 1490.196 1489.493 0.702 tensor(47686.2617) tensor(22.4727) 32
Epoch[40/300] Loss: 1393.024 1392.359 0.665 tensor(44576.7812) tensor(21.2959) 32
Epoch[41/300] Loss: 1351.601 1350.907 0.693 tensor(43251.2266) tensor(22.1918) 32
Epoch[42/300] Loss: 1402.677 1401.960 0.717 tensor(44885.6641) tensor(22.9427) 32
Epoch[43/300] Loss: 1246.855 1246.091 0.764 tensor(39899.3516) tensor(24.4489) 32
Epoch[44/300] Loss: 1228.954 1228.217 0.737 tensor(39326.5195) tensor(23.5713) 32
Epoch[45/300] Loss: 1268.897 1268.156 0.740 tensor(40604.6914) tensor(23.6921) 32
Epoch[46/300] Loss: 1062.536 1061.808 0.728 tensor(34001.1523) tensor(23.3052) 32
Epoch[47/300] Loss: 1046.700 1045.942 0.759 tensor(33494.4102) tensor(24.2736) 32
Epoch[48/300] Loss: 1093.262 1092.495 0.767 tensor(34984.3867) tensor(24.5447) 32
Epoch[49/300] Loss: 1025.675 1024.910 0.765 tensor(32821.6016) tensor(24.4932) 32
Epoch[50/300] Loss: 1023.426 1022.637 0.789 tensor(32749.6289) tensor(25.2387) 32
Epoch[51/300] Loss: 979.018 978.225 0.793 tensor(31328.5859) tensor(25.3716) 32
Epoch[52/300] Loss: 909.334 908.539 0.796 tensor(29098.7031) tensor(25.4682) 32
Epoch[53/300] Loss: 963.502 962.722 0.780 tensor(30832.0625) tensor(24.9724) 32
Epoch[54/300] Loss: 854.510 853.738 0.772 tensor(27344.3203) tensor(24.7162) 32
Epoch[55/300] Loss: 868.091 867.284 0.806 tensor(27778.8965) tensor(25.7973) 32
Epoch[56/300] Loss: 798.809 797.987 0.822 tensor(25561.8750) tensor(26.2885) 32
Epoch[57/300] Loss: 771.781 771.008 0.773 tensor(24696.9883) tensor(24.7405) 32
Epoch[58/300] Loss: 750.685 749.886 0.799 tensor(24021.9082) tensor(25.5571) 32
Epoch[59/300] Loss: 759.747 758.959 0.787 tensor(24311.8945) tensor(25.1977) 32
Epoch[60/300] Loss: 668.543 667.732 0.811 tensor(21393.3770) tensor(25.9527) 32
Epoch[61/300] Loss: 700.680 699.868 0.812 tensor(22421.7500) tensor(25.9881) 32
Epoch[62/300] Loss: 782.138 781.370 0.768 tensor(25028.4121) tensor(24.5734) 32
Epoch[63/300] Loss: 669.014 668.166 0.848 tensor(21408.4590) tensor(27.1362) 32
Epoch[64/300] Loss: 662.997 662.142 0.855 tensor(21215.9004) tensor(27.3565) 32
Epoch[65/300] Loss: 613.526 612.744 0.782 tensor(19632.8281) tensor(25.0124) 32
Epoch[66/300] Loss: 606.274 605.393 0.880 tensor(19400.7656) tensor(28.1759) 32
Epoch[67/300] Loss: 634.583 633.781 0.802 tensor(20306.6562) tensor(25.6784) 32
Epoch[68/300] Loss: 541.506 540.645 0.860 tensor(17328.1836) tensor(27.5313) 32
Epoch[69/300] Loss: 583.074 582.215 0.859 tensor(18658.3691) tensor(27.4878) 32
Epoch[70/300] Loss: 545.977 545.091 0.886 tensor(17471.2598) tensor(28.3383) 32
Epoch[71/300] Loss: 528.777 527.926 0.852 tensor(16920.8730) tensor(27.2547) 32
Epoch[72/300] Loss: 481.317 480.476 0.841 tensor(15402.1523) tensor(26.9158) 32
Epoch[73/300] Loss: 488.113 487.293 0.821 tensor(15619.6299) tensor(26.2696) 32
Epoch[74/300] Loss: 476.606 475.696 0.910 tensor(15251.3857) tensor(29.1145) 32
Epoch[75/300] Loss: 497.661 496.781 0.879 tensor(15925.1387) tensor(28.1419) 32
Epoch[76/300] Loss: 454.993 454.133 0.860 tensor(14559.7666) tensor(27.5209) 32
Epoch[77/300] Loss: 410.050 409.137 0.913 tensor(13121.5850) tensor(29.2058) 32
Epoch[78/300] Loss: 490.402 489.471 0.931 tensor(15692.8662) tensor(29.7821) 32
Epoch[79/300] Loss: 476.849 475.906 0.944 tensor(15259.1836) tensor(30.1943) 32
Epoch[80/300] Loss: 416.306 415.402 0.904 tensor(13321.8027) tensor(28.9388) 32
Epoch[81/300] Loss: 392.119 391.116 1.004 tensor(12547.8213) tensor(32.1139) 32
Epoch[82/300] Loss: 378.540 377.609 0.931 tensor(12113.2842) tensor(29.7946) 32
Epoch[83/300] Loss: 354.441 353.503 0.938 tensor(11342.1162) tensor(30.0120) 32
Epoch[84/300] Loss: 384.407 383.560 0.848 tensor(12301.0293) tensor(27.1210) 32
Epoch[85/300] Loss: 330.662 329.728 0.934 tensor(10581.1943) tensor(29.8951) 32
Epoch[86/300] Loss: 345.550 344.621 0.929 tensor(11057.5986) tensor(29.7313) 32
Epoch[87/300] Loss: 371.995 371.082 0.913 tensor(11903.8428) tensor(29.2129) 32
Epoch[88/300] Loss: 397.002 396.119 0.883 tensor(12704.0518) tensor(28.2511) 32
Epoch[89/300] Loss: 348.369 347.435 0.935 tensor(11147.8223) tensor(29.9068) 32
Epoch[90/300] Loss: 296.600 295.602 0.999 tensor(9491.2158) tensor(31.9546) 32
Epoch[91/300] Loss: 294.885 293.961 0.924 tensor(9436.3164) tensor(29.5798) 32
Epoch[92/300] Loss: 304.923 303.966 0.957 tensor(9757.5303) tensor(30.6122) 32
Epoch[93/300] Loss: 248.322 247.391 0.930 tensor(7946.2905) tensor(29.7708) 32
Epoch[94/300] Loss: 311.975 310.938 1.037 tensor(9983.1943) tensor(33.1931) 32
Epoch[95/300] Loss: 301.231 300.213 1.018 tensor(9639.3936) tensor(32.5785) 32
Epoch[96/300] Loss: 249.339 248.415 0.925 tensor(7978.8628) tensor(29.5900) 32
Epoch[97/300] Loss: 246.348 245.385 0.963 tensor(7883.1206) tensor(30.8090) 32
Epoch[98/300] Loss: 284.181 283.147 1.034 tensor(9093.7871) tensor(33.0851) 32
Epoch[99/300] Loss: 254.510 253.566 0.944 tensor(8144.3052) tensor(30.1973) 32
Epoch[100/300] Loss: 236.207 235.227 0.980 tensor(7558.6104) tensor(31.3482) 32
Epoch[101/300] Loss: 247.101 246.196 0.906 tensor(7907.2446) tensor(28.9804) 32
Epoch[102/300] Loss: 226.983 226.046 0.937 tensor(7263.4556) tensor(29.9700) 32
Epoch[103/300] Loss: 247.175 246.151 1.024 tensor(7909.6123) tensor(32.7770) 32
Epoch[104/300] Loss: 222.084 221.120 0.964 tensor(7106.6963) tensor(30.8517) 32
Epoch[105/300] Loss: 291.544 290.556 0.988 tensor(9329.4131) tensor(31.6112) 32
Epoch[106/300] Loss: 215.486 214.469 1.017 tensor(6895.5596) tensor(32.5561) 32
Epoch[107/300] Loss: 271.868 270.803 1.065 tensor(8699.7812) tensor(34.0918) 32
Epoch[108/300] Loss: 213.096 212.009 1.087 tensor(6819.0693) tensor(34.7930) 32
Epoch[109/300] Loss: 208.780 207.649 1.131 tensor(6680.9473) tensor(36.1790) 32
Epoch[110/300] Loss: 179.398 178.311 1.086 tensor(5740.7266) tensor(34.7658) 32
Epoch[111/300] Loss: 160.141 159.081 1.060 tensor(5124.5054) tensor(33.9051) 32
Epoch[112/300] Loss: 174.213 173.174 1.039 tensor(5574.8213) tensor(33.2624) 32
Epoch[113/300] Loss: 178.927 177.855 1.072 tensor(5725.6787) tensor(34.3102) 32
Epoch[114/300] Loss: 164.245 163.269 0.976 tensor(5255.8350) tensor(31.2204) 32
Epoch[115/300] Loss: 162.286 161.287 0.999 tensor(5193.1421) tensor(31.9603) 32
Epoch[116/300] Loss: 147.703 146.708 0.994 tensor(4726.4902) tensor(31.8189) 32
Epoch[117/300] Loss: 145.005 143.922 1.083 tensor(4640.1548) tensor(34.6589) 32
Epoch[118/300] Loss: 167.469 166.404 1.065 tensor(5359.0181) tensor(34.0779) 32
Epoch[119/300] Loss: 160.884 159.885 0.999 tensor(5148.2764) tensor(31.9628) 32
Epoch[120/300] Loss: 152.345 151.277 1.067 tensor(4875.0337) tensor(34.1563) 32
Epoch[121/300] Loss: 148.201 147.118 1.083 tensor(4742.4287) tensor(34.6626) 32
Epoch[122/300] Loss: 161.838 160.782 1.055 tensor(5178.8125) tensor(33.7745) 32
Epoch[123/300] Loss: 142.799 141.669 1.131 tensor(4569.5771) tensor(36.1785) 32
Epoch[124/300] Loss: 180.220 179.176 1.044 tensor(5767.0454) tensor(33.3995) 32
Epoch[125/300] Loss: 238.363 237.201 1.162 tensor(7627.6118) tensor(37.1887) 32
Epoch[126/300] Loss: 210.021 208.790 1.231 tensor(6720.6631) tensor(39.3869) 32
Epoch[127/300] Loss: 145.340 144.151 1.189 tensor(4650.8760) tensor(38.0528) 32
Epoch[128/300] Loss: 135.285 134.079 1.205 tensor(4329.1064) tensor(38.5664) 32
Epoch[129/300] Loss: 119.423 118.255 1.167 tensor(3821.5212) tensor(37.3578) 32
Epoch[130/300] Loss: 114.789 113.575 1.214 tensor(3673.2461) tensor(38.8366) 32
Epoch[131/300] Loss: 107.378 106.217 1.161 tensor(3436.1057) tensor(37.1673) 32
Epoch[132/300] Loss: 131.057 129.854 1.203 tensor(4193.8174) tensor(38.4931) 32
Epoch[133/300] Loss: 108.901 107.712 1.189 tensor(3484.8223) tensor(38.0459) 32
Epoch[134/300] Loss: 98.233 97.203 1.030 tensor(3143.4492) tensor(32.9469) 32
Epoch[135/300] Loss: 97.240 96.151 1.090 tensor(3111.6938) tensor(34.8642) 32
Epoch[136/300] Loss: 106.299 105.179 1.120 tensor(3401.5559) tensor(35.8289) 32
Epoch[137/300] Loss: 98.680 97.579 1.101 tensor(3157.7754) tensor(35.2454) 32
Epoch[138/300] Loss: 87.523 86.467 1.056 tensor(2800.7449) tensor(33.7930) 32
Epoch[139/300] Loss: 91.158 90.061 1.098 tensor(2917.0627) tensor(35.1201) 32
Epoch[140/300] Loss: 94.011 92.882 1.129 tensor(3008.3677) tensor(36.1375) 32
Epoch[141/300] Loss: 95.029 93.974 1.056 tensor(3040.9312) tensor(33.7791) 32
Epoch[142/300] Loss: 92.347 91.253 1.094 tensor(2955.1165) tensor(35.0095) 32
Epoch[143/300] Loss: 93.861 92.766 1.095 tensor(3003.5532) tensor(35.0433) 32
Epoch[144/300] Loss: 101.375 100.262 1.113 tensor(3244.0010) tensor(35.6151) 32
Epoch[145/300] Loss: 92.731 91.617 1.113 tensor(2967.3762) tensor(35.6256) 32
Epoch[146/300] Loss: 83.373 82.281 1.092 tensor(2667.9312) tensor(34.9326) 32
Epoch[147/300] Loss: 83.207 82.074 1.133 tensor(2662.6265) tensor(36.2701) 32
Epoch[148/300] Loss: 93.545 92.438 1.107 tensor(2993.4382) tensor(35.4150) 32
Epoch[149/300] Loss: 109.406 108.272 1.135 tensor(3501.0066) tensor(36.3050) 32
Epoch[150/300] Loss: 97.624 96.561 1.063 tensor(3123.9573) tensor(34.0089) 32
Epoch[151/300] Loss: 75.915 74.795 1.119 tensor(2429.2744) tensor(35.8192) 32
Epoch[152/300] Loss: 102.476 101.379 1.097 tensor(3279.2266) tensor(35.1087) 32
Epoch[153/300] Loss: 91.457 90.428 1.030 tensor(2926.6396) tensor(32.9503) 32
Epoch[154/300] Loss: 88.783 87.685 1.098 tensor(2841.0500) tensor(35.1262) 32
Epoch[155/300] Loss: 82.035 80.866 1.169 tensor(2625.1089) tensor(37.4080) 32
Epoch[156/300] Loss: 87.365 86.232 1.133 tensor(2795.6804) tensor(36.2628) 32
Epoch[157/300] Loss: 117.723 116.539 1.184 tensor(3767.1365) tensor(37.8946) 32
Epoch[158/300] Loss: 125.064 123.871 1.193 tensor(4002.0452) tensor(38.1669) 32
Epoch[159/300] Loss: 116.700 115.488 1.212 tensor(3734.4014) tensor(38.7737) 32
Epoch[160/300] Loss: 115.791 114.569 1.222 tensor(3705.3037) tensor(39.1105) 32
Epoch[161/300] Loss: 76.213 74.928 1.284 tensor(2438.8022) tensor(41.0902) 32
Epoch[162/300] Loss: 88.785 87.479 1.306 tensor(2841.1118) tensor(41.7811) 32
Epoch[163/300] Loss: 69.981 68.794 1.187 tensor(2239.3772) tensor(37.9829) 32
Epoch[164/300] Loss: 58.894 57.707 1.187 tensor(1884.6232) tensor(37.9887) 32
Epoch[165/300] Loss: 52.816 51.591 1.225 tensor(1690.1156) tensor(39.1979) 32
Epoch[166/300] Loss: 47.792 46.657 1.136 tensor(1529.3580) tensor(36.3485) 32
Epoch[167/300] Loss: 59.653 58.475 1.178 tensor(1908.8979) tensor(37.6968) 32
Epoch[168/300] Loss: 48.539 47.341 1.198 tensor(1553.2533) tensor(38.3395) 32
Epoch[169/300] Loss: 53.494 52.318 1.176 tensor(1711.8197) tensor(37.6438) 32
Epoch[170/300] Loss: 52.898 51.816 1.082 tensor(1692.7344) tensor(34.6271) 32
Epoch[171/300] Loss: 51.869 50.702 1.168 tensor(1659.8230) tensor(37.3625) 32
Epoch[172/300] Loss: 54.563 53.425 1.138 tensor(1746.0144) tensor(36.4068) 32
Epoch[173/300] Loss: 42.104 40.982 1.122 tensor(1347.3380) tensor(35.9189) 32
Epoch[174/300] Loss: 50.132 49.051 1.082 tensor(1604.2314) tensor(34.6098) 32
Epoch[175/300] Loss: 50.422 49.325 1.097 tensor(1613.4974) tensor(35.0980) 32
Epoch[176/300] Loss: 44.538 43.457 1.081 tensor(1425.2150) tensor(34.5788) 32
Epoch[177/300] Loss: 60.079 58.897 1.182 tensor(1922.5435) tensor(37.8267) 32
Epoch[178/300] Loss: 70.729 69.582 1.148 tensor(2263.3369) tensor(36.7262) 32
Epoch[179/300] Loss: 144.673 143.468 1.205 tensor(4629.5337) tensor(38.5496) 32
Epoch[180/300] Loss: 295.882 294.594 1.288 tensor(9468.2275) tensor(41.2298) 32
Epoch[181/300] Loss: 131.978 130.628 1.350 tensor(4223.3066) tensor(43.2119) 32
Epoch[182/300] Loss: 101.154 99.793 1.361 tensor(3236.9380) tensor(43.5643) 32
Epoch[183/300] Loss: 59.824 58.499 1.325 tensor(1914.3652) tensor(42.3889) 32
Epoch[184/300] Loss: 45.435 44.086 1.349 tensor(1453.9242) tensor(43.1617) 32
Epoch[185/300] Loss: 42.086 40.839 1.247 tensor(1346.7524) tensor(39.9024) 32
Epoch[186/300] Loss: 40.048 38.759 1.289 tensor(1281.5498) tensor(41.2581) 32
Epoch[187/300] Loss: 37.682 36.466 1.216 tensor(1205.8203) tensor(38.9225) 32
Epoch[188/300] Loss: 43.028 41.827 1.201 tensor(1376.8894) tensor(38.4372) 32
Epoch[189/300] Loss: 29.346 28.135 1.211 tensor(939.0620) tensor(38.7576) 32
Epoch[190/300] Loss: 42.490 41.173 1.317 tensor(1359.6957) tensor(42.1454) 32
Epoch[191/300] Loss: 34.331 33.072 1.258 tensor(1098.5859) tensor(40.2708) 32
Epoch[192/300] Loss: 27.675 26.488 1.188 tensor(885.6102) tensor(38.0060) 32
Epoch[193/300] Loss: 32.541 31.297 1.244 tensor(1041.3135) tensor(39.8094) 32
Epoch[194/300] Loss: 29.730 28.583 1.147 tensor(951.3732) tensor(36.7031) 32
Epoch[195/300] Loss: 35.251 34.085 1.166 tensor(1128.0433) tensor(37.3223) 32
Epoch[196/300] Loss: 30.579 29.435 1.145 tensor(978.5432) tensor(36.6261) 32
Epoch[197/300] Loss: 36.648 35.522 1.125 tensor(1172.7300) tensor(36.0117) 32
Epoch[198/300] Loss: 35.188 34.086 1.102 tensor(1126.0215) tensor(35.2664) 32
Epoch[199/300] Loss: 35.698 34.595 1.103 tensor(1142.3503) tensor(35.3019) 32
Epoch[200/300] Loss: 45.770 44.661 1.109 tensor(1464.6525) tensor(35.4882) 32
Epoch[201/300] Loss: 34.976 33.831 1.145 tensor(1119.2295) tensor(36.6383) 32
Epoch[202/300] Loss: 48.125 46.949 1.176 tensor(1539.9940) tensor(37.6364) 32
Epoch[203/300] Loss: 59.686 58.483 1.204 tensor(1909.9645) tensor(38.5230) 32
Epoch[204/300] Loss: 50.084 48.963 1.121 tensor(1602.6741) tensor(35.8613) 32
Epoch[205/300] Loss: 48.456 47.237 1.219 tensor(1550.5813) tensor(39.0000) 32
Epoch[206/300] Loss: 35.990 34.837 1.153 tensor(1151.6954) tensor(36.9033) 32
Epoch[207/300] Loss: 37.171 36.043 1.128 tensor(1189.4731) tensor(36.1015) 32
Epoch[208/300] Loss: 49.079 47.956 1.123 tensor(1570.5197) tensor(35.9386) 32
Epoch[209/300] Loss: 32.913 31.779 1.135 tensor(1053.2208) tensor(36.3071) 32
Epoch[210/300] Loss: 24.262 23.153 1.110 tensor(776.3965) tensor(35.5111) 32
Epoch[211/300] Loss: 27.115 26.032 1.083 tensor(867.6855) tensor(34.6557) 32
Epoch[212/300] Loss: 31.089 30.015 1.074 tensor(994.8437) tensor(34.3555) 32
Epoch[213/300] Loss: 26.187 25.053 1.135 tensor(837.9998) tensor(36.3118) 32
Epoch[214/300] Loss: 22.625 21.584 1.041 tensor(724.0060) tensor(33.3119) 32
Epoch[215/300] Loss: 27.665 26.584 1.081 tensor(885.2891) tensor(34.5854) 32
Epoch[216/300] Loss: 38.133 37.068 1.065 tensor(1220.2681) tensor(34.0922) 32
Epoch[217/300] Loss: 40.848 39.792 1.056 tensor(1307.1379) tensor(33.7895) 32
Epoch[218/300] Loss: 114.781 113.629 1.152 tensor(3672.9802) tensor(36.8642) 32
Epoch[219/300] Loss: 324.837 323.588 1.250 tensor(10394.7920) tensor(39.9844) 32
```