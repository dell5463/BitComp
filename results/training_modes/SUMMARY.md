**training_modes**, reference variant `teacher_forcing`, 200 val images (development)

| variant | params | train images | steps | val BCE (nats/bit) | val acc | val ECE |
|---|---:|---:|---:|---:|---:|---:|
| teacher_forcing | 171297 | 1024 | 64000 | 0.4654 | 0.7336 | 0.0039 |
| scheduled | 171297 | 1024 | 64000 | 0.4746 | 0.7310 | 0.0110 |
| rollout | 171297 | 1024 | 64000 | 0.4828 | 0.7304 | 0.0178 |

Operational RD difference vs reference (mean PSNR gain over the overlapping complete-file bpp range):

| variant | raw payload: mean dB [min, max] | bpp range | range-coded payload: mean dB [min, max] |
|---|---:|---:|---:|
| scheduled | +1.64 [-7.55, +7.05] | 0.27-8.27 | +1.86 [-3.54, +7.05] |
| rollout | -0.10 [-8.10, +5.48] | 0.27-8.27 | +0.27 [-4.16, +2.59] |

Per-threshold complete-file bpp / pooled PSNR (dB):

| t | teacher_forcing | scheduled | rollout |
|---:|---:|---:|---:|
| 0.5 | 0.266 / 5.01 | 0.266 / 12.06 | 0.266 / 5.00 |
| 0.55 | 0.315 / 7.16 | 0.334 / 12.63 | 0.457 / 9.29 |
| 0.6 | 0.426 / 9.15 | 0.394 / 11.99 | 0.866 / 8.41 |
| 0.65 | 0.573 / 9.80 | 0.447 / 9.90 | 2.252 / 10.55 |
| 0.7 | 0.781 / 10.03 | 0.567 / 9.96 | 3.398 / 12.28 |
| 0.75 | 1.347 / 10.33 | 1.223 / 10.30 | 4.580 / 13.91 |
| 0.8 | 3.328 / 11.32 | 3.065 / 11.19 | 5.321 / 16.18 |
| 0.85 | 4.938 / 13.66 | 5.135 / 14.43 | 5.908 / 19.23 |
| 0.9 | 5.756 / 16.97 | 5.987 / 18.52 | 6.507 / 22.90 |
| 0.925 | 6.069 / 18.85 | 6.354 / 21.92 | 6.896 / 25.31 |
| 0.95 | 6.405 / 22.06 | 6.735 / 25.86 | 7.383 / 29.35 |
| 0.975 | 6.813 / 26.84 | 7.224 / 32.42 | 7.936 / 38.89 |
| 0.99 | 7.220 / 33.41 | 7.681 / 40.23 | 8.237 / 62.11 |
| 1 | 8.266 / inf | 8.266 / inf | 8.266 / inf |

Rate at quality: complete-file bpp needed for pooled PSNR >= target (Pareto front of the threshold points, interpolated linearly in (bpp, MSE); lossless point included). Paired image bootstrap (1000 resamples, seed 42) of the bpp difference vs `teacher_forcing`; negative = fewer bits. 200 val images (development).

| payload | target dB | teacher_forcing bpp | scheduled bpp | rollout bpp | scheduled - teacher_forcing [95% CI] | rollout - teacher_forcing [95% CI] |
|---|---:|---:|---:|---:|---:|---:|
| raw | 20 | 6.219 | 6.183 | 6.079 | -0.036 [-0.068, -0.003] | -0.139 [-0.200, -0.079] |
| raw | 25 | 6.706 | 6.679 | 6.857 | -0.028 [-0.058, +0.003] | +0.151 [+0.102, +0.207] |
| raw | 30 | 7.083 | 7.121 | 7.470 | +0.037 [+0.009, +0.068] | +0.387 [+0.310, +0.461] |
| raw | 35 | 7.541 | 7.470 | 7.836 | -0.071 [-0.175, +0.051] | +0.295 [+0.180, +0.437] |
| range-coded | 20 | 5.665 | 5.673 | 5.628 | +0.008 [-0.007, +0.023] | -0.037 [-0.062, -0.012] |
| range-coded | 25 | 5.729 | 5.787 | 5.837 | +0.057 [+0.047, +0.068] | +0.108 [+0.097, +0.120] |
| range-coded | 30 | 5.740 | 5.826 | 5.904 | +0.086 [+0.077, +0.093] | +0.164 [+0.154, +0.173] |
| range-coded | 35 | 5.743 | 5.840 | 5.926 | +0.098 [+0.090, +0.105] | +0.183 [+0.174, +0.193] |

Paired image bootstrap (1000 resamples, seed 42) of the operational-envelope mean PSNR gain vs `teacher_forcing` over the overlapping complete-file bpp range; 200 val images (development). Lossless grid points excluded (as in `ablate`).

| variant | payload | observed gain dB | bootstrap mean | 95% CI | P(gain > 0) |
|---|---|---:|---:|---:|---:|
| scheduled | raw | 1.64 | +1.68 | [+1.42, +1.91] | 1.000 |
| scheduled | range-coded | 1.86 | +1.86 | [+1.57, +2.14] | 1.000 |
| rollout | raw | -0.10 | -0.09 | [-0.35, +0.15] | 0.252 |
| rollout | range-coded | 0.27 | +0.23 | [-0.04, +0.51] | 0.950 |

Matched complete-file bpp: each variant's pooled PSNR (dB) / MSE / SSIM linearly interpolated between its own threshold points at the reference (`teacher_forcing`) bpp for thresholds 0.6-0.85; 200 val images (development). n/a = outside the variant's measured bpp range.

| ref t | bpp | teacher_forcing PSNR | scheduled PSNR | rollout PSNR | scheduled dPSNR | rollout dPSNR |
|---:|---:|---:|---:|---:|---:|---:|
| 0.6 | 0.426 | 9.15 | 10.74 | 8.58 | +1.58 | -0.58 |
| 0.65 | 0.573 | 9.80 | 9.96 | 9.04 | +0.16 | -0.76 |
| 0.7 | 0.781 | 10.03 | 10.07 | 8.59 | +0.04 | -1.44 |
| 0.75 | 1.347 | 10.33 | 10.36 | 9.15 | +0.03 | -1.18 |
| 0.8 | 3.328 | 11.32 | 11.60 | 12.18 | +0.28 | +0.86 |
| 0.85 | 4.938 | 13.66 | 14.12 | 15.00 | +0.46 | +1.35 |

| variant | mean dPSNR over matched points (dB) | points |
|---|---:|---:|
| scheduled | 0.426 | 6 |
| rollout | -0.291 | 6 |

MSE and SSIM at the same matched bpp:

| ref t | bpp | teacher_forcing MSE / SSIM | scheduled MSE / SSIM | rollout MSE / SSIM |
|---:|---:|---:|---:|---:|
| 0.6 | 0.426 | 7900.81 / 0.1266 | 5638.85 / 0.1305 | 9797.02 / 0.1212 |
| 0.65 | 0.573 | 6809.63 / 0.1301 | 6559.36 / 0.1233 | 8147.16 / 0.1193 |
| 0.7 | 0.781 | 6462.79 / 0.1296 | 6403.15 / 0.1230 | 9028.63 / 0.1141 |
| 0.75 | 1.347 | 6029.18 / 0.1302 | 5995.82 / 0.1248 | 8118.27 / 0.1159 |
| 0.8 | 3.328 | 4799.09 / 0.1719 | 4614.20 / 0.1888 | 3959.25 / 0.2022 |
| 0.85 | 4.938 | 2802.33 / 0.3518 | 2595.29 / 0.3675 | 2123.81 / 0.4222 |
