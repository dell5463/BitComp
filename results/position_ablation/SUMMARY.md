**position_ablation**, reference variant `bit_only`, 200 val images (development)

| variant | params | train images | steps | val BCE (nats/bit) | val acc | val ECE |
|---|---:|---:|---:|---:|---:|---:|
| bit_only | 161505 | 1024 | 64000 | 0.4846 | 0.7281 | 0.0115 |
| bit_plane | 164641 | 1024 | 64000 | 0.4792 | 0.7301 | 0.0072 |
| bit_plane_rowcol | 171297 | 1024 | 64000 | 0.4654 | 0.7336 | 0.0039 |

Operational RD difference vs reference (mean PSNR gain over the overlapping complete-file bpp range):

| variant | raw payload: mean dB [min, max] | bpp range | range-coded payload: mean dB [min, max] |
|---|---:|---:|---:|
| bit_plane | -1.23 [-7.46, +8.37] | 0.27-8.27 | -2.06 [-7.46, +1.72] |
| bit_plane_rowcol | +0.53 [-7.45, +10.55] | 0.27-8.27 | -1.62 [-7.45, +5.77] |

Per-threshold complete-file bpp / pooled PSNR (dB):

| t | bit_only | bit_plane | bit_plane_rowcol |
|---:|---:|---:|---:|
| 0.5 | 0.266 / 12.45 | 0.266 / 5.00 | 0.266 / 5.01 |
| 0.55 | 0.275 / 12.06 | 0.278 / 9.72 | 0.315 / 7.16 |
| 0.6 | 0.280 / 9.72 | 0.284 / 9.04 | 0.426 / 9.15 |
| 0.65 | 0.284 / 9.76 | 0.314 / 8.96 | 0.573 / 9.80 |
| 0.7 | 0.294 / 9.73 | 0.596 / 9.44 | 0.781 / 10.03 |
| 0.75 | 0.328 / 9.90 | 1.143 / 9.40 | 1.347 / 10.33 |
| 0.8 | 0.775 / 9.67 | 2.583 / 10.08 | 3.328 / 11.32 |
| 0.85 | 2.977 / 9.63 | 4.416 / 11.43 | 4.938 / 13.66 |
| 0.9 | 5.065 / 12.66 | 5.500 / 13.69 | 5.756 / 16.97 |
| 0.925 | 5.678 / 14.14 | 5.823 / 14.73 | 6.069 / 18.85 |
| 0.95 | 6.272 / 16.29 | 6.414 / 18.01 | 6.405 / 22.06 |
| 0.975 | 7.212 / 25.05 | 7.058 / 24.66 | 6.813 / 26.84 |
| 0.99 | 7.928 / 38.54 | 7.586 / 31.53 | 7.220 / 33.41 |
| 1 | 8.266 / inf | 8.266 / inf | 8.266 / inf |

Rate at quality: complete-file bpp needed for pooled PSNR >= target (Pareto front of the threshold points, interpolated linearly in (bpp, MSE); lossless point included). Paired image bootstrap (1000 resamples, seed 42) of the bpp difference vs `bit_only`; negative = fewer bits. 200 val images (development).

| payload | target dB | bit_only bpp | bit_plane bpp | bit_plane_rowcol bpp | bit_plane - bit_only [95% CI] | bit_plane_rowcol - bit_only [95% CI] |
|---|---:|---:|---:|---:|---:|---:|
| raw | 20 | 6.894 | 6.716 | 6.219 | -0.178 [-0.275, -0.096] | -0.676 [-0.750, -0.550] |
| raw | 25 | 7.210 | 7.108 | 6.706 | -0.102 [-0.276, -0.039] | -0.504 [-0.680, -0.459] |
| raw | 30 | 7.721 | 7.529 | 7.083 | -0.193 [-0.267, -0.087] | -0.638 [-0.716, -0.537] |
| raw | 35 | 7.886 | 7.960 | 7.541 | +0.074 [-0.075, +0.167] | -0.345 [-0.487, -0.233] |
| range-coded | 20 | 5.802 | 5.788 | 5.665 | -0.015 [-0.035, +0.011] | -0.137 [-0.167, -0.110] |
| range-coded | 25 | 5.920 | 5.890 | 5.729 | -0.031 [-0.047, -0.013] | -0.191 [-0.214, -0.169] |
| range-coded | 30 | 5.949 | 5.898 | 5.740 | -0.051 [-0.061, -0.043] | -0.209 [-0.224, -0.195] |
| range-coded | 35 | 5.958 | 5.900 | 5.743 | -0.057 [-0.065, -0.051] | -0.215 [-0.228, -0.202] |

Paired image bootstrap (1000 resamples, seed 42) of the operational-envelope mean PSNR gain vs `bit_only` over the overlapping complete-file bpp range; 200 val images (development). Lossless grid points excluded (as in `ablate`).

| variant | payload | observed gain dB | bootstrap mean | 95% CI | P(gain > 0) |
|---|---|---:|---:|---:|---:|
| bit_plane | raw | -1.23 | -1.22 | [-1.56, -0.84] | 0.000 |
| bit_plane | range-coded | -2.06 | -2.06 | [-2.43, -1.66] | 0.000 |
| bit_plane_rowcol | raw | 0.53 | +0.51 | [+0.12, +0.85] | 0.993 |
| bit_plane_rowcol | range-coded | -1.62 | -1.60 | [-1.97, -1.25] | 0.000 |

Paired image bootstrap (1000 resamples, seed 42) of the operational-envelope mean PSNR gain vs `bit_plane_rowcol` over the overlapping complete-file bpp range; 200 val images (development). Lossless grid points excluded (as in `ablate`).

| variant | payload | observed gain dB | bootstrap mean | 95% CI | P(gain > 0) |
|---|---|---:|---:|---:|---:|
| bit_only | raw | -0.53 | -0.51 | [-0.85, -0.12] | 0.007 |
| bit_only | range-coded | 1.62 | +1.60 | [+1.25, +1.97] | 1.000 |
| bit_plane | raw | -1.77 | -1.72 | [-2.06, -1.40] | 0.000 |
| bit_plane | range-coded | -0.56 | -0.56 | [-0.79, -0.31] | 0.000 |

Matched complete-file bpp: each variant's pooled PSNR (dB) / MSE / SSIM linearly interpolated between its own threshold points at the reference (`bit_only`) bpp for thresholds 0.6-0.85; 200 val images (development). n/a = outside the variant's measured bpp range.

| ref t | bpp | bit_only PSNR | bit_plane PSNR | bit_plane_rowcol PSNR | bit_plane dPSNR | bit_plane_rowcol dPSNR |
|---:|---:|---:|---:|---:|---:|---:|
| 0.6 | 0.280 | 9.72 | 9.43 | 5.65 | -0.29 | -4.08 |
| 0.65 | 0.284 | 9.76 | 9.04 | 5.82 | -0.73 | -3.94 |
| 0.7 | 0.294 | 9.73 | 9.01 | 6.23 | -0.72 | -3.50 |
| 0.75 | 0.328 | 9.90 | 8.99 | 7.39 | -0.92 | -2.51 |
| 0.8 | 0.775 | 9.67 | 9.43 | 10.02 | -0.24 | +0.35 |
| 0.85 | 2.977 | 9.63 | 10.37 | 11.14 | +0.74 | +1.52 |

| variant | mean dPSNR over matched points (dB) | points |
|---|---:|---:|
| bit_plane | -0.360 | 6 |
| bit_plane_rowcol | -2.028 | 6 |

MSE and SSIM at the same matched bpp:

| ref t | bpp | bit_only MSE / SSIM | bit_plane MSE / SSIM | bit_plane_rowcol MSE / SSIM |
|---:|---:|---:|---:|---:|
| 0.6 | 0.280 | 6929.90 / 0.1300 | 7435.83 / 0.1329 | 18138.34 / 0.1192 |
| 0.65 | 0.284 | 6865.67 / 0.1310 | 8113.81 / 0.1273 | 17491.13 / 0.1199 |
| 0.7 | 0.294 | 6911.86 / 0.1301 | 8160.37 / 0.1268 | 15968.28 / 0.1217 |
| 0.75 | 0.328 | 6647.96 / 0.1323 | 8214.28 / 0.1258 | 11969.26 / 0.1259 |
| 0.8 | 0.775 | 7014.32 / 0.1331 | 7414.49 / 0.1230 | 6473.08 / 0.1296 |
| 0.85 | 2.977 | 7084.31 / 0.1931 | 6021.13 / 0.1630 | 5017.16 / 0.1645 |
