**objectives**, reference variant `bce`, 200 val images (development)

| variant | params | train images | steps | val BCE (nats/bit) | val acc | val ECE |
|---|---:|---:|---:|---:|---:|---:|
| bce | 171297 | 1024 | 64000 | 0.4654 | 0.7336 | 0.0039 |
| weighted_sqrt | 171297 | 1024 | 64000 | 0.4681 | 0.7330 | 0.0062 |

Operational RD difference vs reference (mean PSNR gain over the overlapping complete-file bpp range):

| variant | raw payload: mean dB [min, max] | bpp range | range-coded payload: mean dB [min, max] |
|---|---:|---:|---:|
| weighted_sqrt | +0.39 [-2.36, +6.96] | 0.27-8.27 | +1.23 [-2.36, +6.96] |

Per-threshold complete-file bpp / pooled PSNR (dB):

| t | bce | weighted_sqrt |
|---:|---:|---:|
| 0.5 | 0.266 / 5.01 | 0.266 / 11.97 |
| 0.55 | 0.315 / 7.16 | 0.449 / 11.76 |
| 0.6 | 0.426 / 9.15 | 0.568 / 10.30 |
| 0.65 | 0.573 / 9.80 | 0.689 / 9.39 |
| 0.7 | 0.781 / 10.03 | 0.967 / 9.62 |
| 0.75 | 1.347 / 10.33 | 1.420 / 9.57 |
| 0.8 | 3.328 / 11.32 | 2.629 / 9.89 |
| 0.85 | 4.938 / 13.66 | 4.540 / 11.83 |
| 0.9 | 5.756 / 16.97 | 5.502 / 14.61 |
| 0.925 | 6.069 / 18.85 | 5.946 / 17.39 |
| 0.95 | 6.405 / 22.06 | 6.338 / 20.48 |
| 0.975 | 6.813 / 26.84 | 6.769 / 25.23 |
| 0.99 | 7.220 / 33.41 | 7.179 / 31.69 |
| 1 | 8.266 / inf | 8.266 / inf |

Rate at quality: complete-file bpp needed for pooled PSNR >= target (Pareto front of the threshold points, interpolated linearly in (bpp, MSE); lossless point included). Paired image bootstrap (1000 resamples, seed 42) of the bpp difference vs `bce`; negative = fewer bits. 200 val images (development).

| payload | target dB | bce bpp | weighted_sqrt bpp | weighted_sqrt - bce [95% CI] |
|---|---:|---:|---:|---:|
| raw | 20 | 6.219 | 6.294 | +0.076 [+0.039, +0.113] |
| raw | 25 | 6.706 | 6.758 | +0.051 [+0.020, +0.098] |
| raw | 30 | 7.083 | 7.122 | +0.039 [+0.008, +0.077] |
| raw | 35 | 7.541 | 7.759 | +0.218 [+0.086, +0.365] |
| range-coded | 20 | 5.665 | 5.689 | +0.024 [+0.004, +0.044] |
| range-coded | 25 | 5.729 | 5.767 | +0.038 [+0.015, +0.046] |
| range-coded | 30 | 5.740 | 5.771 | +0.030 [+0.020, +0.036] |
| range-coded | 35 | 5.743 | 5.771 | +0.029 [+0.021, +0.035] |

Paired image bootstrap (1000 resamples, seed 42) of the operational-envelope mean PSNR gain vs `bce` over the overlapping complete-file bpp range; 200 val images (development). Lossless grid points excluded (as in `ablate`).

| variant | payload | observed gain dB | bootstrap mean | 95% CI | P(gain > 0) |
|---|---|---:|---:|---:|---:|
| weighted_sqrt | raw | 0.39 | +0.42 | [+0.12, +0.73] | 0.998 |
| weighted_sqrt | range-coded | 1.23 | +1.22 | [+0.84, +1.60] | 1.000 |

Matched complete-file bpp: each variant's pooled PSNR (dB) / MSE / SSIM linearly interpolated between its own threshold points at the reference (`bce`) bpp for thresholds 0.6-0.85; 200 val images (development). n/a = outside the variant's measured bpp range.

| ref t | bpp | bce PSNR | weighted_sqrt PSNR | weighted_sqrt dPSNR |
|---:|---:|---:|---:|---:|
| 0.6 | 0.426 | 9.15 | 11.79 | +2.63 |
| 0.65 | 0.573 | 9.80 | 10.26 | +0.46 |
| 0.7 | 0.781 | 10.03 | 9.47 | -0.56 |
| 0.75 | 1.347 | 10.33 | 9.58 | -0.75 |
| 0.8 | 3.328 | 11.32 | 10.60 | -0.72 |
| 0.85 | 4.938 | 13.66 | 12.98 | -0.68 |

| variant | mean dPSNR over matched points (dB) | points |
|---|---:|---:|
| weighted_sqrt | 0.065 | 6 |

MSE and SSIM at the same matched bpp:

| ref t | bpp | bce MSE / SSIM | weighted_sqrt MSE / SSIM |
|---:|---:|---:|---:|
| 0.6 | 0.426 | 7900.81 / 0.1266 | 4311.22 / 0.1266 |
| 0.65 | 0.573 | 6809.63 / 0.1301 | 6129.83 / 0.1236 |
| 0.7 | 0.781 | 6462.79 / 0.1296 | 7354.46 / 0.1235 |
| 0.75 | 1.347 | 6029.18 / 0.1302 | 7163.53 / 0.1229 |
| 0.8 | 3.328 | 4799.09 / 0.1719 | 5789.08 / 0.1980 |
| 0.85 | 4.938 | 2802.33 / 0.3518 | 3431.95 / 0.3911 |
