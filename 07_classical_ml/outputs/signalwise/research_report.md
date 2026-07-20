# WESAD Stress-Detection Experimental Report

## Reproducibility protocol

- Subjects: S2, S3, S4
- Labels: {'1': 0, '2': 1, '3': 0} (1/3 = non-stress; 2 = stress).
- Windows: 5 s, 50% overlap, aligned to 4 Hz.
- Constant evaluation pipeline: median imputation -> StandardScaler -> balanced LogisticRegression(max_iter=3000, random_state=42).
- Subject-independent split: GroupShuffleSplit(test_size=0.20, random_state=42).

Only preprocessing changes between Raw, Filtered, and Filtered + Normalized.  Therefore each row is directly comparable within a signal.

## Pipeline stage record

The accompanying `stage_record.csv` records input shapes, window counts, feature counts, and elapsed time for every subject/signal/variant.

## All-signal comparison

| variant | accuracy | precision | recall | f1 | roc_auc | training_seconds | inference_ms_per_window |
| --- | --- | --- | --- | --- | --- | --- | --- |
| filtered | 0.5618 | 0.0227 | 0.0122 | 0.0159 | 0.7401 | 0.2458 | 0.0620 |
| filtered_normalized | 0.5866 | 0.1329 | 0.0772 | 0.0977 | 0.7147 | 0.2226 | 0.0531 |
| raw | 0.5418 | 0.0446 | 0.0285 | 0.0347 | 0.7336 | 0.2769 | 0.0588 |

## Best preprocessing variant per experiment

| experiment | variant | accuracy | f1 | roc_auc | n_features |
| --- | --- | --- | --- | --- | --- |
| E4_EDA | filtered_normalized | 0.8681 | 0.8133 | 0.9342 | 18 |
| RB_TEMP | filtered_normalized | 0.7750 | 0.7204 | 0.9682 | 18 |
| RB_RESP | raw | 0.8163 | 0.7023 | 0.8503 | 18 |
| E4_BVP | raw | 0.6384 | 0.5356 | 0.6675 | 18 |
| RB_ECG | filtered_normalized | 0.4947 | 0.5301 | 0.8196 | 18 |
| RB_EDA | filtered_normalized | 0.7197 | 0.5182 | 0.7255 | 18 |
| RB_EMG | filtered_normalized | 0.4629 | 0.4610 | 0.6041 | 18 |
| E4_TEMP | filtered_normalized | 0.6431 | 0.3674 | 0.5311 | 18 |
| E4_IBI | filtered_normalized | 0.5771 | 0.3109 | 0.5450 | 18 |
| RB_ACC | raw | 0.6172 | 0.2697 | 0.7826 | 54 |
| E4_HR | filtered | 0.5183 | 0.2058 | 0.3585 | 18 |
| ALL_SIGNALS | filtered_normalized | 0.5866 | 0.0977 | 0.7147 | 288 |
| E4_ACC | filtered | 0.4547 | 0.0000 | 0.0217 | 54 |

## Feature extraction and physiological interpretation

The ML experiments extract exactly these 18 generic features from each available signal channel: mean, median, std, variance, min, max, range, iqr, skewness, kurtosis, rms, energy, entropy, mav, auc, slope, zero_crossing_rate, cv.  These are window-level statistical/time-domain features; their value units inherit the source signal except energy (unit²), variance (unit²), entropy (bits), slope (unit/sample), and zero-crossing rate (unitless).

### E4_ACC: Empatica E4 accelerometer

Movement and motion artefacts; stress can alter activity/restlessness.

Expected ranges are participant-, device-, and context-dependent; use the raw min/max below as the observed study range rather than applying a universal clinical threshold.

| feature | description | value_mean | value_min | value_max | unit | interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| auc | sample-domain trapezoidal area | -31.5170 | -1215.6273 | 1199.9534 | g (source scale is 1/64 g) | Finite values; inspect class separation and preprocessing change. |
| cv | standard deviation divided by mean | 0.4871 | 0.0000 | 337.1413 | g (source scale is 1/64 g) | Finite values; inspect class separation and preprocessing change. |
| energy | sum of squared samples | 27390.0492 | 0.0000 | 81873.3593 | g (source scale is 1/64 g) | Finite values; inspect class separation and preprocessing change. |
| entropy | 10-bin Shannon distribution entropy (bits) | 2.4534 | 0.0000 | 3.2842 | g (source scale is 1/64 g) | Finite values; inspect class separation and preprocessing change. |
| iqr | middle-50% spread | 3.3610 | 0.0000 | 91.9493 | g (source scale is 1/64 g) | Finite values; inspect class separation and preprocessing change. |
| kurtosis | tail/peakedness relative to normal | 1.0043 | -2.0000 | 15.0526 | g (source scale is 1/64 g) | Finite values; inspect class separation and preprocessing change. |
| mav | mean absolute value | 31.1081 | 0.0000 | 63.9817 | g (source scale is 1/64 g) | Finite values; inspect class separation and preprocessing change. |
| max | highest value | 2.8447 | -63.5510 | 85.3000 | g (source scale is 1/64 g) | Finite values; inspect class separation and preprocessing change. |
| mean | average level | -1.6605 | -63.9817 | 63.1509 | g (source scale is 1/64 g) | Finite values; inspect class separation and preprocessing change. |
| median | middle value | -1.5665 | -64.3963 | 63.1512 | g (source scale is 1/64 g) | Finite values; inspect class separation and preprocessing change. |
| min | lowest value | -6.6518 | -88.0451 | 63.0000 | g (source scale is 1/64 g) | Finite values; inspect class separation and preprocessing change. |
| range | max minus min | 9.4965 | 0.0000 | 120.4582 | g (source scale is 1/64 g) | Finite values; inspect class separation and preprocessing change. |
| rate |  | 0.0195 | 0.0000 | 0.9474 | g (source scale is 1/64 g) | Finite values; inspect class separation and preprocessing change. |
| rms | root-mean-square magnitude | 31.5330 | 0.0000 | 63.9818 | g (source scale is 1/64 g) | Finite values; inspect class separation and preprocessing change. |
| skewness | asymmetry of value distribution | -0.0527 | -4.1295 | 4.1295 | g (source scale is 1/64 g) | Finite values; inspect class separation and preprocessing change. |
| slope | linear sample-to-sample trend | -0.0006 | -6.6023 | 6.6466 | g (source scale is 1/64 g) | Finite values; inspect class separation and preprocessing change. |
| std | within-window variability | 2.6879 | 0.0000 | 42.3686 | g (source scale is 1/64 g) | Finite values; inspect class separation and preprocessing change. |
| variance | squared variability | 35.2453 | 0.0000 | 1795.1014 | g (source scale is 1/64 g) | Finite values; inspect class separation and preprocessing change. |

Conclusion: features are numerically complete in this run; model comparison below indicates their discriminative value.

### E4_BVP: Empatica E4 blood-volume pulse

Peripheral pulse morphology reflects autonomic cardiovascular activity.

Expected ranges are participant-, device-, and context-dependent; use the raw min/max below as the observed study range rather than applying a universal clinical threshold.

| feature | description | value_mean | value_min | value_max | unit | interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| auc | sample-domain trapezoidal area | -0.8217 | -869.3482 | 661.6658 | device units | Finite values; inspect class separation and preprocessing change. |
| cv | standard deviation divided by mean | 351.4996 | 1.5037 | 541351.4707 | device units | Finite values; inspect class separation and preprocessing change. |
| energy | sum of squared samples | 53874.6884 | 28.5540 | 1592546.5879 | device units | Finite values; inspect class separation and preprocessing change. |
| entropy | 10-bin Shannon distribution entropy (bits) | 2.8356 | 1.1540 | 3.2842 | device units | Finite values; inspect class separation and preprocessing change. |
| iqr | middle-50% spread | 47.5111 | 1.5779 | 310.0681 | device units | Finite values; inspect class separation and preprocessing change. |
| kurtosis | tail/peakedness relative to normal | -0.1695 | -1.8148 | 11.2316 | device units | Finite values; inspect class separation and preprocessing change. |
| mav | mean absolute value | 30.1513 | 1.0038 | 171.7839 | device units | Finite values; inspect class separation and preprocessing change. |
| max | highest value | 72.6670 | 1.7392 | 646.5089 | device units | Finite values; inspect class separation and preprocessing change. |
| mean | average level | 0.0032 | -41.7420 | 32.1718 | device units | Finite values; inspect class separation and preprocessing change. |
| median | middle value | 2.0272 | -52.6439 | 64.8239 | device units | Finite values; inspect class separation and preprocessing change. |
| min | lowest value | -78.5521 | -821.3371 | -2.0954 | device units | Finite values; inspect class separation and preprocessing change. |
| range | max minus min | 151.2190 | 4.0509 | 1446.2346 | device units | Finite values; inspect class separation and preprocessing change. |
| rate |  | 0.4989 | 0.2105 | 0.8947 | device units | Finite values; inspect class separation and preprocessing change. |
| rms | root-mean-square magnitude | 38.2726 | 1.1949 | 282.1831 | device units | Finite values; inspect class separation and preprocessing change. |
| skewness | asymmetry of value distribution | -0.1460 | -3.2947 | 2.6174 | device units | Finite values; inspect class separation and preprocessing change. |
| slope | linear sample-to-sample trend | -0.0120 | -8.9885 | 12.3106 | device units | Finite values; inspect class separation and preprocessing change. |
| std | within-window variability | 38.1078 | 1.1004 | 282.1789 | device units | Finite values; inspect class separation and preprocessing change. |
| variance | squared variability | 2672.4674 | 1.2109 | 79624.9213 | device units | Finite values; inspect class separation and preprocessing change. |

Conclusion: features are numerically complete in this run; model comparison below indicates their discriminative value.

### E4_EDA: Empatica E4 electrodermal activity

Sympathetic sweat-gland activity; a widely used stress marker.

Expected ranges are participant-, device-, and context-dependent; use the raw min/max below as the observed study range rather than applying a universal clinical threshold.

| feature | description | value_mean | value_min | value_max | unit | interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| auc | sample-domain trapezoidal area | 19.2715 | 1.4632 | 128.0427 | µS | Finite values; inspect class separation and preprocessing change. |
| cv | standard deviation divided by mean | 0.0174 | 0.0006 | 0.9698 | µS | Finite values; inspect class separation and preprocessing change. |
| energy | sum of squared samples | 52.2797 | 0.1185 | 912.6318 | µS | Finite values; inspect class separation and preprocessing change. |
| entropy | 10-bin Shannon distribution entropy (bits) | 2.3583 | 0.7476 | 3.3219 | µS | Finite values; inspect class separation and preprocessing change. |
| iqr | middle-50% spread | 0.0318 | 0.0000 | 1.2484 | µS | Finite values; inspect class separation and preprocessing change. |
| kurtosis | tail/peakedness relative to normal | 0.1009 | -1.9596 | 11.7230 | µS | Finite values; inspect class separation and preprocessing change. |
| mav | mean absolute value | 1.0143 | 0.0770 | 6.7457 | µS | Finite values; inspect class separation and preprocessing change. |
| max | highest value | 1.0504 | 0.0797 | 7.1860 | µS | Finite values; inspect class separation and preprocessing change. |
| mean | average level | 1.0143 | 0.0770 | 6.7457 | µS | Finite values; inspect class separation and preprocessing change. |
| median | middle value | 1.0118 | 0.0759 | 6.9262 | µS | Finite values; inspect class separation and preprocessing change. |
| min | lowest value | 0.9837 | 0.0746 | 6.2218 | µS | Finite values; inspect class separation and preprocessing change. |
| range | max minus min | 0.0667 | 0.0013 | 1.4981 | µS | Finite values; inspect class separation and preprocessing change. |
| rate |  | 0.0000 | 0.0000 | 0.0000 | µS | Finite values; inspect class separation and preprocessing change. |
| rms | root-mean-square magnitude | 1.0148 | 0.0770 | 6.7551 | µS | Finite values; inspect class separation and preprocessing change. |
| skewness | asymmetry of value distribution | -0.1227 | -3.4287 | 3.5109 | µS | Finite values; inspect class separation and preprocessing change. |
| slope | linear sample-to-sample trend | -0.0002 | -0.0622 | 0.0935 | µS | Finite values; inspect class separation and preprocessing change. |
| std | within-window variability | 0.0207 | 0.0005 | 0.5635 | µS | Finite values; inspect class separation and preprocessing change. |
| variance | squared variability | 0.0027 | 0.0000 | 0.3175 | µS | Finite values; inspect class separation and preprocessing change. |

Conclusion: features are numerically complete in this run; model comparison below indicates their discriminative value.

### E4_HR: Empatica E4 heart rate

Cardiac rate commonly rises with sympathetic arousal.

Expected ranges are participant-, device-, and context-dependent; use the raw min/max below as the observed study range rather than applying a universal clinical threshold.

| feature | description | value_mean | value_min | value_max | unit | interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| auc | sample-domain trapezoidal area | 1367.9379 | 1009.1349 | 2318.2095 | beats/min | Finite values; inspect class separation and preprocessing change. |
| cv | standard deviation divided by mean | 0.0040 | 0.0001 | 0.0311 | beats/min | Finite values; inspect class separation and preprocessing change. |
| energy | sum of squared samples | 105979.5681 | 56424.5001 | 297648.2785 | beats/min | Finite values; inspect class separation and preprocessing change. |
| entropy | 10-bin Shannon distribution entropy (bits) | 3.0693 | 1.6568 | 3.3219 | beats/min | Finite values; inspect class separation and preprocessing change. |
| iqr | middle-50% spread | 0.4855 | 0.0032 | 3.8209 | beats/min | Finite values; inspect class separation and preprocessing change. |
| kurtosis | tail/peakedness relative to normal | -0.9406 | -1.8072 | 6.0375 | beats/min | Finite values; inspect class separation and preprocessing change. |
| mav | mean absolute value | 71.9967 | 53.1152 | 121.9930 | beats/min | Finite values; inspect class separation and preprocessing change. |
| max | highest value | 72.4835 | 53.1724 | 122.4661 | beats/min | Finite values; inspect class separation and preprocessing change. |
| mean | average level | 71.9967 | 53.1152 | 121.9930 | beats/min | Finite values; inspect class separation and preprocessing change. |
| median | middle value | 71.9962 | 53.1089 | 122.0312 | beats/min | Finite values; inspect class separation and preprocessing change. |
| min | lowest value | 71.5122 | 53.0679 | 121.3057 | beats/min | Finite values; inspect class separation and preprocessing change. |
| range | max minus min | 0.9713 | 0.0143 | 7.5962 | beats/min | Finite values; inspect class separation and preprocessing change. |
| rate |  | 0.0000 | 0.0000 | 0.0000 | beats/min | Finite values; inspect class separation and preprocessing change. |
| rms | root-mean-square magnitude | 71.9980 | 53.1152 | 121.9935 | beats/min | Finite values; inspect class separation and preprocessing change. |
| skewness | asymmetry of value distribution | 0.0139 | -2.1194 | 2.4233 | beats/min | Finite values; inspect class separation and preprocessing change. |
| slope | linear sample-to-sample trend | 0.0006 | -0.3788 | 0.4006 | beats/min | Finite values; inspect class separation and preprocessing change. |
| std | within-window variability | 0.2982 | 0.0048 | 2.3101 | beats/min | Finite values; inspect class separation and preprocessing change. |
| variance | squared variability | 0.1868 | 0.0000 | 5.3365 | beats/min | Finite values; inspect class separation and preprocessing change. |

Conclusion: features are numerically complete in this run; model comparison below indicates their discriminative value.

### E4_IBI: Empatica E4 inter-beat interval

Beat timing and its variability reflect autonomic regulation.

Expected ranges are participant-, device-, and context-dependent; use the raw min/max below as the observed study range rather than applying a universal clinical threshold.

| feature | description | value_mean | value_min | value_max | unit | interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| auc | sample-domain trapezoidal area | 17.6221 | 9.0843 | 26.1762 | seconds | Finite values; inspect class separation and preprocessing change. |
| cv | standard deviation divided by mean | 0.0241 | 0.0000 | 0.1697 | seconds | Finite values; inspect class separation and preprocessing change. |
| energy | sum of squared samples | 17.6395 | 4.5727 | 37.9619 | seconds | Finite values; inspect class separation and preprocessing change. |
| entropy | 10-bin Shannon distribution entropy (bits) | 2.9193 | 0.0000 | 3.3219 | seconds | Finite values; inspect class separation and preprocessing change. |
| iqr | middle-50% spread | 0.0328 | 0.0000 | 0.3036 | seconds | Finite values; inspect class separation and preprocessing change. |
| kurtosis | tail/peakedness relative to normal | -0.6904 | -1.9186 | 15.0526 | seconds | Finite values; inspect class separation and preprocessing change. |
| mav | mean absolute value | 0.9275 | 0.4781 | 1.3777 | seconds | Finite values; inspect class separation and preprocessing change. |
| max | highest value | 0.9639 | 0.4873 | 1.3899 | seconds | Finite values; inspect class separation and preprocessing change. |
| mean | average level | 0.9275 | 0.4781 | 1.3777 | seconds | Finite values; inspect class separation and preprocessing change. |
| median | middle value | 0.9275 | 0.4781 | 1.3799 | seconds | Finite values; inspect class separation and preprocessing change. |
| min | lowest value | 0.8910 | 0.4690 | 1.3659 | seconds | Finite values; inspect class separation and preprocessing change. |
| range | max minus min | 0.0729 | 0.0000 | 0.4792 | seconds | Finite values; inspect class separation and preprocessing change. |
| rate |  | 0.0000 | 0.0000 | 0.0000 | seconds | Finite values; inspect class separation and preprocessing change. |
| rms | root-mean-square magnitude | 0.9280 | 0.4782 | 1.3777 | seconds | Finite values; inspect class separation and preprocessing change. |
| skewness | asymmetry of value distribution | 0.0082 | -3.8661 | 4.1295 | seconds | Finite values; inspect class separation and preprocessing change. |
| slope | linear sample-to-sample trend | -0.0000 | -0.0268 | 0.0189 | seconds | Finite values; inspect class separation and preprocessing change. |
| std | within-window variability | 0.0222 | 0.0000 | 0.1701 | seconds | Finite values; inspect class separation and preprocessing change. |
| variance | squared variability | 0.0011 | 0.0000 | 0.0289 | seconds | Finite values; inspect class separation and preprocessing change. |

Conclusion: features are numerically complete in this run; model comparison below indicates their discriminative value.

### E4_TEMP: Empatica E4 skin temperature

Peripheral vasoconstriction/thermal response may accompany arousal.

Expected ranges are participant-, device-, and context-dependent; use the raw min/max below as the observed study range rather than applying a universal clinical threshold.

| feature | description | value_mean | value_min | value_max | unit | interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| auc | sample-domain trapezoidal area | 633.4730 | 583.2200 | 683.1300 | °C | Finite values; inspect class separation and preprocessing change. |
| cv | standard deviation divided by mean | 0.0003 | 0.0000 | 0.0011 | °C | Finite values; inspect class separation and preprocessing change. |
| energy | sum of squared samples | 22267.1406 | 18844.8900 | 25853.8036 | °C | Finite values; inspect class separation and preprocessing change. |
| entropy | 10-bin Shannon distribution entropy (bits) | 1.0468 | 0.0000 | 2.1219 | °C | Finite values; inspect class separation and preprocessing change. |
| iqr | middle-50% spread | 0.0133 | 0.0000 | 0.0800 | °C | Finite values; inspect class separation and preprocessing change. |
| kurtosis | tail/peakedness relative to normal | -0.6713 | -2.0000 | 5.1111 | °C | Finite values; inspect class separation and preprocessing change. |
| mav | mean absolute value | 33.3407 | 30.6960 | 35.9540 | °C | Finite values; inspect class separation and preprocessing change. |
| max | highest value | 33.3550 | 30.7100 | 35.9700 | °C | Finite values; inspect class separation and preprocessing change. |
| mean | average level | 33.3407 | 30.6960 | 35.9540 | °C | Finite values; inspect class separation and preprocessing change. |
| median | middle value | 33.3407 | 30.6900 | 35.9500 | °C | Finite values; inspect class separation and preprocessing change. |
| min | lowest value | 33.3262 | 30.6900 | 35.9500 | °C | Finite values; inspect class separation and preprocessing change. |
| range | max minus min | 0.0288 | 0.0000 | 0.1000 | °C | Finite values; inspect class separation and preprocessing change. |
| rate |  | 0.0000 | 0.0000 | 0.0000 | °C | Finite values; inspect class separation and preprocessing change. |
| rms | root-mean-square magnitude | 33.3407 | 30.6960 | 35.9540 | °C | Finite values; inspect class separation and preprocessing change. |
| skewness | asymmetry of value distribution | -0.0160 | -2.6667 | 2.6667 | °C | Finite values; inspect class separation and preprocessing change. |
| slope | linear sample-to-sample trend | -0.0001 | -0.0061 | 0.0046 | °C | Finite values; inspect class separation and preprocessing change. |
| std | within-window variability | 0.0111 | 0.0000 | 0.0380 | °C | Finite values; inspect class separation and preprocessing change. |
| variance | squared variability | 0.0001 | 0.0000 | 0.0014 | °C | Finite values; inspect class separation and preprocessing change. |

Conclusion: features are numerically complete in this run; model comparison below indicates their discriminative value.

### RB_ACC: RespiBAN accelerometer

Movement/context signal and potential physiological artefact indicator.

Expected ranges are participant-, device-, and context-dependent; use the raw min/max below as the observed study range rather than applying a universal clinical threshold.

| feature | description | value_mean | value_min | value_max | unit | interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| auc | sample-domain trapezoidal area | 2.0356 | -13.8521 | 17.3930 | device units | Finite values; inspect class separation and preprocessing change. |
| cv | standard deviation divided by mean | 0.2495 | 0.0003 | 159.8110 | device units | Finite values; inspect class separation and preprocessing change. |
| energy | sum of squared samples | 5.8591 | 0.0000 | 21.5093 | device units | Finite values; inspect class separation and preprocessing change. |
| entropy | 10-bin Shannon distribution entropy (bits) | 2.7892 | 0.7476 | 3.2842 | device units | Finite values; inspect class separation and preprocessing change. |
| iqr | middle-50% spread | 0.0108 | 0.0003 | 0.7605 | device units | Finite values; inspect class separation and preprocessing change. |
| kurtosis | tail/peakedness relative to normal | -0.0747 | -1.8357 | 13.2508 | device units | Finite values; inspect class separation and preprocessing change. |
| mav | mean absolute value | 0.4354 | 0.0009 | 0.9765 | device units | Finite values; inspect class separation and preprocessing change. |
| max | highest value | 0.1227 | -0.7268 | 1.5049 | device units | Finite values; inspect class separation and preprocessing change. |
| mean | average level | 0.1071 | -0.7292 | 0.9155 | device units | Finite values; inspect class separation and preprocessing change. |
| median | middle value | 0.1074 | -0.7291 | 0.9159 | device units | Finite values; inspect class separation and preprocessing change. |
| min | lowest value | 0.0901 | -2.6423 | 0.9142 | device units | Finite values; inspect class separation and preprocessing change. |
| range | max minus min | 0.0326 | 0.0009 | 3.5920 | device units | Finite values; inspect class separation and preprocessing change. |
| rate |  | 0.0073 | 0.0000 | 0.8421 | device units | Finite values; inspect class separation and preprocessing change. |
| rms | root-mean-square magnitude | 0.4364 | 0.0010 | 1.0370 | device units | Finite values; inspect class separation and preprocessing change. |
| skewness | asymmetry of value distribution | -0.1670 | -3.5659 | 3.7150 | device units | Finite values; inspect class separation and preprocessing change. |
| slope | linear sample-to-sample trend | -0.0000 | -0.0585 | 0.0493 | device units | Finite values; inspect class separation and preprocessing change. |
| std | within-window variability | 0.0088 | 0.0002 | 0.7345 | device units | Finite values; inspect class separation and preprocessing change. |
| variance | squared variability | 0.0007 | 0.0000 | 0.5395 | device units | Finite values; inspect class separation and preprocessing change. |

Conclusion: features are numerically complete in this run; model comparison below indicates their discriminative value.

### RB_ECG: RespiBAN ECG

Cardiac waveform; morphology and variability carry stress information.

Expected ranges are participant-, device-, and context-dependent; use the raw min/max below as the observed study range rather than applying a universal clinical threshold.

| feature | description | value_mean | value_min | value_max | unit | interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| auc | sample-domain trapezoidal area | 0.0194 | -0.7408 | 0.5871 | mV/device units | Finite values; inspect class separation and preprocessing change. |
| cv | standard deviation divided by mean | 97.5951 | 1.3753 | 34148.9632 | mV/device units | Finite values; inspect class separation and preprocessing change. |
| energy | sum of squared samples | 0.0978 | 0.0027 | 0.8245 | mV/device units | Finite values; inspect class separation and preprocessing change. |
| entropy | 10-bin Shannon distribution entropy (bits) | 2.9337 | 2.0192 | 3.2842 | mV/device units | Finite values; inspect class separation and preprocessing change. |
| iqr | middle-50% spread | 0.1049 | 0.0150 | 0.2560 | mV/device units | Finite values; inspect class separation and preprocessing change. |
| kurtosis | tail/peakedness relative to normal | -1.0651 | -1.8262 | 5.6256 | mV/device units | Finite values; inspect class separation and preprocessing change. |
| mav | mean absolute value | 0.0560 | 0.0094 | 0.1357 | mV/device units | Finite values; inspect class separation and preprocessing change. |
| max | highest value | 0.1045 | 0.0244 | 0.5513 | mV/device units | Finite values; inspect class separation and preprocessing change. |
| mean | average level | 0.0011 | -0.0356 | 0.0279 | mV/device units | Finite values; inspect class separation and preprocessing change. |
| median | middle value | 0.0033 | -0.0579 | 0.0544 | mV/device units | Finite values; inspect class separation and preprocessing change. |
| min | lowest value | -0.1082 | -0.5340 | -0.0210 | mV/device units | Finite values; inspect class separation and preprocessing change. |
| range | max minus min | 0.2127 | 0.0455 | 0.9479 | mV/device units | Finite values; inspect class separation and preprocessing change. |
| rate |  | 0.5451 | 0.1579 | 0.8947 | mV/device units | Finite values; inspect class separation and preprocessing change. |
| rms | root-mean-square magnitude | 0.0649 | 0.0116 | 0.2030 | mV/device units | Finite values; inspect class separation and preprocessing change. |
| skewness | asymmetry of value distribution | -0.0296 | -2.2764 | 1.5175 | mV/device units | Finite values; inspect class separation and preprocessing change. |
| slope | linear sample-to-sample trend | -0.0000 | -0.0105 | 0.0067 | mV/device units | Finite values; inspect class separation and preprocessing change. |
| std | within-window variability | 0.0647 | 0.0115 | 0.2030 | mV/device units | Finite values; inspect class separation and preprocessing change. |
| variance | squared variability | 0.0049 | 0.0001 | 0.0412 | mV/device units | Finite values; inspect class separation and preprocessing change. |

Conclusion: features are numerically complete in this run; model comparison below indicates their discriminative value.

### RB_EDA: RespiBAN electrodermal activity

Sympathetic sudomotor activity.

Expected ranges are participant-, device-, and context-dependent; use the raw min/max below as the observed study range rather than applying a universal clinical threshold.

| feature | description | value_mean | value_min | value_max | unit | interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| auc | sample-domain trapezoidal area | 72.6055 | 10.5810 | 225.9350 | µS/device units | Finite values; inspect class separation and preprocessing change. |
| cv | standard deviation divided by mean | 0.0049 | 0.0005 | 0.1021 | µS/device units | Finite values; inspect class separation and preprocessing change. |
| energy | sum of squared samples | 457.6570 | 6.2041 | 2829.5181 | µS/device units | Finite values; inspect class separation and preprocessing change. |
| entropy | 10-bin Shannon distribution entropy (bits) | 2.8915 | 1.4568 | 3.2842 | µS/device units | Finite values; inspect class separation and preprocessing change. |
| iqr | middle-50% spread | 0.0299 | 0.0007 | 0.9550 | µS/device units | Finite values; inspect class separation and preprocessing change. |
| kurtosis | tail/peakedness relative to normal | -0.5433 | -1.8013 | 8.2160 | µS/device units | Finite values; inspect class separation and preprocessing change. |
| mav | mean absolute value | 3.8214 | 0.5570 | 11.8939 | µS/device units | Finite values; inspect class separation and preprocessing change. |
| max | highest value | 3.8561 | 0.5633 | 12.0526 | µS/device units | Finite values; inspect class separation and preprocessing change. |
| mean | average level | 3.8214 | 0.5570 | 11.8939 | µS/device units | Finite values; inspect class separation and preprocessing change. |
| median | middle value | 3.8191 | 0.5570 | 11.9143 | µS/device units | Finite values; inspect class separation and preprocessing change. |
| min | lowest value | 3.7909 | 0.5497 | 11.6715 | µS/device units | Finite values; inspect class separation and preprocessing change. |
| range | max minus min | 0.0652 | 0.0024 | 3.6607 | µS/device units | Finite values; inspect class separation and preprocessing change. |
| rate |  | 0.0000 | 0.0000 | 0.0000 | µS/device units | Finite values; inspect class separation and preprocessing change. |
| rms | root-mean-square magnitude | 3.8216 | 0.5570 | 11.8944 | µS/device units | Finite values; inspect class separation and preprocessing change. |
| skewness | asymmetry of value distribution | 0.1737 | -2.8938 | 2.6197 | µS/device units | Finite values; inspect class separation and preprocessing change. |
| slope | linear sample-to-sample trend | -0.0000 | -0.0890 | 0.0746 | µS/device units | Finite values; inspect class separation and preprocessing change. |
| std | within-window variability | 0.0200 | 0.0007 | 0.7699 | µS/device units | Finite values; inspect class separation and preprocessing change. |
| variance | squared variability | 0.0026 | 0.0000 | 0.5928 | µS/device units | Finite values; inspect class separation and preprocessing change. |

Conclusion: features are numerically complete in this run; model comparison below indicates their discriminative value.

### RB_EMG: RespiBAN electromyogram

Muscle activation/tension can increase during stress.

Expected ranges are participant-, device-, and context-dependent; use the raw min/max below as the observed study range rather than applying a universal clinical threshold.

| feature | description | value_mean | value_min | value_max | unit | interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| auc | sample-domain trapezoidal area | -0.0513 | -0.7689 | -0.0298 | mV/device units | Finite values; inspect class separation and preprocessing change. |
| cv | standard deviation divided by mean | 0.1536 | 0.0284 | 3.7483 | mV/device units | Finite values; inspect class separation and preprocessing change. |
| energy | sum of squared samples | 0.0005 | 0.0001 | 0.4597 | mV/device units | Finite values; inspect class separation and preprocessing change. |
| entropy | 10-bin Shannon distribution entropy (bits) | 2.8648 | 1.2918 | 3.2842 | mV/device units | Finite values; inspect class separation and preprocessing change. |
| iqr | middle-50% spread | 0.0005 | 0.0001 | 0.0297 | mV/device units | Finite values; inspect class separation and preprocessing change. |
| kurtosis | tail/peakedness relative to normal | -0.0465 | -1.6873 | 8.2105 | mV/device units | Finite values; inspect class separation and preprocessing change. |
| mav | mean absolute value | 0.0027 | 0.0016 | 0.0674 | mV/device units | Finite values; inspect class separation and preprocessing change. |
| max | highest value | -0.0018 | -0.0037 | 0.1151 | mV/device units | Finite values; inspect class separation and preprocessing change. |
| mean | average level | -0.0027 | -0.0391 | -0.0016 | mV/device units | Finite values; inspect class separation and preprocessing change. |
| median | middle value | -0.0027 | -0.0042 | -0.0016 | mV/device units | Finite values; inspect class separation and preprocessing change. |
| min | lowest value | -0.0039 | -0.6015 | -0.0022 | mV/device units | Finite values; inspect class separation and preprocessing change. |
| range | max minus min | 0.0021 | 0.0004 | 0.7166 | mV/device units | Finite values; inspect class separation and preprocessing change. |
| rate |  | 0.0010 | 0.0000 | 0.6316 | mV/device units | Finite values; inspect class separation and preprocessing change. |
| rms | root-mean-square magnitude | 0.0028 | 0.0017 | 0.1516 | mV/device units | Finite values; inspect class separation and preprocessing change. |
| skewness | asymmetry of value distribution | -0.0386 | -2.8815 | 2.5637 | mV/device units | Finite values; inspect class separation and preprocessing change. |
| slope | linear sample-to-sample trend | 0.0000 | -0.0052 | 0.0057 | mV/device units | Finite values; inspect class separation and preprocessing change. |
| std | within-window variability | 0.0005 | 0.0001 | 0.1465 | mV/device units | Finite values; inspect class separation and preprocessing change. |
| variance | squared variability | 0.0000 | 0.0000 | 0.0215 | mV/device units | Finite values; inspect class separation and preprocessing change. |

Conclusion: features are numerically complete in this run; model comparison below indicates their discriminative value.

### RB_RESP: RespiBAN respiration

Breathing pattern, rate and amplitude can change under stress.

Expected ranges are participant-, device-, and context-dependent; use the raw min/max below as the observed study range rather than applying a universal clinical threshold.

| feature | description | value_mean | value_min | value_max | unit | interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| auc | sample-domain trapezoidal area | 0.9890 | -165.2249 | 97.6941 | device units | Finite values; inspect class separation and preprocessing change. |
| cv | standard deviation divided by mean | 20.8942 | 0.2780 | 3754.4061 | device units | Finite values; inspect class separation and preprocessing change. |
| energy | sum of squared samples | 136.6954 | 5.4183 | 3042.5332 | device units | Finite values; inspect class separation and preprocessing change. |
| entropy | 10-bin Shannon distribution entropy (bits) | 2.9248 | 1.5568 | 3.2842 | device units | Finite values; inspect class separation and preprocessing change. |
| iqr | middle-50% spread | 3.1518 | 0.3342 | 25.7951 | device units | Finite values; inspect class separation and preprocessing change. |
| kurtosis | tail/peakedness relative to normal | -0.9415 | -1.7836 | 6.4546 | device units | Finite values; inspect class separation and preprocessing change. |
| mav | mean absolute value | 1.8405 | 0.4125 | 11.5753 | device units | Finite values; inspect class separation and preprocessing change. |
| max | highest value | 3.7317 | -1.6254 | 19.2156 | device units | Finite values; inspect class separation and preprocessing change. |
| mean | average level | 0.0532 | -8.2976 | 5.1144 | device units | Finite values; inspect class separation and preprocessing change. |
| median | middle value | -0.2475 | -10.3771 | 5.8861 | device units | Finite values; inspect class separation and preprocessing change. |
| min | lowest value | -2.8519 | -19.7883 | 1.0949 | device units | Finite values; inspect class separation and preprocessing change. |
| range | max minus min | 6.5835 | 1.0161 | 32.3393 | device units | Finite values; inspect class separation and preprocessing change. |
| rate |  | 0.1439 | 0.0000 | 0.4737 | device units | Finite values; inspect class separation and preprocessing change. |
| rms | root-mean-square magnitude | 2.1797 | 0.5205 | 12.3340 | device units | Finite values; inspect class separation and preprocessing change. |
| skewness | asymmetry of value distribution | 0.2985 | -2.3156 | 2.4544 | device units | Finite values; inspect class separation and preprocessing change. |
| slope | linear sample-to-sample trend | 0.0025 | -1.8715 | 1.4912 | device units | Finite values; inspect class separation and preprocessing change. |
| std | within-window variability | 2.0534 | 0.3063 | 12.1360 | device units | Finite values; inspect class separation and preprocessing change. |
| variance | squared variability | 5.9826 | 0.0938 | 147.2821 | device units | Finite values; inspect class separation and preprocessing change. |

Conclusion: features are numerically complete in this run; model comparison below indicates their discriminative value.

### RB_TEMP: RespiBAN temperature

Slow peripheral thermal trend; interpret over longer intervals.

Expected ranges are participant-, device-, and context-dependent; use the raw min/max below as the observed study range rather than applying a universal clinical threshold.

| feature | description | value_mean | value_min | value_max | unit | interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| auc | sample-domain trapezoidal area | 603.2752 | 493.5128 | 653.5722 | °C/device units | Finite values; inspect class separation and preprocessing change. |
| cv | standard deviation divided by mean | 0.0012 | 0.0000 | 1.1429 | °C/device units | Finite values; inspect class separation and preprocessing change. |
| energy | sum of squared samples | 20231.4117 | 15871.8835 | 31772.1233 | °C/device units | Finite values; inspect class separation and preprocessing change. |
| entropy | 10-bin Shannon distribution entropy (bits) | 2.9614 | 1.3918 | 3.3219 | °C/device units | Finite values; inspect class separation and preprocessing change. |
| iqr | middle-50% spread | 0.0172 | 0.0005 | 6.1235 | °C/device units | Finite values; inspect class separation and preprocessing change. |
| kurtosis | tail/peakedness relative to normal | -0.6510 | -1.7727 | 8.1983 | °C/device units | Finite values; inspect class separation and preprocessing change. |
| mav | mean absolute value | 31.7599 | 28.1708 | 36.9575 | °C/device units | Finite values; inspect class separation and preprocessing change. |
| max | highest value | 31.7917 | 28.1753 | 57.8344 | °C/device units | Finite values; inspect class separation and preprocessing change. |
| mean | average level | 31.7516 | 26.2315 | 34.3982 | °C/device units | Finite values; inspect class separation and preprocessing change. |
| median | middle value | 31.7572 | 28.1711 | 34.3994 | °C/device units | Finite values; inspect class separation and preprocessing change. |
| min | lowest value | 31.6475 | -88.8571 | 34.3842 | °C/device units | Finite values; inspect class separation and preprocessing change. |
| range | max minus min | 0.1443 | 0.0025 | 146.6915 | °C/device units | Finite values; inspect class separation and preprocessing change. |
| rate |  | 0.0001 | 0.0000 | 0.1053 | °C/device units | Finite values; inspect class separation and preprocessing change. |
| rms | root-mean-square magnitude | 31.7622 | 28.1708 | 39.8573 | °C/device units | Finite values; inspect class separation and preprocessing change. |
| skewness | asymmetry of value distribution | 0.0338 | -2.8784 | 2.1453 | °C/device units | Finite values; inspect class separation and preprocessing change. |
| slope | linear sample-to-sample trend | 0.0001 | -1.0754 | 1.1691 | °C/device units | Finite values; inspect class separation and preprocessing change. |
| std | within-window variability | 0.0319 | 0.0007 | 29.9896 | °C/device units | Finite values; inspect class separation and preprocessing change. |
| variance | squared variability | 0.6988 | 0.0000 | 899.3784 | °C/device units | Finite values; inspect class separation and preprocessing change. |

Conclusion: features are numerically complete in this run; model comparison below indicates their discriminative value.

## Signal-wise preprocessing comparison

| experiment | variant | accuracy | precision | recall | f1 | roc_auc | training_seconds | inference_ms_per_window |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ALL_SIGNALS | filtered | 0.5618 | 0.0227 | 0.0122 | 0.0159 | 0.7401 | 0.2458 | 0.0620 |
| ALL_SIGNALS | filtered_normalized | 0.5866 | 0.1329 | 0.0772 | 0.0977 | 0.7147 | 0.2226 | 0.0531 |
| ALL_SIGNALS | raw | 0.5418 | 0.0446 | 0.0285 | 0.0347 | 0.7336 | 0.2769 | 0.0588 |
| E4_ACC | filtered | 0.4547 | 0.0000 | 0.0000 | 0.0000 | 0.0217 | 0.0955 | 0.0227 |
| E4_ACC | filtered_normalized | 0.4782 | 0.0000 | 0.0000 | 0.0000 | 0.0267 | 0.0824 | 0.0229 |
| E4_ACC | raw | 0.4452 | 0.0000 | 0.0000 | 0.0000 | 0.0237 | 0.6942 | 0.0679 |
| E4_BVP | filtered | 0.4923 | 0.3309 | 0.7358 | 0.4565 | 0.6102 | 0.0491 | 0.0060 |
| E4_BVP | filtered_normalized | 0.4653 | 0.3249 | 0.7846 | 0.4595 | 0.6223 | 0.1160 | 0.0193 |
| E4_BVP | raw | 0.6384 | 0.4265 | 0.7195 | 0.5356 | 0.6675 | 0.1150 | 0.0387 |
| E4_EDA | filtered | 0.7456 | 0.6923 | 0.2195 | 0.3333 | 0.9308 | 0.1203 | 0.0091 |
| E4_EDA | filtered_normalized | 0.8681 | 0.6893 | 0.9919 | 0.8133 | 0.9342 | 0.0685 | 0.0185 |
| E4_EDA | raw | 0.7809 | 0.7273 | 0.3902 | 0.5079 | 0.9149 | 0.1154 | 0.0147 |
| E4_HR | filtered | 0.5183 | 0.1970 | 0.2154 | 0.2058 | 0.3585 | 0.1263 | 0.0161 |
| E4_HR | filtered_normalized | 0.5383 | 0.1651 | 0.1463 | 0.1552 | 0.4376 | 0.0918 | 0.0183 |
| E4_HR | raw | 0.5183 | 0.1970 | 0.2154 | 0.2058 | 0.3585 | 0.0719 | 0.0180 |
| E4_IBI | filtered | 0.6360 | 0.2994 | 0.1911 | 0.2333 | 0.5787 | 0.2189 | 0.0216 |
| E4_IBI | filtered_normalized | 0.5771 | 0.2945 | 0.3293 | 0.3109 | 0.5450 | 0.1589 | 0.0287 |
| E4_IBI | raw | 0.6360 | 0.2994 | 0.1911 | 0.2333 | 0.5787 | 0.1248 | 0.0263 |
| E4_TEMP | filtered | 0.7420 | 0.9655 | 0.1138 | 0.2036 | 0.9760 | 0.1817 | 0.0608 |
| E4_TEMP | filtered_normalized | 0.6431 | 0.3777 | 0.3577 | 0.3674 | 0.5311 | 0.2138 | 0.1050 |
| E4_TEMP | raw | 0.7444 | 0.9677 | 0.1220 | 0.2166 | 0.9772 | 0.3899 | 0.0222 |
| RB_ACC | filtered | 0.6078 | 0.2814 | 0.2276 | 0.2517 | 0.7817 | 0.1536 | 0.0318 |
| RB_ACC | filtered_normalized | 0.7185 | 0.6667 | 0.0569 | 0.1049 | 0.9426 | 0.1199 | 0.0144 |
| RB_ACC | raw | 0.6172 | 0.3015 | 0.2439 | 0.2697 | 0.7826 | 0.2428 | 0.0405 |
| RB_ECG | filtered | 0.4452 | 0.3372 | 0.9472 | 0.4973 | 0.8073 | 0.0943 | 0.0124 |
| RB_ECG | filtered_normalized | 0.4947 | 0.3628 | 0.9837 | 0.5301 | 0.8196 | 0.0978 | 0.0160 |
| RB_ECG | raw | 0.3510 | 0.2996 | 0.9268 | 0.4528 | 0.6454 | 0.1096 | 0.0155 |
| RB_EDA | filtered | 0.7220 | 0.5234 | 0.4553 | 0.4870 | 0.6930 | 0.0755 | 0.0146 |
| RB_EDA | filtered_normalized | 0.7197 | 0.5161 | 0.5203 | 0.5182 | 0.7255 | 0.0816 | 0.0216 |
| RB_EDA | raw | 0.7232 | 0.5256 | 0.4593 | 0.4902 | 0.6860 | 0.1025 | 0.0423 |
| RB_EMG | filtered | 0.4900 | 0.3156 | 0.6504 | 0.4250 | 0.5575 | 0.0726 | 0.0201 |
| RB_EMG | filtered_normalized | 0.4629 | 0.3250 | 0.7927 | 0.4610 | 0.6041 | 0.0492 | 0.0131 |
| RB_EMG | raw | 0.4075 | 0.2701 | 0.6138 | 0.3752 | 0.4597 | 0.1062 | 0.0167 |
| RB_RESP | filtered | 0.7998 | 0.6357 | 0.7236 | 0.6768 | 0.8429 | 0.0788 | 0.0122 |
| RB_RESP | filtered_normalized | 0.7656 | 0.5681 | 0.7967 | 0.6633 | 0.8639 | 0.1126 | 0.0149 |
| RB_RESP | raw | 0.8163 | 0.6619 | 0.7480 | 0.7023 | 0.8503 | 0.0930 | 0.0143 |
| RB_TEMP | filtered | 0.6290 | 0.0000 | 0.0000 | 0.0000 | 0.6890 | 0.0785 | 0.0094 |
| RB_TEMP | filtered_normalized | 0.7750 | 0.5629 | 1.0000 | 0.7204 | 0.9682 | 0.0583 | 0.0164 |
| RB_TEMP | raw | 0.6207 | 0.0000 | 0.0000 | 0.0000 | 0.6865 | 0.0974 | 0.0137 |

## Interpretation and recommended configuration

- Best all-signal preprocessing: **filtered_normalized** (held-out F1 0.0977, ROC-AUC 0.7147).
- Strongest individual experiment: **E4_EDA / filtered_normalized** (F1 0.8133).
- Weakest individual experiment: **E4_ACC / filtered** (F1 0.0000).
- Filtering/normalization are meaningful only when the held-out F1/ROC-AUC improvement persists on a new subject split; the tables report estimates, not clinical claims.
- Recommended feature set: retain features with `quality_flag=usable`, then select within training folds only to avoid test-subject leakage.
- Future work: add dedicated ECG/EDA/BVP peak features from the existing feature_extraction.py module, nested cross-validation, and leave-one-subject-out evaluation.

## References

- Schmidt, P. et al. (2018). Introducing WESAD, a multimodal dataset for wearable stress and affect detection. ICMI 2018.
- The generic statistics are implemented in this repository's `06_feature_engineering/feature_extraction.py`; signal-specific extractors there are not part of the present classical-ML experiment unless explicitly added.
