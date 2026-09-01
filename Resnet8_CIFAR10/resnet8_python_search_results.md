# ResNet8 CIFAR10 Python Approximate Communication Search Results

## Summary

- Dataset: CIFAR10 full test set, 10000 images.
- Exact baseline accuracy: 85.68% (8568/10000).
- Accuracy constraint: approximate accuracy >= 83.68%.
- CPU-only search elapsed time: 25.76 min.
- Shortcut edges are always exact: `1->4`, `4->7`, `7->8`, `8->11`, `11->12`.
- Python thresholds are activation-q16 simulation thresholds; noxim also quantizes weights, so use these mainly as level/sensitivity guidance.

## Final Results

| scheme | accuracy | correct | approx value ratio | eval seconds |
| --- | --- | --- | --- | --- |
| fas | 84.01% | 8401/10000 | 56.49% | 29.6 |

## Layer Thresholds

| layer | name | type | neurons | q16 scale | FAS th0..th3 | SAP delta th0..th3 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | conv1 | Convolution | 16384 | 0.00019170159 | 621 1123 1717 2732 | 67 195 440 1021 |
| 2 | block1.conv1 | Convolution | 16384 | 0.00018715533 | 496 1174 2006 3189 | 217 579 1172 2305 |
| 3 | block1.conv2 | Convolution | 16384 | 0.0002863074 | 470 965 1567 2442 | 151 400 790 1519 |
| 4 | block1.add | Add | 16384 | 0.00028578409 | 712 1335 2039 2869 | 138 358 695 1303 |
| 5 | block2.conv1 | Convolution | 8192 | 0.00017311114 | 602 1293 2149 3351 | 326 775 1403 2455 |
| 6 | block2.conv2 | Convolution | 8192 | 0.00020983293 | 652 1372 2263 3612 | 434 1001 1767 3022 |
| 7 | block2.shortcut.0 | Convolution | 8192 | 0.00018565768 | 198 430 723 1412 | 68 184 382 780 |
| 8 | block2.add | Add | 8192 | 0.00019236937 | 706 1488 2454 3875 | 368 850 1507 2590 |
| 9 | block3.conv1 | Convolution | 4096 | 0.00012997543 | 538 1175 1995 3264 | 434 969 1685 2839 |
| 10 | block3.conv2 | Convolution | 4096 | 0.00067958494 | 440 934 1575 2618 | 320 697 1192 1992 |
| 11 | block3.shortcut.0 | Convolution | 4096 | 0.00014103551 | 691 1424 2267 3458 | 525 1183 2050 3444 |
| 12 | block3.add | Add | 4096 | 0.00068437965 | 492 1087 1878 3169 | 280 626 1099 1885 |
| 13 | avgpool | Pooling | 64 | 0.00016425981 | 1355 2354 3547 5471 | 0 0 0 0 |
| 14 | fc | Dense | 10 | 0.00076434119 | 1377 2820 4454 6732 | 1596 3346 5591 9265 |

## fas Search

| layer | name | neurons | level |
| --- | --- | --- | --- |
| 1 | conv1 | 16384 | 0 |
| 2 | block1.conv1 | 16384 | 0 |
| 3 | block1.conv2 | 16384 | 0 |
| 4 | block1.add | 16384 | 0 |
| 5 | block2.conv1 | 8192 | 3 |
| 6 | block2.conv2 | 8192 | 3 |
| 7 | block2.shortcut.0 | 8192 | -1 |
| 8 | block2.add | 8192 | 0 |
| 9 | block3.conv1 | 4096 | 3 |
| 10 | block3.conv2 | 4096 | 3 |
| 11 | block3.shortcut.0 | 4096 | -1 |
| 12 | block3.add | 4096 | 3 |
| 13 | avgpool | 64 | 0 |
| 14 | fc | 10 | -1 |

### Search History

| stage | level | meets constraint | accuracy | approx value ratio |
| --- | --- | --- | --- | --- |
| global | 0 | yes | 84.22% | 45.30% |
| global | 1 | no | 82.11% | 57.69% |
| layer 1 conv1 | 1 | yes | 83.90% | 47.09% |
| layer 1 conv1 | 2 | yes | 84.16% | 48.95% |
| layer 1 conv1 | 3 | no | 83.37% | 50.73% |
| layer 2 block1.conv1 | 1 | no | 83.60% | 46.87% |
| layer 3 block1.conv2 | 1 | no | 83.29% | 48.68% |
| layer 4 block1.add | 1 | no | 83.16% | 47.44% |
| layer 5 block2.conv1 | 1 | yes | 84.25% | 45.98% |
| layer 5 block2.conv1 | 2 | yes | 84.37% | 46.70% |
| layer 5 block2.conv1 | 3 | yes | 84.80% | 47.52% |
| layer 6 block2.conv2 | 1 | yes | 84.76% | 49.14% |
| layer 6 block2.conv2 | 2 | yes | 84.51% | 50.72% |
| layer 6 block2.conv2 | 3 | yes | 84.12% | 52.37% |
| layer 8 block2.add | 1 | yes | 83.89% | 53.01% |
| layer 8 block2.add | 2 | yes | 83.69% | 53.64% |
| layer 8 block2.add | 3 | no | 83.36% | 54.14% |
| layer 9 block3.conv1 | 1 | yes | 84.11% | 52.57% |
| layer 9 block3.conv1 | 2 | yes | 84.01% | 52.80% |
| layer 9 block3.conv1 | 3 | yes | 84.01% | 53.06% |
| layer 10 block3.conv2 | 1 | yes | 83.92% | 53.90% |
| layer 10 block3.conv2 | 2 | yes | 84.04% | 54.72% |
| layer 10 block3.conv2 | 3 | yes | 84.09% | 55.52% |
| layer 12 block3.add | 1 | yes | 84.03% | 55.91% |
| layer 12 block3.add | 2 | yes | 83.96% | 56.26% |
| layer 12 block3.add | 3 | yes | 84.01% | 56.49% |
| layer 13 avgpool | 1 | yes | 83.87% | 56.50% |
| layer 13 avgpool | 2 | yes | 83.75% | 56.51% |
| layer 13 avgpool | 3 | no | 83.41% | 56.52% |

## noxim Mapping Notes

- FAS templates from this search are written as `resnet8_fas_*_python.txt`.
- SAP templates from this search are written as `resnet8_sap_*_python_*.txt`.
- ABDTR template from this search is written as `resnet8_abdtr_drop_python.txt`.
- Re-check final choices in noxim because Python only simulates activation q16 communication effects.
