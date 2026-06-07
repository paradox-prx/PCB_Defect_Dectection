# Run1 vs Run2 Comparison (Detailed)

## Scope
- **Run 1:** `exp_001_yolov11n_baseline_200ep`
- **Run 2:** `exp_002_yolov11n_kaggle_pcb_200ep`
- **Goal:** Compare what changed between runs and how results moved.

## Artifact Paths
- Run 1 dir: `/home/vector/Documents/abdullah_workspace/PCB_Defect_Dectection/results/exp_001_yolov11n_baseline_200ep`
- Run 2 dir: `/home/vector/Documents/abdullah_workspace/PCB_Defect_Dectection/results/exp_002_yolov11n_kaggle_pcb_200ep`
- Run 1 test-eval dir: `/home/vector/Documents/abdullah_workspace/PCB_Defect_Dectection/results/exp_001_yolov11n_baseline_200ep_test_eval`
- Run 2 test-eval dir: `/home/vector/Documents/abdullah_workspace/PCB_Defect_Dectection/results/exp_002_yolov11n_kaggle_pcb_200ep_test_eval`

## What Changed (Config Diff)
| Field | Run 1 | Run 2 |
|---|---|---|
| model | `yolo11n.pt` | `yolo11n.pt` |
| epochs (requested) | 200 | 200 |
| batch | 16 | 16 |
| imgsz | 640 | 640 |
| optimizer | auto | auto |
| seed | 42 | 42 |
| data | `/home/vector/Documents/abdullah_workspace/PCB_Defect_Dectection/data_pcb_github.yaml` | `/home/vector/.cache/kagglehub/datasets/norbertelter/pcb-defect-dataset/versions/2/pcb-defect-dataset/data.yaml` |

**Key point:** Hyperparameters are effectively the same; the major change is the dataset source (local GitHub-style dataset vs Kaggle PCB dataset).

## Dataset Comparison (From YAML + Labels)
### Split Size / Annotation Coverage
| Split | Run 1 images | Run 1 objects | Run 1 missing labels | Run 2 images | Run 2 objects | Run 2 missing labels |
|---|---:|---:|---:|---:|---:|---:|
| train | 483 | 2058 | 0 | 8534 | 12991 | 2164 |
| val | 138 | 589 | 0 | 1066 | 1595 | 264 |
| test | 72 | 306 | 0 | 1068 | 1662 | 239 |

### Class Distribution (Total objects across train+val+test)
| Class | Run 1 count | Run 2 count | Delta | Delta % |
|---|---:|---:|---:|---:|
| missing_hole | 497 | 2709 | +2212 | +445.07% |
| mouse_bite | 492 | 2763 | +2271 | +461.59% |
| open_circuit | 482 | 2661 | +2179 | +452.07% |
| short | 491 | 2631 | +2140 | +435.85% |
| spur | 488 | 2727 | +2239 | +458.81% |
| spurious_copper | 503 | 2757 | +2254 | +448.11% |

## Training Progress / Runtime
| Item | Run 1 | Run 2 |
|---|---:|---:|
| epochs completed (from results.csv) | 187 | 200 |
| training time column at final epoch | 423.764 | 8023.870 |

**Observation:** Run 1 stopped at epoch 187 (before 200), while Run 2 completed epoch 200.

## Validation Metrics Comparison (from `results.csv`)
### Best-Epoch Snapshot (by highest mAP50)
| Metric | Run 1 | Run 2 | Delta | Delta % |
|---|---:|---:|---:|---:|
| metrics/precision(B) | 0.95134 | 0.97988 | +0.02854 | +3.00% |
| metrics/recall(B) | 0.87674 | 0.98647 | +0.10973 | +12.52% |
| metrics/mAP50(B) | 0.92435 | 0.99089 | +0.06654 | +7.20% |
| metrics/mAP50-95(B) | 0.44993 | 0.61232 | +0.16239 | +36.09% |
| val/box_loss | 1.74230 | 1.11374 | -0.62856 | -36.08% |
| val/cls_loss | 0.77183 | 0.42246 | -0.34937 | -45.27% |
| val/dfl_loss | 0.90939 | 0.84183 | -0.06756 | -7.43% |
| best epoch index | 154 | 126 | -28 | n/a |

### Final-Epoch Snapshot
| Metric | Run 1 (epoch end) | Run 2 (epoch end) | Delta | Delta % |
|---|---:|---:|---:|---:|
| metrics/precision(B) | 0.93443 | 0.98475 | +0.05032 | +5.39% |
| metrics/recall(B) | 0.88552 | 0.98716 | +0.10164 | +11.48% |
| metrics/mAP50(B) | 0.91622 | 0.99044 | +0.07422 | +8.10% |
| metrics/mAP50-95(B) | 0.46331 | 0.63453 | +0.17122 | +36.96% |
| val/box_loss | 1.70473 | 1.04637 | -0.65836 | -38.62% |
| val/cls_loss | 0.75267 | 0.40410 | -0.34857 | -46.31% |
| val/dfl_loss | 0.90608 | 0.82761 | -0.07847 | -8.66% |

## Notes on Test Results
- Both `*_test_eval` folders contain plots/images (PR/F1 curves, confusion matrices, prediction samples), but no tabular metric file was found there.
- Numeric comparisons above are therefore based on training-run `results.csv` validation metrics.

## Bottom Line
- The dominant experimental change was dataset scale/quality, not model hyperparameters.
- Run 2 substantially outperformed Run 1 on all key validation metrics, with much lower validation losses and significantly higher mAP.
- Because class order differs between dataset YAML files, class names were aligned by label text when comparing class distributions.