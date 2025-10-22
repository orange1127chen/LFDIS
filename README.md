# LFDIS:Lightweight and Fast High-Resolution Dichotomous Image Segmentation with Depth Guidance

##**Note**: This work will be submitted to The Visual Computer. 
## Environment preparation

### Requirements

```bash
Linux with python ≥ 3.8
Pytorch ≥ 1.7 and torchvison that matches the Pytorch installation.
Opencv
Numpy
Apex
```

## Dataset preparation

## Download the datasets and annotation files


[DIS5K](https://github.com/xuebinqin/DIS)


```bash
DATASET_ROOT/
    ├── DIS5K
       ├── DIS-TR
          ├── depth
          ├── im
          ├── gt
          ├── trunk-origin
          ├── struct-origin
       ├── DIS-VD
          ├── im
          ├── gt
          ├── depth
       ├── DIS-TE1
       ...
       ├── DIS-TE2
       ...
       ├── DIS-TE3
       ...
       ├── DIS-TE4
       ...
```

## Usage

### Train&Test

To train our PFNet on single GPU by following command,the trained models will be saved in savePath folder. You can modify datapath if you want to run your own datases.

```bash
python train.py
```

To test our PFNet on DIS5K, the prediction maps will be saved in DIS5K_Pre folder.

```python
python3 test.py 
```
## Evaluation
To Evaluate the predicted results.

```bash
cd metrics
python3 test_metrics.py 
python3 hce_metric_main.py
```

Get trunk map and struct map
Split the ground truth into trunk map and struct map, which will be saved into DIS5K-TR/gt/Trunk-origin and DIS5K-TR/gt/struct-origin.

```bash
cd utils
python3 utils.py
```

Get depth map
[Depth maps](https://github.com/Westlake-AGI-Lab/Distill-Any-Depth)
