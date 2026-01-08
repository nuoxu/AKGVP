# Aligning Knowledge Graph with Visual Perception for Object-goal Navigation (ICRA 2024)

https://github.com/nuoxu/AKGVP/assets/26222001/63f38873-c51c-4b1e-9d76-cf716ef0de07

## Update
- The dataset used in the paper can be found [here](https://github.com/xiaobaishu0097/ECCV-VN?tab=readme-ov-file). Since the link of RGB data has expired, we have uploaded a backup copy of the [RGB data](https://www.kaggle.com/datasets/hellob/ai2thor-clip). Please check it.

## Setup
- Clone the repository and move into the top-level directory `cd AKGVP`
- Create conda environment. `conda env create -f environment.yml`
- Activate the environment. `conda activate akgvp`
- Our settings of dataset follows previous works, please refer to [HOZ](https://github.com/sx-zhang/HOZ.git) and [L-sTDE](https://github.com/sx-zhang/Layout-based-sTDE.git) for AI2THOR.
- After placing the dataset, use CLIP to generate image features. `python create_image_feat.py`
- For zero-shot navigation, lines 70-73 in `runners/a3c_train.py` can be enabled. In this way, certain categories will be filtered during the training.

## Training and Evaluation
### Train the AKGVP model 
```shell
python main.py \
      --title AKGVPModel \
      --model AKGVPModel \
      --workers 4 \
      --gpu-ids 0 \
      --images-file-name clip_featuremap.hdf5
```
### Evaluate the AKGVP model
```shell
python full_eval.py \
        --title AKGVPModel \
        --model AKGVPModel \
        --results-json AKGVPModel.json \
        --gpu-ids 0 \
        --images-file-name clip_featuremap.hdf5 \
        --save-model-dir trained_models
```
### Visualization
```shell
python visualization.py
```
