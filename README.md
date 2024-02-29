# AKGVP
Aligning Knowledge Graph with Visual Perception for Object-goal Navigation (ICRA 2024)

## Setup
- Clone the repository and move into the top-level directory `cd AKGVP`
- Create conda environment. `conda env create -f environment.yml`
- Activate the environment. `conda activate akgvp`
- Our settings of dataset follows previous works, please refer to [HOZ](https://github.com/sx-zhang/HOZ.git) and [L-sTDE](https://github.com/sx-zhang/Layout-based-sTDE.git) for AI2THOR.
- After placing the dataset, use CLIP to generate image features. `python create_image_feat.py`

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
