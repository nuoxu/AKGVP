# AKGVP
Aligning Knowledge Graph with Visual Perception for Object-goal Navigation (ICRA 2024)

## Setup
- Clone the repository and move into the top-level directory `cd AKGVP`
- Create conda environment. `conda env create -f environment.yml`
- Activate the environment. `conda activate ng`
- Our settings of dataset follows previous works, please refer to [HOZ](https://github.com/sx-zhang/HOZ.git) for AI2THOR and [SemExp](https://github.com/devendrachaplot/Object-Goal-Navigation.git) for Gibson.  
## Training and Evaluation
### Train our Layout-based model 
```shell
python main.py \
      --title AKGVPModel \
      --model AKGVPModel \
      --workers 12 \
      --gpu-ids 0 \
      --images-file-name clip_featuremap.hdf5
```
### Evaluate our model with sTDE (our Layout-based sTDE model) 
```shell
python full_eval.py \
        --title AKGVPModel \
        --model AKGVPModel \
        --results-json AKGVPModel.json \
        --gpu-ids 0 \
        --images-file-name clip_featuremap.hdf5 \
        --save-model-dir trained_models
```
