# HeartDeformNets

This package provides a deep learning framework to predict deformation of whole heart mesh templates from volumetric patient image data by leveraging a graph convolutional network to predict the translations of a small set of control handles to smoothly deform a whole heart template using biharmonic coordinates. Further details of the underlying network architecture are described in this [paper](https://doi.org/10.1109/TMI.2022.3219284).

<img width="1961" alt="network2" src="https://user-images.githubusercontent.com/31931939/184846001-eb3b9442-ae46-4152-a3dc-791e1ccdf946.png">


## Installation ![Python versions](https://img.shields.io/badge/python-3.7-blue) ![CUDA versions](https://img.shields.io/badge/cuda-10.0-green) ![CUDA versions](https://img.shields.io/badge/cuda-10.1-green) ![Ubuntu versions](https://img.shields.io/badge/ubuntu-20.04.6.LTS-orange)
To download our source code with submodules, please run 
```
git clone --recurse-submodules https://github.com/charlenemauger1/HeartDeformNets.git
```
The dependencies of our implementation can be installed by running the following command.
```
cd HeartDeformNets
conda create -n deformnet python=3.7
conda activate deformnet
pip install -r requirements.txt
```

If the tensorflow-gpu version cannot be installed, try terminating the above shell script and run the following command:

```
pip install tensorflow-gpu==1.15.5
```
Tensorflow 1.15 expects cuda 10.0 but you can also make it work with cuda 10.1 by installing the following packages with Anaconda:
```
conda install cudatoolkit=10.0
conda install cudnn=7.6.5
```
## Template Meshes

We provide the following template meshes in `templates/meshes`
- `mmwhs_template.vtp`: The whole heart mesh template used in Task1, with the blood pools of 4 chambers, aorta and pulmonary arteries.
- `wh_template.vtp`: The whole heart mesh template used in Task2, containing pulmonary veins and vena cava inlet geometries for CFD simulations. 
- `lh_template.vtp` and  `rh_template.vtp`: The left heart and the right heart mesh template constructed from `wh_template.vtp` for 4-chamber CFD simulations of cardiac flow.
- `lv_template.vtp`: The left ventricle template constructed from `mmwhs_template.vtp` for CFD simulation of left ventricle flow. 

Those whole heart templates were created from the ground truth segmentation of a training sample. We include an example segmentation we used to create `wh_template.vtp` here: `templates/segmentation/wh.nii.gz`. To construct the training template mesh as well as the associated biharmonic coordinates and mesh information required during training, you need the following steps

- Compile the C++ code for computing biharmonc coordinates in `templates/bc` by 
```
cd templates/bc
mkdir build && cd build && cmake .. && make
```
- Specify the `output_dir` and the path of the segmentation file `seg_fn` in `create_template.sh` and then 
```
source create_template.sh
```

If you get the warning <code style="color : darkorange">[C6ECF740] vtkMath.cxx:596 WARN| Unable to factor linear system</code> when running `source create_template.sh`, it can be ignored. The warning message is from mesh decimation.

## Evaluation

We provide the pretrained network in `pretrained`. 
- `pretrained/task1_mmwhs.hdf5` is the pretrained network used in Task1, whole heart segmentation of the MMWHS dataset. 
- `pretrained/task2_wh.hdf5` is the pretrained network used in Task2, whole heart mesh generation with inlet vessel geometries for CFD simulations. 

The config files for both tasks are stored in `config`. The first task uses a template mesh without pulmonary veins and vena cava geometries and the second task uses another template mesh with those structures so that the predictions can be used for CHD simulations. Please make sure to use the correct template mesh depending on the task. The template mesh can be generated from the previous steps using the corresponding segmentation files. After changing the pathnames in the config files, you can use `predict.py` with the following arguments to generate predictions. 
```
python predict.py --config config/task2_lv_myo.yaml
```

Some notes about the config options:
- `--image`: the images should be stored under with in `<image>/ct<attr>`, thus for `--attr _test`, and `--modality ct` the image volumes should be in `image_dir_name/ct_test`. You can use `--modality ct mr' to predict both on CT and MR images where CT images are stored in `image_dir_name/ct_test` and MR images are stored in `image_dir_name/mr_test`.
- `--mesh_dat` is the `<date>_bbw.dat` file generated from running `create_template.sh` on the training template.
- `--swap_dat`is optional for providing the biharmonic coordinates corresponding to a modified template (e.g. the CFD-suitable template created from the training template). (TO-DO provide instructions on interpolating biharmonic coordinates to a modified template)
- `--seg_id` is the list of segmentation class IDs we expect the predictions to have
- The results contain deformed test template meshes from each deformation block. The final mesh from the last deformation block is `block_2_*.vtp`.

## Training

### Data Structure
The data preparation code is copied from the author's [MeshDeformNet](https://github.com/fkong7/MeshDeformNet.git) repository.

Ensure you have a directory structure as follows (e.g., for the __MMWHS__ dataset):
```
|-- MMWHS
    |-- nii
        |-- ct_train
            |-- 01.nii.gz
            |-- 02.nii.gz
            |-- ...
        |-- ct_train_seg
            |-- 01.nii.gz
            |-- 02.nii.gz
            |-- ...
        |-- ct_val
        |-- ct_val_seg
        |-- mr_train
        |-- mr_train_seg
        |-- mr_val
        |-- mr_val_seg
```

*I have all images and labels foreground cropped, resized, and padded to 128x128x128. Not sure what will happen if not doing so.*
### Data Augmentation

Data augmentation were applied on the training data. Specifically, we applied random scaling, random rotation, random shearing as well as elastic deformations. 

```
mpirun -n 20 python data/data_augmentation.py \
    --im_dir  /path/to/image/data \
    --seg_dir  /path/to/segmentation/data \
    --out_dir  /path/to/output \
    --modality ct \ # ct or mr
    --mode val \ # train or val
    --num 10 # number of augmented copies per image
```
### Data Pre-Processing

The data pre-processing script will apply intensity normalization and resize the image data. The pre-processed images and segmentation will be converted to .tfrecords.

```
python data/data2tfrecords.py --folder /path/to/top/image/directory \
    --modality ct mr \
    --size 128 128 128 \ # image dimension for training
    --folder_postfix _train \ # _train or _val, i.e. will process the images/segmentation in ct_train and ct_train_seg
    --deci_rate 0  \ # decimation rate on ground truth surface meshes
    --smooth_ite 50 \ # Laplacian smoothing on ground truth surface meshes
    --out_folder /path/to/output \
    --seg_id 1 2 3 4 5 6 7 # segmentation ids, 1-7 for seven cardiac structures here for example
```

### Compile nndistance Loss

If you do not see a `tf_nndistance_so.so` file in the `external/` directory, which is a required Python module compiled in C++ for training the network, compile the module by running the following command:
```
cd external/
make
cd ..
```
Please change the paths to the cuda and tf libraries in the Makefile to match with the locations on your system.

If you are getting <code style="color : red">/usr/bin/ld: cannot find -ltensorflow_framework</code>, you need to create a symbolic link. In my case, the file libtensorflow_framework.so.1 existed inside my TF_LIB directory instead of libtensorflow_framework.so. In order to solve this issue, I had to create a symbolic link as follows:

```
sudo ln -s /home/cm21/anaconda3/envs/deformnet37/lib/python3.7/site-packages/tensorflow_core/libtensorflow_framework.so.1 /usr/lib/libtensorflow_framework.so
sudo ldconfig
```

### Training
To train our network model, please run the following command.
```
python train.py --config config/task2_lv_myo.yaml
```

If you see at the start of training, 
<code style="color : darkorange">[W tensorflow/stream_executor/cuda/redzone_allocator.cc:312] Not found: ./bin/ptxas not found</code>

you can run in the terminal

```
export CUDA_HOME=/usr/local/cuda-10.1/
export PATH="${CUDA_HOME}/bin:$PATH" 
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```