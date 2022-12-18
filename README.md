# Object Detection in the Urban Environment Project
This repository contains my project submission to the Object Detection in the Urban Environment project of the Udacity Data Scientist course.

## Project Summary

This project explores detecting objects in an urban environment annotating cyclists, pedestrians and vehicles, based on the Waymo dataset and the TensorFlow Object Detection API.

Detecting such objects accurately and without error is a crucial task in self-driving car scenarios, since seeing and placing them correctly is the basis for taking correct controlling decisions, with potentially catastrophic consequences in case of errors. 

The code is basically structured into the following parts:

1. An exploratory data analysis has been done using the notebook `Exploratory Data Analysis.ipynb`. See [discussion below](#dataset-analysis) for the results.

1. Models can be trained using a set of config files, and bash and python scripts. 

1. An exploratory augmentation notebook visualizes the configured augmentations.

1. Models can be exported and animations can be rendered using these stored models.

This repository has both the necessary code to reproduce these steps in the Udacity classroom workspace, as well as [a detailed discussion of their previous execution at the end of this file](#dataset).

## Repository Structure
The repository is structured as follows:

```bash
ds-object-detection-urban-environment
├── experiments
│   ├── exporter_main_v2.py           # Script to create an inference model.
│   ├── label_map.pbtxt
│   ├── model_main_tf2.py             # Script to to launch training.
│   ├── experiment-1
│   │   └── pipeline_new.config       # Experiment configuration.
│   ├── experiment-2
│   │   └── pipeline_new.config       # Experiment configuration.
│   ├── experiment-3
│   │   └── pipeline_new.config       # Experiment configuration.
│   ├── experiment-4
│   │   └── pipeline_new.config       # Experiment configuration.
│   ├── pretrained_model              
│   └── reference                     # Target directory for trained reference model.
│       └── pipeline_new.config       # Experiment configuration.
├── visualizations
│   └── ...                           # Charts, images, etc. 
├── .gitignore                        # Gitignore file. 
├── download_pretrained_model.sh      # Bash script to dowload and unpack pre-trained model.
├── edit_config.py                    # Script to update experiment config.
├── Exploratory Data Analysis.ipynb   # Notebook with exporatory data analysis.
├── Explore augmentations.ipynb       # Notebook with augmentations exploration.
├── filenames.txt
├── inference_video.py                # Python script to generate animations from model run.
├── label_map.pbtxt                 
├── requirements.txt                  # List of imported Python packages.
├── train_reference_model.sh          # Bash script to train reference model.
├── update_udacity_workspace.sh       # Bash script to update Udacity workspace.
├── utils.py                          # Helper functions.
└── README.md                         # This file.
```

The contents of this repository have been partially built on the template files in the [Udacity Computer Vision Starter GitHub repository](https://github.com/udacity/nd013-c1-vision-starter).

## Usage

### Prerequisites
This project is ready to be executed in the Udacity classroom workspace environment, which has the readily split Waymo data already downloaded and stored in subfolders under `/home/workspace/data`. To download and split the data yourself, please refer to the according steps in the [Computer Vision Starter github project](https://github.com/udacity/nd013-c1-vision-starter).

The Udacity classroom workspace uses Python 3.7.3 and uses the following packages (list generated with `pipreqs`):

```
absl_py==0.10.0
matplotlib==3.4.1
numpy==1.18.5
object_detection==0.1
protobuf==4.21.12
tensorflow==2.4.1
waymo_open_dataset==1.0.1
waymo_open_dataset_tf_2_3_0==1.3.0
```

The following command will install these packages and their dependencies according to the configuration file `requirements.txt`. The file was generated using a combination of `pipreqs` and `pip-compile` from `piptools`.

```
$ pip install -r requirements.txt
```

In the classroom workspace, every library and package should already be installed in your environment, so installing the requirements should not be necessary.

### Installation

1. Checkout this repository. 

    ```
    $ git clone https://github.com/budde/ds-object-detection-urban-environment.git
    ```

1. Copy its contents into the folder `/home/workspace` in the Udacity classroom environment, overwriting existing files and folders by the same name.

    ```
    $ /bin/cp -rf ds-object-detection-urban-environment/* /home/workspace/
    ```
    
    :warning: The above command will overwrite any existing file or folder by the same name **without prompting for confirmation**, so make sure that is what you want and/or backup the folder first.


1. To remedy potential stability issues when running the Jupyter notebooks in firefox, this repository includes a bash script that will update the installed firefox. 

    ```
    $ ./update_udacity_workspace.sh
    ```

    :warning: This will only work in a GPU enabled workspace, as running the desktop is not possible otherwise at all. 

    This step is not needed when running this project locally outside of the Udacity classroom workspace.

In case this project should be run locally instead of in the Udacity classroom workspace, the steps are very similar: Make sure to also clone this repository into a folder `/home/workspace`, then additionally download, transform and split the Waymo data according to the steps described in the [Computer Vision Starter github project](https://github.com/udacity/nd013-c1-vision-starter).

### Execution

After copying the files from this repository into the Udacity classroom workspace, the three parts of the script can be executed as follows:

#### Running the Exploratory Data Analysis notebook

When all the prerequisites are met, the Jupyter notebook can simply be started as usual:

```
$ jupyter notebook
```

However, since in the Udacity classroom desktop workspace this will result in an error, instead run 

```
$ ./launch_jupyter.sh
```

This should launch a browser containing your notebooks. If the browser doesn't launch on its own, just open a browser window and open up the URL shown in the output of the respective command above.

#### Training and evaluating models

1. Creating and running the reference model

    First, the pre-trained SSD Resnet 50 640x640 model needs to be downloaded and the configuration edited. To facilitate this for the reference model, the according steps have been grouped into a bash script:

    ```
    $ ./download_pretrained_model.sh
    ```

    Subsequently, the training can be started:

    ```
    $ ./train_reference_model.sh
    ```
    This will write the output of the script to `train_reference_model.log`. To monitor the training and/or evaluation of the reference model graphically, a tensorboard instance can be launched:

    ```
    python -m tensorboard.main --logdir experiments/reference/
    ```

    The findings of the monitoring of the reference model training can be found [in the discussion below](#reference-experiment).

1. Training other models

    Generally, to start the training for any of the experiments, execute

    ```
    python experiments/model_main_tf2.py --model_dir=experiments/<experiment-folder>/ --pipeline_config_path=experiments/<experiment-folder>/pipeline_new.config
    ```
     substituting `<experiment-folder>` with the appropriate paths.
 
    :warning: **Important note:** The disc space in the Udacity classroom environment is severely limited and it is likely that it will not accommodate the full training data before filling up! Should this be the case, you can prolongue this by deleting all checkpoint files **prior to the last** while the **training is still ongoing**. 

    Example: if the most recent checkpoint were 27, you can delete the previous one (and the ones before) without affecting the training. 

    ```
    $ cd /home/workspace/experiments/experiment-2 
    $ rm ckpt-26.data-00000-of-00001
    $ rm ckpt-26.index
    ```

    :warning: **Important note:** However, for large models such as the one trained in experiment 2, **the available space may still not suffice**! To remedy this, either train the models outside of the Udacity classroom workspace or move it to another partition. The `/home/backup` path e.g. has much more space available. Be advised though that (ironically) this is **not backed** up, so if your session disconnects, the **progress will be lost**!

1. Model evaluation

    Once the training is finished, launch the evaluation process:
    ```
    python experiments/model_main_tf2.py --model_dir=experiments/<experiment-folder>/ --pipeline_config_path=experiments/<experiment-folder>/pipeline_new.config --checkpoint_dir=experiments/<experiment-folder>/
    ```
     again substituting `<experiment-folder>` with the appropriate paths.

    
    :warning: Both training and evaluation may display some Tensorflow warnings, which can be ignored. You may have to kill the evaluation script manually using
    <kbd>CTRL</kbd>+<kbd>C</kbd>.
    
    To monitor the training, you can launch a tensorboard instance by running `python -m tensorboard.main --logdir experiments/<experiment-folder>/`. 

#### Exporting and creating animations

1. Export the final model

    In case you have added experiments beyond the ones configured in the repository, modify the arguments of the following function to adjust it to your final model:
    
    ```
    python experiments/exporter_main_v2.py --input_type image_tensor --pipeline_config_path experiments/experiment-2/pipeline_new.config --trained_checkpoint_dir experiments/experiment-2/ --output_directory experiments/experiment-2/exported/
    ```
    This should create a new folder `experiments/experiment-2/exported/saved_model`. You can read more about the Tensorflow SavedModel format [here](https://www.tensorflow.org/guide/saved_model).
    
1. Creating an animation
    Finally, you can create a video of the final model's inferences for any tf record file. To do so, run the following command (if necessary, adjust file paths):
    ```
    python inference_video.py --labelmap_path label_map.pbtxt --model_path experiments/experiment-2/exported/saved_model --tf_record_path /data/test/segment-12200383401366682847_2552_140_2572_140_with_camera_labels.tfrecord --config_path experiments/experiment-2/pipeline_new.config --output_path animation.gif
    ```
    The animation generated based on the final model is displayed [at the end of the discussion below](#animation-of-final-model-performance).

## Dataset
### Dataset analysis
The dataset features point-of-view images from cars. The images are accompanied by ground truth data which contains bounding boxes for three classes of objects (cars, pedestrians, and cyclists). The following example image shows a highway traffic situation. The cars are highlighted by red bounding boxes:

![png](visualizations/sample_highway.png)


While the images all have the same dimensions and resolution, they are extremely heterogeneous in terms of lighting (day, night), quality (blur, lens flares, etc.), weather conditions (sunshine, rain, fog, etc.), as well as the type and number of objects visible in any given image, their density/line-of-sight (clear, partially occluded, cut off by edge of image), and size (close/large, far/small). The following image illustrates this contrast when compared to the one above, as it was taken at nighttime and features pedestrians (blue bounding boxes) in addition to (parking) cars:

![png](visualizations/sample_night.png)

In addition to these qualitative differences, the dataset is also uneven in terms of the class distribution. In a sample of 50,000 images, the vast majority either featured only cars or cars and pedestrians. The third largest class (well below 10%) had at least one object from each class. Images with only pedestrians, only cyclists or both without cars are very seldom within the dataset, and even combined lie below the number of images without any objects at all (~1%):


![png](visualizations/diagram_class_distribution.png)

This is also reflected when looking at the absolute number of objects in the same sample. Over the same 50,000 images, we counted well in excess of 800,000 cars (~17 per image on average), a little more than 200,000 pedestrians (~4 per image on average) and in comparison practically no cyclists.

![png](visualizations/diagram_sample_count_total.png)


## Training
### Reference experiment
The reference model was trained according to the example instructions above, which initiate the model with the following parameters:

| Parameter         | Setting                                        |
|-------------------|------------------------------------------------|
| Pre-trained model | SD Resnet 50 640x640                           |
| Batch size        | 2                                              |
| Training steps    | 2,500                                          | 
| Augmentations     | Random horizontal flip <br/> Random crop image |
| Optimizer         | Momentum optimizer (value 0.9)                 |
| Learning rate     | Cosine decay (base: 0.04, warmup rate: 0.013333, warmup steps: 200) | 
  
When looking at the results of the reference training experiment, it becomes clear that the algorithm's performance is very poor:

![png](visualizations/tensorboard_reference_training_ignore_outliers.png)

The classification loss fluctuates strongly and goes up and down, but plateaus at a high level. The total loss does not drop below ~20. The reference model does not seem to have converged with the number of epochs. Therefore, an increase of the number of steps seems to be a first sensible direction to explore.

### Improve on the reference
This section details the different attempts made to improve the model. 

#### Experiment-1

In the first experiment, the only change to the reference model was to increase the number of training steps to 10,000, resulting in the following parameters:

| Parameter         | Setting                                        |
|-------------------|------------------------------------------------|
| Pre-trained model | SD Resnet 50 640x640                           |
| Batch size        | 2                                              |
| Training steps    | 10,000                                         | 
| Augmentations     | Random horizontal flip <br/> Random crop image |
| Optimizer         | Momentum optimizer (value 0.9)                 |
| Learning rate     | Cosine decay (base: 0.04, warmup rate: 0.013333, warmup steps: 200) | 
  
When looking at the results of this experiment, it becomes clear that the model still does not seem to have converged. As a result, further increase of the number of epochs is warranted. Training and evaluation loss are on par, indicating that overfitting is not an issue here.

Additionally, we can observe a jump in the loss after approximately 500 steps and again at ca. 2.5k steps and a subsequent decline plateauing on a higher level. While it is not completely evident what caused this, one explanation may be that due to a too large learning rate, the training jumped out and got stuck in a local minimum. We will therefore investigate a lower initial learning rate.

![png](visualizations/tensorboard_experiment-1_training_ignore_outliers.png)

Finally, the performance of the model is poor. The mean average precision rests at 0, mean average recall for large and medium objects with 100 detections (`AR@100`) is around 0.1, for small objects it is close to 0.

![png](visualizations/tensorboard_experiment-1_eval_precision.png)
![png](visualizations/tensorboard_experiment-1_eval_recall.png)

#### Experiment-2 and experiment-3

In the second and third experiment, the number of epochs was further increased. Experiment 2 was evaluated after ~15,000, experiment 3 after 25,000 epochs. All other modifications were identical in both runs: The batch size was increased to 4, and additional augmentations were introduced according to the tests in `Explore augmentations.ipynb`. The learning rate was also adjusted to a lower setting:

| Parameter         | Setting                                        |
|-------------------|------------------------------------------------|
| Pre-trained model | SD Resnet 50 640x640                           |
| Batch size        | 4                                              |
| Training steps    | 15,000 (experiment-2) resp. 25,000 (experiment-3) | 
| Augmentations     | Random horizontal flip <br/> Random crop image <br/> [Random adjust brightness](https://www.tensorflow.org/api_docs/python/tf/image/random_brightness) <br/> [Random adjust contrast](https://www.tensorflow.org/api_docs/python/tf/image/random_contrast) <br/> [Random adjust hue](https://www.tensorflow.org/api_docs/python/tf/image/random_hue) <br/> [Random adjust saturation](https://www.tensorflow.org/api_docs/python/tf/image/random_saturation) <br/> [Random rgb to gray](https://www.tensorflow.org/api_docs/python/tf/image/rgb_to_grayscale) <br/> |
| Optimizer         | Momentum optimizer (value 0.9)                 |
| Learning rate     | Cosine decay (base: 0.01, warmup rate: 0.004, warmup steps: 200) | 
 
##### Data augmentations
Samples of the augmentations that were applied here can be seen below. In addition to the random horizontal flip and crop, random modifications of [brightness](https://www.tensorflow.org/api_docs/python/tf/image/random_brightness), [saturation](https://www.tensorflow.org/api_docs/python/tf/image/random_saturation), [hue](https://www.tensorflow.org/api_docs/python/tf/image/random_hue) and [contrast](https://www.tensorflow.org/api_docs/python/tf/image/random_contrast) were introduced and 10% of the images were randomly [converted to grayscale](https://www.tensorflow.org/api_docs/python/tf/image/rgb_to_grayscale):

```
data_augmentation_options {
  random_horizontal_flip {
  }
}
data_augmentation_options {
  random_crop_image {
    min_object_covered: 0.0
    min_aspect_ratio: 0.75
    max_aspect_ratio: 3.0
    min_area: 0.75
    max_area: 1.0
    overlap_thresh: 0.0
  }
}
data_augmentation_options {
  random_adjust_brightness {
    max_delta: 0.2
  }
}
data_augmentation_options {
  random_adjust_contrast {
    min_delta: 0.8
    max_delta: 1.25
  }
}
data_augmentation_options {
  random_adjust_hue {
    max_delta: 0.02
  }
}
data_augmentation_options {
  random_adjust_saturation {
    min_delta: 0.8
    max_delta: 1.25
  }
}
data_augmentation_options {
  random_rgb_to_gray {
    probability: 0.1
  }
}
```

These augmentations were chosen to alleviate the diversity of lighting and environmental conditions in the training data and better represent more seldom conditions in the data.

The following images were generated using the code in `Explore augmentations.ipynb`:


![png](visualizations/sample_augmentation_1.png)
![png](visualizations/sample_augmentation_5.png)
![png](visualizations/sample_augmentation_7.png)
![png](visualizations/sample_augmentation_4.png)
![png](visualizations/sample_augmentation_3.png)
![png](visualizations/sample_augmentation_6.png)
![png](visualizations/sample_augmentation_2.png)

##### Model performance

With these modifications, the performance of the model significantly increased.

After ~15,000 epochs, there was already a clear improvement visible.

![png](visualizations/tensorboard_experiment-2_training_ignore_outliers.png)

There is some discrepancy between the training and validation losses though. This may either indicate a bit of overfitting, a generally not too well balanced split between training and validation, or (since we only did 1 evaluation epoch).

![png](visualizations/tensorboard_experiment-2_eval_precision.png)

The overall mean average precision (mAP) has risen from practically zero in the first experiment to 0.14. This, however, is strongly different for the different size classes of objects: the best value is exhibited for large objects (~0.58), medium objects are detected slightly worse (~0.48), and small objects detection is much worse (0.07). 

![png](visualizations/tensorboard_experiment-2_eval_recall.png)

This tendency is the same for recall: Average recall (`AR@100`) is best for large objects (0.66) and a little worse for medium ones (0.54). Small object AR is comparatively small, with a value of 0.14.

##### Further improvements

An idea to improve the model further, especially towards detecting small objects more accurately, is employing the `random_crop_pad_image` augmentation:

```
  data_augmentation_options {
    random_crop_pad_image {
    }
  }
```

The augmentation results are shown below:

![png](visualizations/sample_augmentation_8.png)
![png](visualizations/sample_augmentation_9.png)
![png](visualizations/sample_augmentation_10.png)
![png](visualizations/sample_augmentation_11.png)

Unfortunately, due to stability issues with the classroom environment, I could not test this approach successfully with the allotted GPU hours.

The configuration to do this is saved as `experiment-4` though within this repository.
