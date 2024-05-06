## File structure

```
data/
  sensorium/
    dynamic29156-11-10-Video-8744edeac3b4d1ce16b680916b5267ce.zip
    dynamic29228-2-10-Video-8744edeac3b4d1ce16b680916b5267ce.zip
    ...
  README.md
```
### Sensorium 2023 datasets
- Original dataset [gin.g-node.org/pollytur/Sensorium2023Data](https://gin.g-node.org/pollytur/Sensorium2023Data) with Mouse A, B, C, D, E.
  ```
  dynamic29156-11-10-Video-8744edeac3b4d1ce16b680916b5267ce.zip # mouse A
  dynamic29228-2-10-Video-8744edeac3b4d1ce16b680916b5267ce.zip # mouse B
  dynamic29234-6-9-Video-8744edeac3b4d1ce16b680916b5267ce.zip # mouse C
  dynamic29513-3-5-Video-8744edeac3b4d1ce16b680916b5267ce.zip # mouse D
  dynamic29514-2-9-Video-8744edeac3b4d1ce16b680916b5267ce.zip # mouse E
  ```
- New dataset [gin.g-node.org/pollytur/sensorium_2023_dataset](https://gin.g-node.org/pollytur/sensorium_2023_dataset) with Mouse F, G, H, I, J.
  ```
  dynamic29515-10-12-Video-9b4f6a1a067fe51e15306b9628efea20.zip # mouse F
  dynamic29623-4-9-Video-9b4f6a1a067fe51e15306b9628efea20.zip # mouse G
  dynamic29647-19-8-Video-9b4f6a1a067fe51e15306b9628efea20.zip # mouse H
  dynamic29712-5-9-Video-9b4f6a1a067fe51e15306b9628efea20.zip # mouse I
  dynamic29755-2-8-Video-9b4f6a1a067fe51e15306b9628efea20.zip # mouse J
  ```


### Data information
- See [gin.g-node.org/pollytur/Sensorium2023Data/src/master/README.md](https://gin.g-node.org/pollytur/Sensorium2023Data/src/master/README.md)
- Each zip file consists of two folders `data/` and `meta/`.
- `data/`: includes the variables that were recorded during the experiment. The experimental variables are saved as a collection of numpy arrays. Each numpy array contains the value of that variable at a specific image presentation (i.e. trial). Note that the name of the files does not contain any information about the order or time at which the trials took place in experimental time. They are randomly ordered.
  - `videos`: This directory contains NumPy arrays where each single `X.npy` contains the video that was shown to the mouse in trial X.
  - `responses`: This directory contains NumPy arrays where each single `X.npy` contains the deconvolved calcium traces (i.e. responses) recorded from the mouse in trial X in response to the particular presented image.
  - `behavior`: Behavioral variables include pupil dilation and running speed. The directory contain NumPy arrays (of size 1 x 2) where each single `X.npy` contains the behavioral variables (in the same order that was mentioned earlier) for trial X.
  - `pupil_center`: the eye position of the mouse, estimated as the center of the pupil. The directory contain NumPy arrays (of size 1 x 2) for horizontal and vertical eye positions.
- `meta/`: includes metadata of the experiment
  - `neurons`: This directory contains neuron-specific information. Below are a list of important variables in this directory
    - `cell_motor_coordinates.npy`: contains the position (x, y, z) of each neuron in the cortex, given in microns. Note: The
  - `statistics`: This directory contains statistics (i.e. mean, median, etc.) of the experimental variables (i.e. behavior, images, pupil_center, and responses). 
    Note: The statistics of the responses are or particular importance, because we provide the deconvolved calcium traces here in the responses. 
    However, for the evaluation of submissions in the competition, we require the responses to be standardized (i.e. `r = r/(std_r)`).
  - `trials`: This directory contains trial-specific meta data.
    - `tiers.npy`: contains labels that are used to split the data into train, validation, and test set
      - The training and validation split is only present for convenience, and is used by our ready-to-use PyTorch DataLoaders.
      - The test set is used to evaluate the model performance. In the competition datasets, the responses to all test images is withheld.
