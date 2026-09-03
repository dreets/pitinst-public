# Summary
The purpose of this repo is to develop models for instrument classification, segmentation, and tracking during the sella phase of endoscopic pituitary surgery, for the purposes of predicting OSATS (Objective Structured Assessment of Technical Skills).

# Research
- The code in this repository is written by Adrito Das, a Honorary Research Fellow of University College London (UCL) at the UCL Hawkes Institute: https://www.ucl.ac.uk/hawkes-institute.
- For related work see: https://scholar.google.com/citations?user=F1w5uDkAAAAJ&hl=en&authuser=1
- For personal details see: https://sites.google.com/view/adrito-das
- If you spot anything wrong, or have any queries, please email: adrito(dot)das(dot)20(at)ucl(dot)ac(dot)uk

# Example 
Below is an example output clip (10s as a gif).

![video_output](sample/video_output.gif)

# License
This project is dual-licensed under both the MIT License and the Apache License (Version 2.0). You may choose which license you want to use the software under. See LICENSE.md for details on this and licenses of software used in this repositry.

# Citation
If you use this software in your research, please cite it as follows:
## BibTeX
```bibtex
@software{Das_PitInst_2026,
  author       = {Das, Adrito},
  title        = {PitInst},
  month        = {Sep},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{[https://github.com/dreets/pitinst-public](https://github.com/dreets/pitinst-public)}}
}
```

# Data
- Public data is available at: (placeholder)
- All video data was collected from the National Hospital for Neurology and Neurosurgery, London, WC1N 3BG.
- Instrument annotations were performed by medical trainees and surgical trainees, with verficiation from surgical consultants. This was done using the commercially available Encord annotation software (https://app.encord.com/).
- OSATS annotations were performed by surgical consultants.

# Environment
- The code is written for Python 3.12.0 with CUDA 12.1
- Training neural networks use PyTorch Lightning and MMseg.
- All executable code can be run via command-line (via argparse), and examples are given at the top of each script.
- A docker is provided to run the entire pipeline, to predict OSATS for a given video using the trained models (see docker/).

# Coding standards
The following modern Python (3.10+) coding standards shall be adhered to:
- PEP 8 (Style Guide): Follow consistent naming, spacing, import ordering, and layout rules to keep code readable and maintainable.
- PEP 257 (Docstrings): Write clear docstrings for public modules, classes, and functions, including purpose, parameters, return values, and exceptions.
- Type Hints (PEP 484): Use type annotations in function signatures and key variables to improve correctness, readability, and tooling support. Hugarian typing is used where appropriate.
- Modern Type Syntax (PEP 585 and PEP 604): Prefer built-in generics (for example, list[str], dict[str, int]) and union syntax (for example, str | None).
- PEP 621 (Project Metadata in pyproject.toml): Keep project/package metadata in pyproject.toml as the modern packaging standard.
- Testing Standard (pytest): Write focused unit tests with clear Arrange-Act-Assert structure, including edge cases and failure paths.
- Security and Robustness: Validate all external inputs, handle exceptions explicitly, avoid silent failures, and never commit secrets.

# Directories
The directory structure is as follows:
```
root/
|- data/
|  |- class_metadata.csv: contains the name of each instrument, and whether it is included in classification
|  |- model_metadata.csv: contains the parameters of the model once training has started
|  |- ostats_metadata.csv: contains the OSATS score for each video (placeholder)
|  |- video_metadata.csv: contains the video name, length, and split (placeholder)
|  |- annotations/: contains the video annotations as .csv (placeholder)
|  |- frames/: contains the de-framed videos at 1fps (placeholder)
|  |- osats/: contains the final osats predictions as .csv (placeholder)
|  |- results/: contains the classification & segmentation model evaluations after training (placeholder)
|  |- segmentation/: contains the predicted video annotations as .csv (1fps, placeholder)
|  |- tracking/: contains the predicted video annotations as .csv (25fps, placeholder)
|  \- videos/: contains the videos (placeholder)
|- docker/
|  |- inputs/: contains the video to run the models on (sample video provided)
|  |- models/: contains the models (download from: https://huggingface.co/drdreets/PitInst)
|  |- outputs/: contains the predictions as a .csv and .mp4 overlay
|- logs/: contains the tensorboard logs of the models during training
|- models/: contains the models during training
|- sample/: contains sample video input and output (10s video)
\- scripts/: contains all scripts for model training and running (see below)
```

# Scripts
The scripts under scripts/ are used to train and run all models. They are intended to be run in the following order:
0) dataloaders.py: dataloading functions for model training
0) utils.py: general utility file containing hard-coded paths, variables, and helper functions
1) train_classification.py: train the backbone classification model (ResNet50, DinoV2, etc)
2) (a) train_cnnsegmentation.py: train the segmentation model based on segmentation_models_pytorch (models_classification.py)
2) (b) train_mmsegmentation.py: train the segmentation model based on mmengine (models_segmentation.py)
3) (a) run_models.py: run the chosen trained model on a video to output the predictions as a .csv
3) (b) run_smoothing.py: run temporal smoothing on a .csv output, used primarily for evaluation purposes (can run with 0 smoothing)
4) run_tracker.py: run SAM2 tracking on the run_models.py .csv predictions
5) (a) run_kinematics.py: run kinematics calculations for a given model and save the .csv output
5) (b) train_osats.py: train osats models (basic regression) on the kinematics and save the .csv output
![schematic diagram](schematics.png)
tests.md contains commands and outputs to ensure each component of the training and running of the models is correct.