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


# Tests (all run from root directory)
0) dataloader.py
- Expects input for frames under data/frames/{str_video}/{int_frame:6d}.png
- Expects input for annotations under data/annotations/{str_video}.csv in the form [STR_VIDEO,INT_FRAME,STR_CLASS,LST_SEGMENTATION]
0) utils.py 
- Expects video input data in data/video_metadata.csv in the form [STR_VIDEO,INT_START,INT_STOP,STR_SPLIT_ORIGINAL,STR_SPLIT]
- Expects instrument input in data/class_metadata.csv in the form [STR_CLASS,BL_INCLUDED,INT_CLASS,INT_TRAIN]
- Expects osats input in data/ostats_metadata.csv in the form [STR_VIDEO,RESPECT_FOR_TISSUE,TIME_AND_MOTION,INSTRUMENT_HANDLING,FLOW_OF_OPERATION,KNOWLEDGE_OF_INSTRUMENTS,KNOWLEDGE_OF_PROCEDURE,INT_TOTAL]

1) train_classification.py:
- Expected output in models/{int_model}_clf.ckpt containing the best performing model; logs/{int_model}/ containing tensorboard output; data/model_metadata.csv updated with {int_model} parameters (int_model defined as the next model in this .csv); data/results/classification.csv updated with evaluation results.

i. To test the simplest form of the script:

python scripts/train_classification.py --int_epochs 2 --int_epochs_freeze 1 --str_clf_backbone resnet50 

ii. To test multi-devices (i.e. 2 GPUs)

python scripts/train_classification.py --int_epochs 2 --int_epochs_freeze 1 --str_clf_backbone resnet50 --int_devices 2

iii. For multi-class training:

python scripts/train_classification.py --int_epochs 2 --int_epochs_freeze 1 --str_clf_backbone resnet50 --bl_multiclass

iv. For dinov2 training:

python scripts/train_classification.py --int_epochs 2 --int_epochs_freeze 1 --bl_multiclass --str_clf_backbone dinov2 --int_size 518

v. To train with the pre-trained moco:

python scripts/train_classification.py --int_epochs 2 --int_epochs_freeze 1 --bl_multiclass --str_clf_backbone moco

vi. To train the semi-supervised flexmatch model on-top of the backbone:

python scripts/train_classification.py --int_epochs 2 --int_epochs_freeze 1 --str_clf_backbone resnet50 --int_devices 2 --bl_semisupervised

vii. To train the lstm on-top of the backbone:

python scripts/train_classification.py --int_epochs 2 --int_epochs_freeze 1 --str_clf_backbone resnet50 --int_devices 2 --bl_lstm --int_clip 5

viii. To train flexmatch or lstm on an existing model:

python scripts/train_classification.py --int_epochs 2 --int_epochs_freeze 1 --str_clf_backbone resnet50 --int_devices 2 --bl_lstm --int_clip 5 --pth_backbone_checkpoint models/1_clf.ckpt

ix. Other boolean parameters (--bl_no_dropout; --bl_no_shuffle, --bl_no_weighted_sampler) can be added as necessary to change defaults.

x. Other usual hyperparameters (--flt_lr; --str_loss; --int_batch) can be changed as necesssary.

2) (a) train_cnnsegmentation.py
- Expected output in models/{int_model}_seg.ckpt containing the best performing model; logs/{int_model}/ containing tensorboard output; data/model_metadata.csv updated with {int_model} parameters (int_model defined as the next model in this .csv); data/results/segmentation.csv updated with evaluation results.

i. To test the simplest form of the script: 

python scripts/train_cnnsegmentation.py --int_epochs 2 --int_epochs_freeze 1 --str_architecture deeplabv3plus --str_encoder resnet50

ii. To test other functionality:

python scripts/train_cnnsegmentation.py --int_epochs 2 --int_epochs_freeze 1 --int_devices 2 --str_architecture deeplabv3plus --str_encoder dinov2 --pth_backbone_checkpoint data/models/1_clf.ckpt --bl_multiclass --int_size 518 --bl_no_shuffle --bl_no_weighted_sampler

iii. As with train_classification.py, can vary boolean parameters and hyperparmeters as needed.

2) (b) train_mmsegmentation.py

i. To test the simplest form of the script: 

python scripts/train_mmsegmentation.py --int_epochs 1

ii. To test other functionality:

python scripts/train_mmsegmentation.py --int_epochs 1 --str_architecture mask2former --str_encoder dinov2 --pth_backbone_checkpoint models/1_clf.ckpt --int_size 518 --flt_lr 1e-4 --bl_no_shuffle --bl_no_weighted_sampler

iii. As with train_classification.py, can vary boolean parameters and hyperparmeters as needed. (Note can only be run for a dinov2 backbone).

3) (a) run_models.py
- Expected outputs will be segmentation/{int_model}/{str_video}.csv of the form [STR_VIDEO,INT_FRAME,STR_CLASS_PRED,LST_SEGMENTATION_PRED] at 1fps

i. To test a trained base classifier:

python scripts/run_models.py --str_evaluator clf --pth_model models/1_clf.ckpt

ii. To test a trained semi-supervised classifier:

python scripts/run_models.py --str_evaluator flex --pth_model models/1_flex.ckpt

iii. To test a trained lstm classifier:

python scripts/run_models.py --str_evaluator lstm --pth_model models/1_lstm.ckpt --pth_lstm_backbone models/1_lstm_backbone.ckpt --int_clip 5

iv. To test a trained cnn-based segmentation model:

python scripts/run_models.py --str_evaluator cnnseg --pth_model models/1_seg.ckpt --str_splits all

v. To test a trained transfomer-based segmentation model:

python scripts/run_models.py --str_evaluator mmseg --pth_model models/1_seg.pth --str_mmseg_architecture mask2former --str_mmseg_encoder dinov2 --int_size 518

3) (b) run_smoothing.py
- Expected output found in data/results/segmentation.csv as evaluation metrics

i. The simplest way of running the script without a smoothing function:

python scripts/run_smoothing.py --int_clf 1 --int_seg 1

ii. To combine different classifiers and segmentation models:

python scripts/run_smoothing.py --int_clf 1 --int_seg 2

iii. The run online temporal smoothing:

--int_clf 1 --int_seg 1 --int_clip 7 

iv. The run offline temporal smoothing:

--int_clf 1 --int_seg 1 --int_clip 7 --bl_offline

4) run_tracker.py
- Expected outputs will be data/tracking/{int_model}/{str_video}.csv of the form [STR_VIDEO,INT_FRAME,STR_CLASS_PRED,LST_SEGMENTATION_PRED] at 25fps

i. To test the simplest form of the tracking:

python scripts/run_tracker.py --int_seg 1 --str_video data/videos/Antimony.mp4

ii. To test other SAM2 trackers:

python scripts/run_tracker.py --int_seg 1 --str_video data/videos/Antimony.mp4 --int_clip 5 --sam2checkpoint models/sam2_hiera_large.pt --sam2cfg sam2_hiera_l.yaml

5) run_kinematics.py
- Expected outputs will be data/kinematics/{int_model}/{str_video}.csv of the form [INT_CLASS,FLT_TIME,FLT_DISTANCE,FLT_SPEED,FLT_ACCELERATION]

i. To test the simplest form of kinematics calculations:

python scripts/run_kinematics.py --int_model 1

ii. To run with other parameters:

python scripts/run_kinematics.py --int_model 2 --str_split ss_val_test --int_size 720 --int_fps 25

6) train_osats.py
- Expected outputs will be data/osats/{int_model}/ with train model outputs (.joblib) and results (.csv)

i. There is only one form of the script:

python scripts/train_osats.py --int_model 1
