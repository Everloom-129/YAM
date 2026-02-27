# YAM 
Teleoperation, Data Collection, and model evaluation on Bimanual YAM.  

# Motor Configuration
For long time bimanual teleop, data collection, or evaluation, the default timeout is too short and often causes abrupt collapse. To prevent that, we turn off the motor timeout for both arms.
```
python i2rt/i2rt/motor_config_tool/set_timeout.py --channel can_left
python i2rt/i2rt/motor_config_tool/set_timeout.py --channel can_right
```

## Gello Configuration
Everything for gello is located in gello_software.
```
cd gello_software
```

Upon reconnecting the YAM to the PC, make sure to reset the CAN.
```
bash scripts/reset_all_can.sh
```
Configuration of the left arm is in ```configs/yam_left.yaml``` and configuration of the right arm is in ```configs/yam_right.yaml```.


### Teleoperation
To perform teleoperation, simply run
```
python experiments/launch_yaml.py --left_config_path=configs/yam_left.yaml --right_config_path=configs/yam_right.yaml
```

### Data Collection
To perform data collection, we need to change the content of the configuration file ```configs/yam_left.yaml```.

Goto ```yam_left.yaml``` and you will see sections called ```storage``` and ```lerobot```:
```bash
# Data storage configuration
storage:
  episodes: 30
  base_dir: "/home/sean/Desktop/YAM/gello_software/data_recurrent"
  task_directory: "test"
  language_instruction: "test"
  teleop_device: "oculus" # ["oculus", "keyboard", "gello", "none"]
  save_format: "json" # ["json", "npy"]
  old_format: false

# LeRobot conversion + upload pipeline
lerobot:
  auto_convert_and_upload: true
  hf_repo_id: "your_huggingface_user/your_dataset_name"
  delete_local_after_upload: true
  fps: 30
  robot_type: "molmoact_dual_arm"
  skip_initial_frames: 0
  action_mode: "next_joint_fields" # ["next_joint_fields", "next_state", "copy_state"]
  sanitize_online_viz_meta: true
```
```storage.episodes``` is the maximum episode index to collect. The collection loop ends when this limit is reached.

```storage.base_dir``` is the location to store collected raw json episodes.

```storage.task_directory``` is the subdirectory name for that task.

```storage.language_instruction``` is the instruction written into collected data.

```lerobot.auto_convert_and_upload``` enables post-collection automation:
1. convert json to LeRobot v3.0
2. upload to Hugging Face
3. add dataset tag (`v3.0`) via ```add_tag.py```
4. optionally delete local json + lerobot data if ```delete_local_after_upload``` is true

```lerobot.hf_repo_id``` is the destination Hugging Face dataset repo in the form ```username/dataset_name```.

```lerobot.delete_local_after_upload``` controls whether local raw json and local LeRobot output are deleted after successful upload and tagging.

```lerobot.fps``` is the frame rate metadata written into the generated LeRobot dataset (set this to match your collection/control frequency).

```lerobot.robot_type``` sets the robot metadata field saved in the LeRobot dataset.

```lerobot.skip_initial_frames``` skips the first N frames of each episode during conversion (useful to remove startup transients).

```lerobot.action_mode``` controls how action is derived:
- ```next_joint_fields``` (recommended): use ```next_left_joint```/```next_right_joint``` from json.
- ```next_state```: use shifted joint state at t+1 as action.
- ```copy_state```: use current joint state at t as action.

```lerobot.sanitize_online_viz_meta``` removes quantile-only metadata columns after conversion to improve compatibility with some online visualizers.

To perform data collection after configuration simply run:
```bash
python experiments/launch_yaml_collect_data.py --left_config_path=configs/yam_left.yaml --right_config_path=configs/yam_right.yaml
```

The program will launch a color pad to take keyboard input. 

Press ```s``` to start collecting 1 episode of data.

Press ```a``` to end and save collected episode.

Press ```b``` to end and delete collected episode.

After all episodes are collected, the script automatically runs conversion/upload/tagging.
If the LeRobot output directory already exists, it will ask:
```bash
Do you want to remove it and continue? (y/n)
```
Type ```y``` to remove and continue, or ```n``` to cancel conversion/upload.

Important: pressing ```ctrl+c``` exits early and only performs robot/socket cleanup.  
It does **not** run convert/upload/tag pipeline.

Note: make sure you are on the color pad so it can take in the keyboard input (don't put it in the background).  
To kill the program with ```ctrl+c```, you will need to be on your IDE or Terminal.

### Data Converstion
Manual conversion is still available if needed. Data is saved in json format and can be converted with ```molmoact_to_lerobot_v30.py```:
```
python molmoact_to_lerobot_v30.py \
        --data_dir /path/to/molmoact \
        --output_dir /path/to/molmoact_lerobot_v30 \
        --repo_id your_huggingface_user/molmoact_v30 \
        --fps 10
```
After successful conversion, upload to Hugging Face:
```
hf upload huggingface_user/dataset_name /path/to/local_lerobot_v30_dataset --repo-type=dataset
```
Then add tag:
```
python add_tag.py --repo_id huggingface_user/dataset_name
```

### Model Evaluation
Current evaluation supports two policy types in ```experiments/launch_yaml_eval.py```:
- ```dp``` (DiffusionPolicy)
- ```pi05``` (PI05Policy)

Set these fields in ```configs/yam_left.yaml``` under ```policy```:
```bash
# Policy configuration
policy:
  type: "dp"   # ["dp", "pi05"]
  repo_id: "your_hf_user/your_dataset_or_local_dataset_path"
  checkpoint_path: "your_model_repo_or_local_checkpoint_path"
```

```policy.type``` selects which evaluator path is used:
- ```dp```: runs ```run_control_loop_eval``` with diffusion policy.
- ```pi05```: runs ```run_control_loop_eval_pi``` with PI05 chunked actions.

```policy.repo_id``` is used to load dataset metadata/statistics (HF dataset id or local dataset path).

```policy.checkpoint_path``` is the policy checkpoint source (HF model id or local checkpoint path).

For ```pi05```, task text is taken from ```storage.language_instruction```.

After configuring policy, run:
```bash
python experiments/launch_yaml_eval.py --left_config_path=configs/yam_left.yaml --right_config_path=configs/yam_right.yaml
```

Notes: ```preprocess_observation``` in ```experiments/launch_yaml_eval.py``` converts robot observations to model inputs for the ```dp``` path. Make sure image size/camera mapping matches your model's expected input format.
```bash
# Define the target image size
TARGET_HEIGHT = 256
TARGET_WIDTH = 342

# Map cameras "observation : model"
camera_mapping = {"left_camera_rgb": 'left', "right_camera_rgb": 'right', "front_camera_rgb": 'front'}
```


