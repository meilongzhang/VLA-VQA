<h1 align="center">
ChatVLA: Unified Multimodal Understanding and Robot Control
with Vision-Language-Action Model
</h1>

![](./doc/chatvla_framework.png)

* **ChatVLA: Unified Multimodal Understanding and Robot Control with Vision-Language-Action Model** <br>
  [![arXiv](https://img.shields.io/badge/Arxiv-2502.05855-b31b1b.svg?logo=arXiv)](https://arxiv.org/pdf/2502.14420)

  


## 📰 News
* **`May 29, 2025`**: We are excited to announce the release of **ChatVLA-2**! The **paper** is available [here](https://arxiv.org/abs/2505.21906) and the **project website** can be accessed [here](https://chatvla-2.github.io/).
* **`Feb 20, 2025`**: We are excited to announce the release of **ChatVLA**! The **paper** is available [here](https://arxiv.org/pdf/2502.14420) and the **project website** can be accessed [here](https://chatvla.github.io/).

## Contents
- [Install](#install)
- [Download Pretrained VLM](#Download-Pretrained-VLM)
- [Data Preparation](#data-preparation)
- [Training](#train)
- [Evaluation](#evaluation)
- [ChatVLA Weights](#chatvla-weights)

## Install

1. Clone this repository and navigate to ChatVLA folder
```bash
git clone https://github.com/tutujingyugang1/ChatVLA_public.git
```

2. Install Packages
```Shell
conda create -n chatvla python=3.10 -y
conda activate chatvla
pip install --upgrade pip  # 
pip install -r requirements.txt
cd policy_heads
pip install -e .
```
For training acceleration, please install [flash_attention](https://github.com/Dao-AILab/flash-attention).
```shell
pip install flash-attn --no-build-isolation
```

3. For evaluation on multimodal understanding task, you should install other packages in [here](https://github.com/tutujingyugang1/ChatVLA_public/blob/main/evaluate/VLMEvalKit/requirements.txt). 

## Download  Qwen2_VL Weights

We construct the VLM backbone by integrating Qwen2-VL-2B, a powerful and efficient model, into our framework. 
The Qwen2-VL 2B serves as the core of our architecture, providing robust capabilities 
for vision-language tasks. We use off-the-shelf Qwen2-VL model proposed 
in [Qwen2-VL](https://arxiv.org/pdf/2409.12191) without any post training on VLM itself. You can download the official weights from this link:

| Model               | Link                                                           |
|---------------------|----------------------------------------------------------------|
| Qwen2-VL (~2B)      | [huggingface](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) |

**❗❗** After downloading the standard weights, you have to replace the official "config.json"
with our "doc/config.json" designed for VLA.

## Data Preparation
1. Our data format is the same as [DexVLA](https://github.com/juruobenruo/DexVLA), you should transfer your data into h5py format. More example data and preparation detail can refer to it.

2. Download llava_v1_5_mix665k dataset from [here](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json) or use your own Vision-Language Data using LLaVA-format.
```python
    [
        {
            "id": "000000033471",
            "image": "coco/train2017/000000033471.jpg",
            "conversations": [
            {
                "from": "human",
                "value": "<image>\nWhat are the colors of the bus in the image?"
            },
            {
                "from": "gpt",
                "value": "The bus in the image is white and red."
            }
            ]
        },
    ]
```


3. Add entries in [constants.py](https://github.com/tutujingyugang1/ChatVLA_public/blob/main/aloha_scripts/constants.py) to specify the path of your data as follows. 
```python
    "example_tasks_stage_1": { # for Stage 1 w/o Vision-Language Data
        'dataset_dir': [
            ROBOT_DATA_DIR + '/your_task_1',
            ROBOT_DATA_DIR + '/your_task_2',
        ],
        'episode_len': 1000,
        'camera_names': ['left', 'right', 'wrist'],
    },
    "example_tasks_stage_2": { # for Stage 2 with Vision-Language Data
        'dataset_dir': [
            ROBOT_DATA_DIR + '/your_task_1',
            ROBOT_DATA_DIR + '/your_task_2',
        ],
        'vl_file': os.path.join(VL_IMAGE_DIR, "llava_v1_5_mix665k.json"), # replace to your own VL Data if needed
        'vl_image_dir': os.path.join(VL_IMAGE_DIR, "data"),
        'episode_len': 1000,
        'camera_names': ['left', 'right', 'wrist'],
    },
```

4. Save original Qwen2_VL weights to init MoE. 

    You can refer to [save_mlp_weights_for_init_moe.py](https://github.com/tutujingyugang1/ChatVLA_public/blob/main/save_mlp_weights_for_init_moe.py)



## 🦾Training
We provided training scripts for both stages in [train_stage_1.sh](https://github.com/tutujingyugang1/ChatVLA_public/blob/main/scripts/train_chatvla_stage_1.sh) and [train_stage_2.sh](https://github.com/tutujingyugang1/ChatVLA_public/blob/main/scripts/train_chatvla_stage_2.sh).
For each script, you should change the following parameters:
1. **OUTPUT**: the save directory for training. 

    **❗** the keyword "qwen2" must be included in OUTPUT.

2. **TASK**: the tasks used for training. This should be corresponded to your own task name in [constants.py](https://github.com/tutujingyugang1/ChatVLA_public/blob/main/aloha_scripts/constants.py).

    **❗** Stage 2 should use a different task name compared with Stage 1 as it utilize vision-language data in training.

3. **MNOP**: model name or path. You should change to path to the pretrained VLM weights.

Other hyperparameters like "batch_size", "save_steps" could be customized according to your computation resources.

Start training by following commands:

### Stage 1: Training with Robot Data only
Key arguments:
```shell
--using_moe True \
--init_moe True \
--freeze_vl_expert True \
```
### Stage 2: Co-training with VL Data
Key arguments:
```shell
--using_moe True \
--init_moe False \
--vl_ratio 0.33 \
```
vl_ratio contols the ratio of VL Data and Robot Data, you can change it as you like.

## ChatVLA Weights
You can download weights of ChatVLA from this link:
| Model               | Link                                                           |
|---------------------|----------------------------------------------------------------|
| ChatVLA | [huggingface](https://huggingface.co/zzymeow/ChatVLA) |

## Evaluation
**❗❗** Make sure your trained checkpoint dir has two files: "preprocessor_config.json" and "chat_template.json".
If not, please copy them from downloaded Qwen2_VL weights or this [link](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct/tree/main).

### Evaluation on Real Robot

You can refer to our evaluation script [evaluate_robot.py](https://github.com/tutujingyugang1/ChatVLA_public/blob/main/evaluate/evaluate_robot.py) to evaluate your ChatVLA.

![](./doc/chatvla_mani_result.png)


### Evaluation on Multi-modal Understanding Tasks
We leverage the excellent [VLMEvalKit](https://arxiv.org/abs/2407.11691) for evaluating ChatVLA. The toolkit has been integrated into our project with minor modifications to support ChatVLA's evaluation framework.

To evaluate on multi-modal understanding tasks, you should:
1. Set a path "LMUData" to download datasets (default path is '~'). Your LMUData folder should looks like:
```shell
LMUData
├── images/
│   ├── MMMU/
│   └── MMStar/
├── MMStar.tsv
└── MMMU_DEV_VAL.tsv
``` 
2. Modify the config [config_vla.json](https://github.com/tutujingyugang1/ChatVLA_public/blob/main/evaluate/VLMEvalKit/config_vla.json) to decide the model path and the benchmarks you want to evaluate on. 
3. Run the evaluation script [evaluate_vqa.sh](https://github.com/tutujingyugang1/ChatVLA_public/blob/main/evaluate/evaluate_vqa.sh) to evaluate ChatVLA on multi-modal understanding tasks.

Note: To evalutae our ChatVLA on more benchmarks, you should modify the config following the original [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) setting. You can refer to it for more details.


![](./doc/chatvla_vqa_result.png)

## Acknowledgement
We build our project based on:
- [LLaVA](https://github.com/haotian-liu/LLaVA): an amazing open-sourced project for vision language assistant
- [act-plus-plus](https://github.com/MarkFzp/act-plus-plus): an amazing open-sourced project for robotics visuomotor learning
- [Miphi](https://github.com/zhuyiche/llava-phi): an amazing open-sourced project for tiny vision language model
- [DexVLA](https://github.com/juruobenruo/DexVLA): an amazing open-sourced Vision-Language Model with Plug-In Diffusion Expert for Visuomotor Policy Learning
- [VLMEvalKit](https://github.com/open-compass/VLMEvalKit): an amazing open-source evaluation toolkit of large vision-language models (LVLMs)

## Citation

```bibtex
# ChatVLA
@article{zhou2025chatvla,
  title={Chatvla: Unified multimodal understanding and robot control with vision-language-action model},
  author={Zhou, Zhongyi and Zhu, Yichen and Zhu, Minjie and Wen, Junjie and Liu, Ning and Xu, Zhiyuan and Meng, Weibin and Cheng, Ran and Peng, Yaxin and Shen, Chaomin and others},
  journal={arXiv preprint arXiv:2502.14420},
  year={2025}
}

# ChatVLA-2
@article{zhou2025vision,
  title={Vision-Language-Action Model with Open-World Embodied Reasoning from Pretrained Knowledge},
  author={Zhou, Zhongyi and Zhu, Yichen and Wen, Junjie and Shen, Chaomin and Xu, Yi},
  journal={arXiv preprint arXiv:2505.21906},
  year={2025}
}

```