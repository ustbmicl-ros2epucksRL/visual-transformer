# visual-transformer
上传有关我的diffusion代码，**重点可以参阅文件 “原代码提供的readme.md”**。

### 代码目录结构：

`depolyment：`作用不是很大，主要是原代码仓库用来测试的文件，不适用于我们的场景，但可以看看他们的测试方法：

- `./deployment/src/record_bag.sh`: script to collect a demo trajectory as a ROS bag in the target environment on the robot. This trajectory is subsampled to generate a topological graph of the environment.
- `./deployment/src/create_topomap.sh`: script to convert a ROS bag of a demo trajectory into a topological graph that the robot can use to navigate.
- `./deployment/src/navigate.sh`: script that deploys a trained GNM/ViNT/NoMaD model on the robot to navigate to a desired goal in the generated topological graph. Please see relevant sections below for configuration settings.
- `./deployment/src/explore.sh`: script that deploys a trained NoMaD model on the robot to randomly explore its environment. Please see relevant sections below for configuration settings.

`train：`主要用来训练的文件夹，这个里面包含了三种方法，有GNM、VINT、Nomad，我们使用的是Nomad的训练方法。

- `./train/train.py`: training script to train or fine-tune the ViNT model on your custom data.
- `./train/vint_train/models/`: contains model files for GNM, ViNT, and some baselines.
- `./train/process_*.py`: scripts to process rosbags or other formats of robot trajectories into training data.

~~`use_nomad：`新增的代码文件夹，里面是我们自己的做法。~~

`train/evaluation_plots` ： **测试的输出对比结果**。

`Nomad.pdf`:  **该代码对应的论文**。

`Nomad论文的PPT.pptx`：**阅读nomad论文制作的PPT文件**。

### 数据集介绍：

原项目代码使用的数据集有多种，分别是下列，但是目前我们能下载并使用的是第四种[GoStanford2 (Modified)](https://drive.google.com/drive/folders/1RYseCpbtHEFOsmSX2uqNY_kvSxwZLVP_?usp=sharing)

In the [papers](https://general-navigation-models.github.io), we train on a combination of publicly available and unreleased datasets. Below is a list of publicly available datasets used for training; please contact the respective authors for access to the unreleased data.

- [RECON](https://sites.google.com/view/recon-robot/dataset)
- [TartanDrive](https://github.com/castacks/tartan_drive)
- [SCAND](https://www.cs.utexas.edu/~xiao/SCAND/SCAND.html#Links)
- [GoStanford2 (Modified)](https://drive.google.com/drive/folders/1RYseCpbtHEFOsmSX2uqNY_kvSxwZLVP_?usp=sharing)
- [SACSoN/HuRoN](https://sites.google.com/view/sacson-review/huron-dataset)

#### 数据集的处理：

这一块参考原项目说明


We provide some sample scripts to process these datasets, either directly from a rosbag or from a custom format like HDF5s:

1. Run `process_bags.py` with the relevant args, or `process_recon.py` for processing RECON HDF5s. You can also manually add your own dataset by following our structure below (if you are adding a custom dataset, please checkout the [Custom Datasets](#custom-datasets) section).
2. Run `data_split.py` on your dataset folder with the relevant args.

After step 1 of data processing, the processed dataset should have the following structure:

```
├── <dataset_name>
│   ├── <name_of_traj1>
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   ├── ...
│   │   ├── T_1.jpg
│   │   └── traj_data.pkl
│   ├── <name_of_traj2>
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   ├── ...
│   │   ├── T_2.jpg
│   │   └── traj_data.pkl
│   ...
└── └── <name_of_trajN>
    	├── 0.jpg
    	├── 1.jpg
    	├── ...
        ├── T_N.jpg
        └── traj_data.pkl
```

Each `*.jpg` file contains an forward-facing RGB observation from the robot, and they are temporally labeled. The `traj_data.pkl` file is the odometry data for the trajectory. It’s a pickled dictionary with the keys:

- `"position"`: An np.ndarray [T, 2] of the xy-coordinates of the robot at each image observation.
- `"yaw"`: An np.ndarray [T,] of the yaws of the robot at each image observation.


After step 2 of data processing, the processed data-split should the following structure inside `vint_release/train/vint_train/data/data_splits/`:

```
├── <dataset_name>
│   ├── train
|   |   └── traj_names.txt
└── └── test
        └── traj_names.txt 
```

### ppt展示

放置在文件夹中，文件夹名称：“组会ppt"，里面记录了完成的工作。

### 测试展示：

**`evaluate_nomad.py `是专门用来测试nomad模型的测试文件代码。**

```python
python train/evaluate_nomad.py --config-path "config/nomad.yaml"
```

**输入：**数据集中的格式

**输出：**子目标[x,y]

问题：

1. **代码中没有把位姿作为输入??是否需要呢？**

   不需要，位姿不作为训练输入，位姿不进网络当特征，但它是“标注尺子”——用来把未来轨迹从世界坐标统一、纠正、投影到机器人自车系，生成正确、可学习的监督信号。

**测试结果:**

已提供的nomad.pth模型有些失效，预测的点几乎都是在0左右。输出在`train/evaluation_plots`文件夹下。测试效果并不好，这一部分需要重新考虑。。。

### 新的更新

**VINT模型论文的思维导图：**

![image-20251104162635979](https://store-image-mj.oss-cn-beijing.aliyuncs.com/img/image-20251104162635979.png)

**Nomad论文的思维导图：**

![image-20251104162721454](https://store-image-mj.oss-cn-beijing.aliyuncs.com/img/image-20251104162721454.png)

1. **subgoal diffsuion本身能做什么？**

Subgoal Diffusion 本质是一种图像到图像的条件生成模型，其**核心功能是 “基于当前环境视觉观测，生成符合物理规律的短期子目标图像**”，Subgoal Diffusion 能以当前图像（`o_t`）为条件，生成 5-20 步后机器人可能到达的 “潜在子目标图像”。

2. **VINT中使用subgoal diffusion** ?

   输入：当前视觉观测图像（`o_t`）

   输出：可达的候选子目标图像集合（`S = {s_1, s_2, ..., s_K}`）

   生成的候选子目标并非直接用于导航， ViNT 计算每个子目标与当前位置的 “动态距离（到达所需时间步）” 和 “初步动作序列”，筛选掉不可达的无效子目标，再结合启发式函数（如 GPS、卫星图像）评分，最终选择最优子目标输入 ViNT 生成执行动作。

   

3. **nomad训练过程是怎么获取子目标图像的？** 

在`vint_dataset.py`中，子目标图像通过`_sample_goal`方法采样得到。

- 在`_sample_goal`方法中，对于当前时间步`curr_time`，根据最大目标距离（`max_goal_dist`）随机采样未来时间步`goal_time`，满足`goal_time = curr_time + goal_offset * waypoint_spacing`（`goal_offset`为随机整数，范围`[1, max_goal_dist]`）。

- 若`goal_offset=0`，则通过`_sample_negative`随机采样其他轨迹的时间步作为负样本子目标。

- 采样得到`goal_time`后，通过`_load_image(trajectory_name, goal_time)`从 LMDB 缓存中读取对应时间步的 RGB 图像，作为`goal_img`。

- 例如：当前轨迹为`traj_001`，`curr_time=10`，采样`goal_offset=5`，`waypoint_spacing=2`，则`goal_time=10 + 5*2=20`，加载`traj_001/20.jpg`作为子目标图像。

  

5. **loss函数具体是怎么更新的？**

​		$\mathcal{L}_{\text{NoMaD}}(\phi,\psi,f,\theta,f_d)
= \mathrm{MSE}\!\left(\varepsilon_\xi^{k},\, \varepsilon_{\theta}\!\left(c_t,\, a_t^{0}+\varepsilon_\xi^{k},\, k\right)\right)\lambda \cdot \mathrm{MSE}\!\left(d(o_t,o_g),\, f_d(c_t)\right)$

代码中对应的是：

```python
            # Total loss
            loss = alpha * dist_loss + (1-alpha) * diffusion_loss
```

损失函数主要包括扩散模型的噪声预测损失和距离预测损失：

扩散模型的噪声预测损失计算逻辑（对应 `train_utils.py` 中 `train_nomad` 函数）：

1. 对真实动作序列（`naction`）添加随机噪声，得到带噪声的动作序列（`noisy_actions`）。
2. 模型预测噪声（`noise_pred`），并与真实噪声（`noise`）计算 MSE 损失。
3. 引入目标掩码机制（`goal_mask_prob`），对部分样本随机掩盖子目标信息，增强模型鲁棒性，此时损失需包含掩码场景下的噪声预测误差。

距离预测损失计算逻辑：

模型通过 `dist_pred_net` 预测当前观测到子目标的距离，损失为预测距离与真实距离的 MSE 损失，辅助模型感知子目标的距离。



6. **子目标图像在loss函数更新中起到了什么作用？**

会影响噪声预测结果，损失函数的第一项构成。因为子目标图像会和观测图像一起作为噪声预测网络的输入，影响噪声的预测。





