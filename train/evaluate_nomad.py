import os
import io
import yaml
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from vint_train.models.nomad.nomad import NoMaD, DenseNetwork
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from vint_train.data.vint_dataset import ViNT_Dataset
from vint_train.data.data_utils import (
    get_data_path, to_local_coords, img_path_to_data, calculate_sin_cos
)

# ---------------- split builder ----------------
def build_split_from_folder(data_root: str, split_dir: str) -> str:
    os.makedirs(split_dir, exist_ok=True)
    trajs = []
    for name in sorted(os.listdir(data_root)):
        full = os.path.join(data_root, name)
        if os.path.isdir(full) and os.path.exists(os.path.join(full, "traj_data.pkl")):
            trajs.append(name)
    assert len(trajs) > 0, f"No valid trajectories found in {data_root}"
    split_file = os.path.join(split_dir, "traj_names.txt")
    with open(split_file, "w", encoding="utf-8") as f:
        f.write("\n".join(trajs))
    print(f"[split] wrote {len(trajs)} trajs to {split_file}")
    return split_dir

# ---------------- plotting utils ----------------
def unnormalize_actions(actions, data_config, waypoint_spacing):
    return actions * data_config["metric_waypoint_spacing"] * waypoint_spacing

def sin_cos_to_angle(actions: torch.Tensor):
    x = actions[..., 0]; y = actions[..., 1]
    sin_yaw = actions[..., 2]; cos_yaw = actions[..., 3]
    yaw = torch.atan2(sin_yaw, cos_yaw)
    return torch.stack([x, y, yaw], dim=-1)

def plot_trajectory(pred_actions, gt_actions, out_dir, fname, data_cfg, waypoint_spacing):
    pred_unn = unnormalize_actions(pred_actions, data_cfg, waypoint_spacing)
    gt_unn   = unnormalize_actions(gt_actions,   data_cfg, waypoint_spacing)

    if pred_unn.shape[-1] == 4: pred_unn = sin_cos_to_angle(pred_unn)
    if gt_unn.shape[-1]   == 4: gt_unn   = sin_cos_to_angle(gt_unn)

    # actions 已是相对于起点的绝对偏移，不再累加
    pred_xy = pred_unn[0, :, :2].detach().cpu().numpy()
    gt_xy   = gt_unn[0,   :, :2].detach().cpu().numpy()
    pred_path = np.vstack([[0, 0], pred_xy])
    gt_path   = np.vstack([[0, 0], gt_xy])

    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(8, 8))
    plt.plot(gt_path[:, 0], gt_path[:, 1], "go-", label="Ground Truth", linewidth=2)
    plt.plot(pred_path[:, 0], pred_path[:, 1], "bo-", label="Predicted", linewidth=2)
    plt.scatter(gt_path[0, 0], gt_path[0, 1], c="red", marker="x", s=100, label="Start", zorder=3)
    plt.scatter(gt_path[-1, 0], gt_path[-1, 1], c="green", marker="*", s=150, label="Goal", zorder=3)
    plt.xlabel("X (meters)"); plt.ylabel("Y (meters)")
    plt.title("Predicted vs. Ground Truth Trajectory (Goal-Conditioned)")
    plt.axis("equal"); plt.legend(); plt.grid(True)
    path = os.path.join(out_dir, fname)
    plt.savefig(path); plt.close()
    print(f"Saved plot to {path}")

# ---------------- goal-conditioned positive-only eval dataset ----------------
class EvalPositiveDataset(Dataset):
    """
    评测专用：只采样“同轨迹、向前 eval_horizon*waypoint_spacing”的正样本；
    图像加载与标准化与训练保持一致；包含 dtype=object 兜底净化。
    """
    def __init__(self, base: ViNT_Dataset, eval_horizon: int):
        self.base = base
        self.eval_horizon = eval_horizon
        self.w = base.waypoint_spacing
        self.len_pred = base.len_traj_pred
        assert self.len_pred == eval_horizon, \
            f"建议 eval_horizon == len_traj_pred ({self.len_pred}); 当前={eval_horizon}"

        self.items = []
        for traj_name in self.base.traj_names:
            traj = self.base._get_trajectory(traj_name)
            T = len(traj["position"])
            begin = self.base.context_size * self.w
            end   = T - self.base.end_slack - self.len_pred * self.w
            for curr_time in range(begin, end):
                goal_time = curr_time + self.eval_horizon * self.w
                if goal_time < T:
                    self.items.append((traj_name, curr_time, goal_time))
        assert len(self.items) > 0, "No valid positive samples for evaluation."

    def __len__(self):
        return len(self.items)

    def _load_image(self, trajectory_name, time):
        image_path = get_data_path(self.base.data_folder, trajectory_name, time)
        try:
            with self.base._image_cache.begin() as txn:
                buf = txn.get(image_path.encode())
            if buf is None:
                raise RuntimeError("cache miss")
            image_bytes = io.BytesIO(bytes(buf))
            return img_path_to_data(image_bytes, self.base.image_size)
        except Exception:
            return img_path_to_data(image_path, self.base.image_size)

    def __getitem__(self, idx):
        traj_name, curr_time, goal_time = self.items[idx]

        context_times = list(range(curr_time + -self.base.context_size * self.w,
                                   curr_time + 1, self.w))
        obs_image = torch.cat([self._load_image(traj_name, t) for t in context_times])
        goal_image = self._load_image(traj_name, goal_time)

        traj = self.base._get_trajectory(traj_name)
        start_idx = curr_time
        end_idx   = curr_time + self.len_pred * self.w + 1

        yaw = traj["yaw"][start_idx:end_idx:self.w]
        pos = traj["position"][start_idx:end_idx:self.w]

        yaw = np.asarray(yaw)
        pos = np.asarray(pos)
        if len(yaw.shape) == 2:
            yaw = yaw.squeeze(1)
        if yaw.shape != (self.len_pred + 1,):
            const_len = self.len_pred + 1 - yaw.shape[0]
            yaw = np.concatenate([yaw, np.repeat(yaw[-1], const_len)])
            pos = np.concatenate([pos, np.repeat(pos[-1][None], const_len, axis=0)], axis=0)
        yaw = yaw.astype(np.float32, copy=False)
        pos = pos.astype(np.float32, copy=False)

        current_yaw = float(yaw[0])
        waypoints   = to_local_coords(pos, pos[0], current_yaw)

        if self.base.learn_angle:
            dyaw = (yaw[1:] - yaw[0]).astype(np.float32, copy=False)
            actions = np.concatenate([waypoints[1:], dyaw[:, None]], axis=-1)
        else:
            actions = waypoints[1:]

        if self.base.normalize:
            actions[:, :2] /= self.base.data_config["metric_waypoint_spacing"] * self.w

        if actions.dtype == np.object_:
            actions = np.array(actions.tolist()).astype(np.float32)
        else:
            actions = actions.astype(np.float32, copy=False)

        actions_torch = torch.as_tensor(actions, dtype=torch.float32)
        if self.base.learn_angle:
            actions_torch = calculate_sin_cos(actions_torch)

        distance = torch.as_tensor(self.eval_horizon, dtype=torch.int64)
        goal_pos = torch.zeros(2, dtype=torch.float32)
        which_dataset = torch.as_tensor(self.base.dataset_index, dtype=torch.int64)
        action_mask = torch.tensor(1.0, dtype=torch.float32)

        return (
            torch.as_tensor(obs_image, dtype=torch.float32),
            torch.as_tensor(goal_image, dtype=torch.float32),
            actions_torch,
            distance,
            goal_pos,
            which_dataset,
            action_mask,
        )

# ---------------- UNet1D 安全调用：自适配 BTC/BCT ----------------
def call_unet_safe(unet, x_BTC: torch.Tensor, cond: torch.Tensor, t, act_dim: int) -> torch.Tensor:
    """
    x_BTC: (B, T, C)  —— scheduler 侧维度布局
    返回:  (B, T, C)
    按第一层 Conv1d 的 in_channels 自动判断是否需要 permute 到 (B, C, T)。
    """
    first_w = None
    for n, p in unet.named_parameters():
        if "weight" in n and p.ndim == 3:   # Conv1d weight: [out, in, k]
            first_w = p
            break

    if first_w is None:
        # 兜底：按 BCT 喂入
        x_BCT = x_BTC.permute(0, 2, 1)
        y_BCT = unet(sample=x_BCT, timestep=t, global_cond=cond)
        return y_BCT.permute(0, 2, 1)

    in_ch = first_w.shape[1]

    if in_ch == act_dim:
        # 模型期望 (B, C, T)
        x_BCT = x_BTC.permute(0, 2, 1)
        y_BCT = unet(sample=x_BCT, timestep=t, global_cond=cond)
        return y_BCT.permute(0, 2, 1)
    elif in_ch == x_BTC.shape[1]:
        # 极少数实现：把 T 当 in_channels；按 BTC 直接喂
        y = unet(sample=x_BTC, timestep=t, global_cond=cond)
        # 若返回 (B, C, T) 则转回 BTC
        if y.ndim == 3 and y.shape[1] == act_dim:
            y = y.permute(0, 2, 1)
        return y
    else:
        # 不匹配，退回 BCT 路径
        x_BCT = x_BTC.permute(0, 2, 1)
        y_BCT = unet(sample=x_BCT, timestep=t, global_cond=cond)
        return y_BCT.permute(0, 2, 1)

# ---------------- main eval ----------------
def evaluate(args):
    torch.manual_seed(0); np.random.seed(0)

    cfg_path = os.path.join("train", args.config_path)
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset_name = args.dataset_name if args.dataset_name in config["datasets"] else next(iter(config["datasets"]))
    if dataset_name != args.dataset_name:
        print(f"[warn] dataset '{args.dataset_name}' not in config; fallback to '{dataset_name}'")

    split_dir = build_split_from_folder(args.data_path, os.path.join("train", args.split_path))

    base_ds = ViNT_Dataset(
        data_folder=args.data_path,
        data_split_folder=split_dir,
        dataset_name=dataset_name,
        image_size=tuple(config["image_size"]),
        waypoint_spacing=1,
        min_dist_cat=config["action"]["min_dist_cat"],
        max_dist_cat=config["action"]["max_dist_cat"],
        min_action_distance=config["action"]["min_dist_cat"],
        max_action_distance=config["action"]["max_dist_cat"],
        negative_mining=False,
        len_traj_pred=config["len_traj_pred"],
        learn_angle=config["learn_angle"],
        context_size=config["context_size"],
        context_type="temporal",
        end_slack=config["datasets"][dataset_name]["end_slack"],
        goals_per_obs=1,
        normalize=config["normalize"],
    )

    eval_horizon = args.eval_horizon if args.eval_horizon is not None else config["len_traj_pred"]
    ds = EvalPositiveDataset(base_ds, eval_horizon=eval_horizon)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    with open(os.path.join("train", "vint_train", "data", "data_config.yaml"), "r") as f:
        data_cfg_all = yaml.safe_load(f)
    data_cfg = data_cfg_all[dataset_name]
    print("[data] metric_waypoint_spacing:", data_cfg.get("metric_waypoint_spacing", None))

    vision_encoder = NoMaD_ViNT(
        obs_encoding_size=config["encoding_size"],
        context_size=config["context_size"],
        mha_num_attention_heads=config["mha_num_attention_heads"],
        mha_num_attention_layers=config["mha_num_attention_layers"],
        mha_ff_dim_factor=config["mha_ff_dim_factor"],
    )
    vision_encoder = replace_bn_with_gn(vision_encoder)

    act_dim = 4 if config.get("learn_angle", False) else 2
    noise_pred_net = ConditionalUnet1D(
        input_dim=act_dim,
        global_cond_dim=config["encoding_size"],
        down_dims=config["down_dims"],
        cond_predict_scale=config["cond_predict_scale"],
    )
    dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])
    model = NoMaD(vision_encoder=vision_encoder,
                  noise_pred_net=noise_pred_net,
                  dist_pred_net=dist_pred_network).to(device)

    scheduler = DDPMScheduler(
        num_train_timesteps=config["num_diffusion_iters"],
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )
    steps = args.num_inference_steps or config.get("num_inference_steps", 20)
    scheduler.set_timesteps(steps)

    print(f"Loading model weights from {args.model_path}")
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    out_dir = os.path.join("train", args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    n, total_mse_norm, total_mse_m = 0, 0.0, 0.0

    for i, (obs, goal, gt_actions, _, _, _, _) in enumerate(dl):
        if i >= args.num_plots:
            print(f"\nFinished generating {args.num_plots} plots.")
            break

        obs, goal, gt_actions = obs.to(device), goal.to(device), gt_actions.to(device)

        B = obs.shape[0]
        goal_mask = torch.ones(B, dtype=torch.bool, device=device)
        cond = model("vision_encoder", obs_img=obs, goal_img=goal, input_goal_mask=goal_mask)

        use_cfg = args.cfg_scale > 0.0
        if use_cfg:
            uncond = model("vision_encoder", obs_img=obs, goal_img=goal,
                           input_goal_mask=torch.zeros_like(goal_mask))

        pred_actions = torch.randn((B, base_ds.len_traj_pred, act_dim), device=device)
        assert pred_actions.shape == (B, base_ds.len_traj_pred, act_dim), \
            f"pred_actions shape {pred_actions.shape} != (B,{base_ds.len_traj_pred},{act_dim})"

        # 3. Iteratively denoise
        for t in scheduler.timesteps:
            # predict noise model output
            # UNet expects (B, C, T) but we have (B, T, C), so we permute
            model_input = pred_actions.permute(0, 2, 1)
            noise_pred = model('noise_pred_net', sample=model_input, timestep=t, global_cond=cond)
            # The output is (B, C, T), so we permute it back to (B, T, C)
            noise_pred = noise_pred.permute(0, 2, 1)

            # compute previous image: x_t -> x_t-1
            pred_actions = scheduler.step(
                model_output=noise_pred,
                timestep=t,
                sample=pred_actions
            ).prev_sample

        # metrics
        mean_step_pred = pred_actions.norm(dim=-1).mean().item()
        mean_step_gt   = gt_actions.norm(dim=-1).mean().item()
        v_pred = pred_actions[0, -1, :2]; v_gt = gt_actions[0, -1, :2]
        cos_sim = torch.nn.functional.cosine_similarity(v_pred, v_gt, dim=0).item()

        mse_norm = torch.nn.functional.mse_loss(pred_actions, gt_actions).item()
        pred_m = unnormalize_actions(pred_actions, data_cfg, waypoint_spacing=1)
        gt_m   = unnormalize_actions(gt_actions,   data_cfg, waypoint_spacing=1)
        mse_m  = torch.nn.functional.mse_loss(pred_m, gt_m).item()

        print(f"[{dataset_name}] sample {i}: avg|Δ| pred={mean_step_pred:.3f}, "
              f"gt={mean_step_gt:.3f} | cos(final)={cos_sim:.3f} "
              f"| MSE(norm)={mse_norm:.3f}, MSE(m)={mse_m:.3f}")

        total_mse_norm += mse_norm
        total_mse_m    += mse_m
        n += 1

        plot_trajectory(pred_actions, gt_actions, out_dir,
                        f"{dataset_name}_pos_only_sample_{i}.png", data_cfg, waypoint_spacing=1)

    if n > 0:
        print(f"\n--- Done (Goal-Conditioned, positive-only) --- "
              f"Avg MSE(norm)={total_mse_norm/n:.4f}, Avg MSE(m)={total_mse_m/n:.4f}")
    else:
        print("No valid samples.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="nomad.pth")
    parser.add_argument("--data-path", type=str, default="train/go_stanford")
    parser.add_argument("--split-path", type=str, default="temp_eval_split")
    parser.add_argument("--config-path", type=str, default="config.yaml")
    parser.add_argument("--output-dir", type=str, default="evaluation_plots")
    parser.add_argument("--num-plots", type=int, default=20)
    parser.add_argument("--dataset-name", type=str, default="go_stanford")
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--cfg-scale", type=float, default=3.0)
    parser.add_argument("--eval-horizon", type=int, default=None,
                        help="正样本目标的地平线（单位：waypoints），默认等于 len_traj_pred")
    args = parser.parse_args()
    evaluate(args)
