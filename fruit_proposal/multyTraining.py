import argparse
import subprocess
import os
import glob

def parse_args():
    parser = argparse.ArgumentParser(
        description="Automate Nerfacto RGB + Fruit-Proposal semantic training"
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Path to dataset directory with RGB and semantic masks"
    )
    parser.add_argument(
        "--output_dir", default="outputs",
        help="Directory to save logs and checkpoints"
    )
    parser.add_argument(
        "--nerfacto_steps", type=int, default=None,
        help="Max iterations for Nerfacto (overrides default max)"
    )
    return parser.parse_args()

def build_ns_args(args, method):
    cmd = [
        "ns-train",
        method,
        "--data", args.dataset,
        "--output-dir", args.output_dir,
        "--experiment-name", os.path.basename(args.dataset),
        "--viewer.quit-on-train-completion", "True",
    ]
    if args.nerfacto_steps is not None and method == "nerfacto":
        cmd += ["--max-num-iterations", str(args.nerfacto_steps)]
    return cmd

def run_training(args, method):
    cmd = build_ns_args(args, method)
    print(f"Running {method}: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    # output structure: output_dir/<dataset_name>/<method>/<timestamp>/nerfstudio_models
    base = os.path.join(args.output_dir, os.path.basename(args.dataset), method)
    # find latest timestamp folder
    times = sorted(glob.glob(os.path.join(base, "*")))
    if not times:
        raise FileNotFoundError(f"No timestamp dirs in {base}")
    latest_time = times[-1]
    model_dir = os.path.join(latest_time, "nerfstudio_models")
    return model_dir

def latest_checkpoint(model_dir):
    ckpts = glob.glob(os.path.join(model_dir, "step-*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {model_dir}")
    # sort by step number
    ckpts_sorted = sorted(
        ckpts,
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('-')[-1])
    )
    return ckpts_sorted[-1]

def main():
    args = parse_args()
    # Stage I: Nerfacto RGB + density
    nerf_model_dir = run_training(args, "nerfacto")
    ckpt = latest_checkpoint(nerf_model_dir)

    # Stage II: Fruit-Proposal semantic with frozen Nerfacto ckpt
    # add method-specific flags
    method = "fruit-proposal"
    cmd = build_ns_args(args, method)
    cmd += [
        "--method-name", method,
        "--load-checkpoint", ckpt,
    ]
    print(f"Running {method}: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()