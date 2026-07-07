import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "evaluation" / "train.py"
DEFAULT_DATA_DIR = PROJECT_ROOT / "datasets" / "BC1" / "features"
DEFAULT_RANK_LIST_DIR = PROJECT_ROOT / "datasets" / "rank_list"
DEFAULT_FEATURES = ["conch", "midnight", "resnet50", "uni"]


def discover_rank_lists(rank_list_dir):
	return sorted(path.stem for path in rank_list_dir.glob("*.json"))


def build_command(feature_name, json_path, data_dir, model_key, seed):
	return [
		sys.executable,
		str(TRAIN_SCRIPT),
		"--data-dir",
		str(data_dir),
		"--feature-name",
		feature_name,
		"--json-paths",
		json_path,
		"--model-key",
		model_key,
		"--seed",
		str(seed),
	]


def make_jobs(features, json_paths, data_dir, model_key, seed):
	jobs = []
	for feature_name in features:
		for json_path in json_paths:
			jobs.append(
				{
					"name": f"{feature_name} x {json_path}",
					"cmd": build_command(feature_name, json_path, data_dir, model_key, seed),
				}
			)
	return jobs


def set_cuda_device(env, cuda_devices, job_index):
	if not cuda_devices:
		return env

	devices = [device.strip() for device in cuda_devices.split(",") if device.strip()]
	if not devices:
		return env

	env = env.copy()
	env["CUDA_VISIBLE_DEVICES"] = devices[job_index % len(devices)]
	return env


def run_jobs(jobs, parallel, cuda_devices, dry_run):
	if dry_run:
		for job in jobs:
			print(shlex.join(job["cmd"]))
		return

	running = []
	next_job_index = 0
	failed = []

	while next_job_index < len(jobs) or running:
		while next_job_index < len(jobs) and len(running) < parallel:
			job = jobs[next_job_index]
			env = set_cuda_device(os.environ, cuda_devices, next_job_index)
			print(f"\n>>> Start {next_job_index + 1}/{len(jobs)}: {job['name']}")
			process = subprocess.Popen(job["cmd"], cwd=PROJECT_ROOT, env=env)
			running.append((job, process))
			next_job_index += 1

		still_running = []
		for job, process in running:
			return_code = process.poll()
			if return_code is None:
				still_running.append((job, process))
				continue
			if return_code != 0:
				failed.append((job["name"], return_code))
				print(f"<<< Failed: {job['name']} (exit {return_code})")
			else:
				print(f"<<< Done: {job['name']}")
		running = still_running

		if running and (next_job_index >= len(jobs) or len(running) >= parallel):
			running[0][1].wait()

	if failed:
		failed_text = ", ".join(f"{name}: {code}" for name, code in failed)
		raise SystemExit(f"Failed jobs: {failed_text}")


def parse_args():
	parser = argparse.ArgumentParser(
		description="Run feature/rank-list combinations without validation."
	)
	parser.add_argument(
		"--data-dir",
		default=DEFAULT_DATA_DIR,
		type=Path,
		help="Directory containing feature subfolders.",
	)
	parser.add_argument(
		"--features",
		nargs="+",
		default=DEFAULT_FEATURES,
		help="Feature folders to evaluate.",
	)
	parser.add_argument(
		"--json-paths",
		nargs="+",
		default=None,
		help="Rank list names or JSON paths. Examples: conch, midnight_0p3, midnight_fscore_0p3.",
	)
	parser.add_argument(
		"--model-key",
		default="ff",
		choices=["ff", "linear"],
		help="Model architecture key.",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=42,
		help="Random seed.",
	)
	parser.add_argument(
		"--parallel",
		type=int,
		default=1,
		help="Number of train.py processes to run at the same time.",
	)
	parser.add_argument(
		"--cuda-devices",
		default=None,
		help="Comma-separated CUDA device ids. Jobs are assigned round-robin.",
	)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="Print commands without running them.",
	)
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	json_paths = args.json_paths or discover_rank_lists(DEFAULT_RANK_LIST_DIR)
	jobs = make_jobs(
		features=args.features,
		json_paths=json_paths,
		data_dir=args.data_dir,
		model_key=args.model_key,
		seed=args.seed,
	)
	run_jobs(
		jobs=jobs,
		parallel=max(1, args.parallel),
		cuda_devices=args.cuda_devices,
		dry_run=args.dry_run,
	)

'''
python evaluation/run_feature_grid.py

python evaluation/run_feature_grid.py \
	--data-dir datasets/BC1/features

python evaluation/run_feature_grid.py \
	--data-dir datasets/BC1/features \
	--parallel 10

python evaluation/run_feature_grid.py \
	--data-dir datasets/BC1/features \
	--json-paths midnight_0p3 resnet50_0p3 uni_0p3 \
	--parallel 20

python evaluation/run_feature_grid.py \
	--data-dir datasets/BC1/features \
	--features conch uni \
	--json-paths conch uni \
	--parallel 2
'''
