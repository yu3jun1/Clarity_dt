"""F training/evaluation and a non-destructive resource-aware seed campaign."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from clarity_rrt_v3 import evaluate, train
from clarity_rrt_v3.replicate_audit import read_json, write_json
from clarity_rrt_v3.reproducibility import file_sha256, utc_now
from .protocol import DEFAULT_CONFIG, REPO_ROOT, assert_design, install_runtime, source_fingerprint


def gpu_information(index):
    line = subprocess.check_output([
        'nvidia-smi', '-i', str(index),
        '--query-gpu=uuid,name,memory.used,memory.total,utilization.gpu',
        '--format=csv,noheader,nounits',
    ], text=True).strip()
    uuid, name, used, total, utilization = [field.strip() for field in line.split(',')]
    return {'index': index, 'uuid': uuid, 'name': name, 'memory_used_mib': int(used),
            'memory_total_mib': int(total), 'memory_free_mib': int(total) - int(used),
            'utilization_percent': int(utilization)}


def gpu_ready(info, allow_shared):
    if allow_shared:
        return info['memory_free_mib'] >= 60000
    return info['memory_used_mib'] <= 1024 and info['utilization_percent'] <= 5


def job_arguments(args, seed):
    return ['--config', str(Path(args.config).resolve()), '--seed', str(seed),
            '--output-root', str(Path(args.output_root).resolve()), '--replicate', args.replicate]


def environment(seed, gpu_uuid):
    return dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_uuid,
                TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')


def run_job(args):
    root = Path(args.output_root).resolve()
    directory = root / 'primary' / f'F_seed{args.seed}_{args.replicate}'
    with (root / f'F_seed{args.seed}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if directory.exists():
            raise FileExistsError(f'Refusing to overwrite or silently resume {directory}')
        protocol = read_json(root / 'protocol.json')
        fingerprint = source_fingerprint()
        if fingerprint['source_sha256'] != protocol['source_sha256']:
            raise RuntimeError('F core/extension changed before launch')
        directory.mkdir(parents=True)
        state = {'variant': 'F', 'seed': args.seed, 'replicate_id': args.replicate,
                 'state': 'training', 'gpu_uuid': os.environ['CUDA_VISIBLE_DEVICES'],
                 'train_start_utc': utc_now()}
        write_json(directory / 'status.json', state)
        arguments = job_arguments(args, args.seed)
        try:
            with (directory / 'train.log').open('x') as log:
                subprocess.run([sys.executable, '-m', 'clarity_tf_ensemble', 'train', *arguments],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            state.update(state='evaluating', evaluate_start_utc=utc_now())
            write_json(directory / 'status.json', state)
            with (directory / 'evaluate.log').open('x') as log:
                subprocess.run([sys.executable, '-m', 'clarity_tf_ensemble', 'evaluate', *arguments],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            if source_fingerprint()['source_sha256'] != fingerprint['source_sha256']:
                raise RuntimeError('F core/extension changed during the run')
            state.update(state='complete', complete_utc=utc_now())
        except Exception as error:
            state.update(state='failed', failed_utc=utc_now(), error=str(error))
            write_json(directory / 'status.json', state)
            raise
        write_json(directory / 'status.json', state)


def snapshot_controls(root):
    entries = []
    for seed in (42, 43, 44):
        source = REPO_ROOT / f'outputs/pure_rrt_v3_step2400/primary/D_seed{seed}'
        destination = root / f'historical_controls/D_seed{seed}'
        if not source.is_dir():
            continue
        destination.mkdir(parents=True, exist_ok=False)
        checksums = {}
        for name in ('config.yaml', 'metrics.json', 'recursive_predictions.csv',
                     'run_metadata.json', 'evaluation_metadata.json'):
            path = source / name
            if path.is_file():
                shutil.copy2(path, destination / name)
                checksums[name] = file_sha256(path)
                if file_sha256(destination / name) != checksums[name]:
                    raise RuntimeError('Historical-control snapshot checksum mismatch')
        entries.append({'variant': 'D', 'seed': seed, 'source': str(source),
                        'snapshot': str(destination), 'sha256': checksums,
                        'limitation': 'Historical execution protocol; not a deterministic matched rerun'})
    write_json(root / 'historical_controls/manifest.json', {'created_at_utc': utc_now(), 'runs': entries})


def campaign(args):
    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'campaign.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'protocol.json').exists():
            raise FileExistsError('Campaign already exists; never overwrite its protocol')
        config = train.resolve_run_config(args.config)
        assert_design(config)
        frozen = source_fingerprint()
        if not read_json(Path(config['data']['mri_cache_dir']) / 'manifest.json').get('complete'):
            raise RuntimeError('Requested shared-memory MRI cache is incomplete')
        assignments = dict(zip((42, 43, 44), args.gpus))
        protocol = {**frozen, 'created_at_utc': utc_now(), 'experiment': 'F',
                    'definition': 'Teacher-forced Stage-wise Ensemble; 3 independent dynamics members, no RRT training',
                    'behavioral_baseline_commit': '9871306',
                    'source_baseline': 'Current core/extension frozen at this campaign start; no dependency on old audit output files',
                    'seeds': [42, 43, 44], 'formal_replicate': args.replicate,
                    'config': str(Path(args.config).resolve()), 'assignments': assignments,
                    'gpu_inventory_at_schedule': [gpu_information(gpu) for gpu in args.gpus],
                    'shared_gpu_authorized_at_launch': args.allow_shared_gpu,
                    'mri_cache_dir': config['data']['mri_cache_dir'],
                    'mri_cache_manifest_sha256': file_sha256(Path(config['data']['mri_cache_dir']) / 'manifest.json'),
                    'core_files_unmodified': True,
                    'resource_policy': 'Never signal existing GPU jobs; wait for idle cards unless shared use is explicitly approved'}
        write_json(root / 'protocol.json', protocol)
        workers = {}
        pending = set(assignments)
        try:
            snapshot_controls(root)
            while pending or any(worker.poll() is None for worker in workers.values()):
                if source_fingerprint()['source_sha256'] != frozen['source_sha256']:
                    raise RuntimeError('F frozen source changed; other isolated jobs are left untouched')
                approval = root / 'shared_gpu_approval.json'
                shared = args.allow_shared_gpu or (approval.exists() and read_json(approval).get('approved') is True)
                availability = []
                for seed in sorted(tuple(pending)):
                    info = gpu_information(assignments[seed])
                    availability.append({'seed': seed, **info})
                    if not gpu_ready(info, shared):
                        continue
                    with (root / f'F_seed{seed}_runner.log').open('x') as log:
                        workers[seed] = subprocess.Popen([
                            sys.executable, '-m', 'clarity_tf_ensemble', 'run',
                            *job_arguments(args, seed),
                        ], env=environment(seed, info['uuid']), cwd=REPO_ROOT,
                            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
                    pending.remove(seed)
                    protocol.setdefault('actual_launches', {})[str(seed)] = {
                        'gpu': info, 'start_utc': utc_now(), 'worker_pid': workers[seed].pid,
                        'shared_gpu_approved': shared,
                    }
                    write_json(root / 'protocol.json', protocol)
                    print(f'START F seed={seed} GPU={info["index"]} isolated_worker={workers[seed].pid}', flush=True)
                for seed, worker in workers.items():
                    if worker.poll() not in (None, 0):
                        raise RuntimeError(f'F{seed} failed; see its run log. Other isolated workers are not killed.')
                write_json(root / 'campaign_status.json', {
                    'state': 'running' if workers else 'waiting_for_gpu', 'updated_at_utc': utc_now(),
                    'pending_seeds': sorted(pending), 'worker_pids': {str(seed): w.pid for seed, w in workers.items()},
                    'completed_seeds': sorted(seed for seed, w in workers.items() if w.poll() == 0),
                    'waiting_gpu_inventory': availability, 'shared_gpu_approved': shared,
                    'poll_seconds': 30,
                })
                if pending or any(worker.poll() is None for worker in workers.values()):
                    time.sleep(30)
            metrics = []
            for seed in (42, 43, 44):
                if workers[seed].returncode != 0:
                    raise RuntimeError(f'F{seed} worker did not finish successfully')
                directory = root / 'primary' / f'F_seed{seed}_{args.replicate}'
                if read_json(directory / 'status.json')['state'] != 'complete':
                    raise RuntimeError(f'Incomplete F{seed}')
                metrics.append(read_json(directory / 'metrics.json'))
            write_json(root / 'F_seed_summary.json', {'variant': 'F', 'seeds': [42, 43, 44],
                'summary': evaluate.summarize_runs(metrics), 'runs': metrics,
                'note': 'E-vs-F comparisons require matching seeds and training/evaluation protocols. Historical controls are descriptive unless protocol matching is established.'})
            write_json(root / 'campaign_status.json', {'state': 'complete', 'completed_at_utc': utc_now(),
                       'completed_seeds': [42, 43, 44], 'pending_seeds': []})
        except Exception as error:
            write_json(root / 'campaign_status.json', {'state': 'failed', 'failed_at_utc': utc_now(), 'error': str(error),
                       'pending_seeds': sorted(pending), 'worker_pids': {str(seed): w.pid for seed, w in workers.items()}})
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('train', 'evaluate', 'run', 'campaign'))
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument('--output-root', default=str(REPO_ROOT / 'outputs/ablations/teacher_forced_stagewise_ensemble'))
    parser.add_argument('--seed', type=int, choices=(42, 43, 44))
    parser.add_argument('--replicate', default='rep01')
    parser.add_argument('--gpus', type=int, nargs=3, default=[5, 6, 7])
    parser.add_argument('--allow-shared-gpu', action='store_true')
    args = parser.parse_args(argv)
    os.chdir(REPO_ROOT)
    if args.command == 'campaign':
        if len(set(args.gpus)) != 3 or any(gpu not in range(8) for gpu in args.gpus):
            parser.error('The three seed assignments require distinct valid GPU indices')
        campaign(args)
        return
    if args.seed is None:
        parser.error('--seed is required for train/evaluate/run')
    if args.command == 'run':
        run_job(args)
        return
    install_runtime()
    if args.command == 'train':
        train.train_one(args.config, 'F', args.seed, 'cuda:0', args.output_root, args.replicate)
    else:
        metrics = evaluate.evaluate_one(args.config, 'F', args.seed, 'cuda:0', args.output_root, args.replicate)
        directory = Path(args.output_root) / 'primary' / f'F_seed{args.seed}_{args.replicate}'
        metadata = read_json(directory / 'evaluation_metadata.json')
        metrics['ensemble_size'] = 3
        metrics['rrt_training'] = False
        metrics['extension_provenance'] = {key: metadata[key] for key in
                                         ('source_sha256', 'core_source_sha256', 'extension_sha256')}
        write_json(directory / 'metrics.json', metrics)


if __name__ == '__main__':
    main()
