"""Process-local F registration and combined core/extension provenance."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from clarity_rrt_v3 import evaluate, reproducibility, train

EXTENSION_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXTENSION_ROOT.parents[1]
DEFAULT_CONFIG = EXTENSION_ROOT / 'configs/F.yaml'
F_KIND = 'teacher_forced_stagewise_ensemble_ablation'
F_VARIANTS = {'F': {'training_scheme': 'teacher_forced_stagewise', 'ensemble_size': 3}}
CORE_ASSERT_DESIGN = train.assert_experiment_design
CORE_COLLECT_METADATA = reproducibility.collect_metadata


def assert_design(config):
    if config.get('experiment_kind') != F_KIND:
        return CORE_ASSERT_DESIGN(config)
    if config['variants'] != F_VARIANTS:
        raise AssertionError('F must use teacher-forced stage-wise training and exactly 3 dynamics members')
    baseline = train.load_config(REPO_ROOT / 'configs/ablations/teacher_forced_stagewise.yaml')
    for field in ('upstream_root', 'upstream_commit', 'data', 'model', 'training'):
        if config[field] != baseline[field]:
            raise AssertionError(f'F must retain E baseline settings: {field}')
    if not config.get('deterministic'):
        raise AssertionError('The F campaign requires strict deterministic training')


def extension_fingerprint():
    files = {}
    digest = hashlib.sha256()
    for path in sorted(EXTENSION_ROOT.rglob('*')):
        if not path.is_file() or path.suffix not in ('.py', '.yaml', '.sh'):
            continue
        name = path.relative_to(REPO_ROOT).as_posix()
        content = path.read_bytes()
        files[name] = hashlib.sha256(content).hexdigest()
        digest.update(name.encode())
        digest.update(content)
    return digest.hexdigest(), files


def source_fingerprint():
    core = reproducibility.source_sha256()
    extension, files = extension_fingerprint()
    combined = hashlib.sha256(json.dumps(
        {'core_source_sha256': core, 'extension_sha256': extension}, sort_keys=True,
    ).encode()).hexdigest()
    return {'source_sha256': combined, 'core_source_sha256': core,
            'extension_sha256': extension, 'extension_files_sha256': files}


def collect_metadata(config, variant, seed, device):
    metadata = CORE_COLLECT_METADATA(config, variant, seed, device)
    if variant == 'F':
        metadata.update(source_fingerprint())
        metadata['experiment_definition'] = {
            'name': 'Teacher-forced Stage-wise Ensemble', 'ensemble_size': 3,
            'dynamics_implementation': 'unchanged clarity_rrt_v3.model.StagewiseDynamics (C/D implementation)',
            'shared_modules': ['MRI encoder', 'text encoder', 'survival head'],
            'training_scheme': 'teacher_forced_stagewise', 'rrt_training': False,
            'teacher_inputs': ['s0', 'stopgrad(s1_true)', 'stopgrad(s2_true)'],
            'targets_detached': True, 'evaluation': 'independent member recursive rollout, then equal-weight fusion',
        }
    return metadata


def install_runtime():
    """Extend this Python process only; no core file or A–E variant map is changed."""
    train.assert_experiment_design = assert_design
    evaluate.assert_experiment_design = assert_design
    train.collect_metadata = collect_metadata
    evaluate.collect_metadata = collect_metadata
