from __future__ import annotations

import copy
import importlib.util

import pytest
import torch

from clarity_rrt_v3 import evaluate, reproducibility, train
from clarity_rrt_v3.model import StagewiseDynamics
from clarity_tf_ensemble import protocol
from clarity_tf_ensemble.__main__ import gpu_ready


@pytest.fixture
def f_config():
    return train.load_config(protocol.DEFAULT_CONFIG)


@pytest.fixture
def tiny():
    spec = importlib.util.spec_from_file_location(
        'clarity_f_tiny_fixtures', protocol.REPO_ROOT / 'tests/test_model.py',
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_f_retains_e_with_exactly_three_members(f_config):
    protocol.assert_design(f_config)
    baseline = train.load_config(protocol.REPO_ROOT / 'configs/ablations/teacher_forced_stagewise.yaml')
    for field in ('data', 'model', 'training', 'upstream_root', 'upstream_commit'):
        assert f_config[field] == baseline[field]
    assert f_config['variants']['F'] == {
        'training_scheme': 'teacher_forced_stagewise', 'ensemble_size': 3,
    }


@pytest.mark.parametrize('drift', ['recursive', 'one_member', 'steps', 'cache', 'nondeterministic'])
def test_f_rejects_protocol_drift(f_config, drift):
    config = copy.deepcopy(f_config)
    if drift == 'recursive':
        config['variants']['F']['training_scheme'] = 'pure_recursive'
    elif drift == 'one_member':
        config['variants']['F']['ensemble_size'] = 1
    elif drift == 'steps':
        config['training']['total_steps'] = 1200
    elif drift == 'cache':
        config['data']['mri_cache_dir'] = '/tmp/wrong-cache'
    else:
        config['deterministic'] = False
    with pytest.raises(AssertionError):
        protocol.assert_design(config)


def test_e_design_still_rejects_ensemble():
    config = train.load_config(protocol.REPO_ROOT / 'configs/ablations/teacher_forced_stagewise.yaml')
    protocol.assert_design(config)
    config['variants']['E']['ensemble_size'] = 3
    with pytest.raises(AssertionError):
        protocol.assert_design(config)


def test_registration_is_process_local_and_does_not_change_core(monkeypatch, f_config):
    before = reproducibility.source_sha256()
    variant_maps = (copy.deepcopy(train.MAIN_FACTORIAL_VARIANTS), copy.deepcopy(train.TF_ABLATION_VARIANTS))
    with monkeypatch.context() as patch:
        for module, name in ((train, 'assert_experiment_design'), (evaluate, 'assert_experiment_design'),
                             (train, 'collect_metadata'), (evaluate, 'collect_metadata')):
            patch.setattr(module, name, getattr(module, name))
        protocol.install_runtime()
        train.assert_experiment_design(f_config)
        evaluate.assert_experiment_design(f_config)
        assert train.collect_metadata is protocol.collect_metadata
    assert train.MAIN_FACTORIAL_VARIANTS == variant_maps[0]
    assert train.TF_ABLATION_VARIANTS == variant_maps[1]
    assert reproducibility.source_sha256() == before


def test_f_metadata_covers_core_and_extension(monkeypatch, f_config):
    monkeypatch.setattr(protocol, 'CORE_COLLECT_METADATA', lambda *args: {'gpu': {'name': 'test-GPU'}})
    metadata = protocol.collect_metadata(f_config, 'F', 42, 'cpu')
    assert metadata['core_source_sha256'] == reproducibility.source_sha256()
    assert metadata['source_sha256'] != metadata['core_source_sha256']
    assert any(name.endswith('configs/F.yaml') for name in metadata['extension_files_sha256'])
    assert metadata['experiment_definition']['rrt_training'] is False
    assert metadata['experiment_definition']['ensemble_size'] == 3


def test_f_uses_c_d_ensemble_with_shared_encoders(tiny):
    clarity = tiny.TinyClarity()
    model = StagewiseDynamics(clarity, ensemble_size=3, seed=42)
    assert len(model.predictors) == 3
    assert model.mri_encoder is clarity.mri_encoder
    assert model.text_encoder is clarity.shared_text_encoder
    assert model.survival_module is clarity.survival_module
    members = [{id(parameter) for parameter in predictor.parameters()} for predictor in model.predictors]
    assert not (members[0] & members[1] or members[0] & members[2] or members[1] & members[2])


@pytest.mark.parametrize('compute_cf', [False, True])
def test_f_teacher_forcing_all_members_no_rrt_and_detached_targets(tiny, f_config, monkeypatch, compute_cf):
    model = StagewiseDynamics(tiny.TinyClarity(), ensemble_size=3, seed=42)
    trainer = object.__new__(train.Trainer)
    trainer.model = model
    trainer.device = torch.device('cpu')
    trainer.variant = 'F'
    trainer.config = f_config
    mri = torch.randn(2, 4, 5, 3, requires_grad=True)
    batch = {
        'mri': mri,
        'deltas': torch.tensor([[10., 20., 30.], [11., 21., 31.]]),
        'survival_time': torch.tensor([[800., 700., 600.], [700., 600., 500.]]),
        'event': torch.ones(2, 3),
        'full_text': ['full treatment one', 'full treatment two'],
        'step_text': [
            [tiny.interval_text('Temozolomide'), tiny.interval_text('Avastin')],
            [tiny.interval_text('Avastin'), tiny.interval_text('Lomustine')],
            [tiny.interval_text('Lomustine'), tiny.interval_text(None)],
        ],
        'clinical_text': ['clinical one', 'clinical two'],
    }

    def forbidden_rollout(*args, **kwargs):
        raise AssertionError('F must not use recursive training')

    monkeypatch.setattr(model, 'rollout', forbidden_rollout)
    total, values, risks = trainer.step(batch, optimizer_step=1, compute_cf=compute_cf)
    total.backward()
    assert set(values) == {'loss', 'latent', 'cox', 'bce', 'cf'}
    assert risks.shape == (2,)
    for predictor in model.predictors:
        assert len(predictor.calls) == 3
        for stage in range(3):
            torch.testing.assert_close(predictor.calls[stage], mri[:, stage])
    assert mri.grad[:, 0].abs().sum() > 0
    torch.testing.assert_close(mri.grad[:, 1:], torch.zeros_like(mri.grad[:, 1:]))
    if compute_cf:
        assert len(model.clarity.cf_calls) == 9
        for index, call in enumerate(model.clarity.cf_calls):
            torch.testing.assert_close(call['pre_latent'], mri[:, index % 3])


def test_gpu_sharing_requires_approval_and_memory_headroom():
    busy = {'memory_used_mib': 30000, 'memory_free_mib': 67000, 'utilization_percent': 100}
    assert not gpu_ready(busy, allow_shared=False)
    assert gpu_ready(busy, allow_shared=True)
    assert not gpu_ready(dict(busy, memory_free_mib=59000), allow_shared=True)
    idle = {'memory_used_mib': 32, 'memory_free_mib': 97000, 'utilization_percent': 0}
    assert gpu_ready(idle, allow_shared=False)
