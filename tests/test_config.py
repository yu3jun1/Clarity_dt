from pathlib import Path

import pytest

from clarity_rrt_v3.train import (
    FACTORIAL_VARIANTS,
    assert_factorial_design,
    assert_upstream_commit,
    load_config,
    upstream_args,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "pure_rrt_v3.yaml"
UPSTREAM_COMMIT = "dadb82241a24f5ec5e4e4dc994e3116fd4a9da04"


def test_pure_rrt_v3_config_builds_upstream_args():
    config = load_config(CONFIG_PATH)
    args = upstream_args(config, seed=42)

    assert args.lambda_l1 == 0.5
    assert args.lambda_cox == 1.0
    assert args.lambda_bce == 1.0
    assert config["training"]["batch_size"] == 16
    assert config["training"]["evaluation_batch_size"] == 16
    assert args.seed == 42
    assert config["upstream_commit"] == UPSTREAM_COMMIT
    assert config["variants"] == FACTORIAL_VARIANTS
    assert_factorial_design(config)
    assert config["output_root"] == "outputs/pure_rrt_v3_clarity_allpair"
    assert assert_upstream_commit(config) == UPSTREAM_COMMIT


def test_factorial_definition_rejects_variant_drift():
    config = load_config(CONFIG_PATH)
    config["variants"]["C"]["training_scheme"] = "open_loop"

    with pytest.raises(AssertionError, match="frozen 2x2"):
        assert_factorial_design(config)
