from pathlib import Path

from clarity_rrt_v3.train import assert_upstream_commit, load_config, upstream_args


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
    assert config["variants"]["A"]["training_scheme"] == "clarity_all_pair"
    assert config["output_root"] == "outputs/pure_rrt_v3_clarity_allpair"
    assert assert_upstream_commit(config) == UPSTREAM_COMMIT
