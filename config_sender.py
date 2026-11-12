configurations = {
    "receiver": {
        "host": "192.168.1.2",
        "port": 50028
    },
    "max_file_chunk_in_MB": 512, # MB
    "data_dir": "/mnt/nvme0n1/src/",
    "B": 10, # severity of the packet loss punishment
    "K": 1.02, # cost of increasing concurrency
    "loglevel": "info",
    "probing_sec": 5, # probing interval in seconds
    "network_limit": 2000, # Network limit (Mbps) per thread
    "io_limit": -1, # I/O limit (Mbps) per thread
    "fixed_probing": {
        "bsize": 10,
        "thread": 8
    },
    "bayes": {
        "initial_run": 3,
        "num_of_exp": -1 #-1 for infinite
    },
    "max_cc": {
        "network": 100,
        "io": 30,
    },
    "file_transfer": False,
    "method": "ppo", # options: [gradient, ppo]
    "model_version": 'heur_uniform_15', # just a tag for the model
    "mode": 'inference', # random or inference
}

# Optional per-run overrides for experiment scripts (see experiments/chunk_sweep.py), e.g.
#   LDM_CONFIG_OVERRIDES='{"max_file_chunk_in_MB": 256, "mode": "random"}'
# Top-level keys only. Applied before the derived defaults below, so overriding
# "mode" or "model_version" also updates the model paths, max_episodes and multiplier.
import json as _json
import os as _os
if _os.environ.get("LDM_CONFIG_OVERRIDES"):
    configurations.update(_json.loads(_os.environ["LDM_CONFIG_OVERRIDES"]))

mv = configurations["model_version"]
configurations.setdefault("inference_value_model",
                          f"best_models/{mv}_offline_value.pth")
configurations.setdefault("inference_policy_model",
                          f"best_models/{mv}_offline_policy.pth")
configurations.setdefault("max_episodes",
                            6 if configurations["mode"] == "random" else 20000)
configurations.setdefault("multiplier",
                            20 if configurations["mode"] == "random" else 16)