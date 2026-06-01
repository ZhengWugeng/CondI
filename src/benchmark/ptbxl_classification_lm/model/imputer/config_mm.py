def get_config():
    config = {
        "model": {
            "timeemb": 128,
            "featureemb": 64,
            "is_unconditional": False,
            "target_strategy": "random",
        },
        "diffusion": {
            "num_steps": 50,
            "schedule": "linear",
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "channels": 64,
            "layers": 4,
            "nheads": 8,
            "is_linear": False,
            "diffusion_embedding_dim": 128,
        },
        "multimodal": {
            "ctx_dim": 128,
            "ctx_hidden": 256,
            "num_experts": 3,
        },
        "data": {
            "text_dim": 768,
            "ecg_dim": 256,
            "cxr_dim": 1024,
            "eval_mask_ratio": 0.2,
        },
        "train": {
            "seed": 42,
            "batch_size": 32,
            "lr": 1.0e-4,
            "weight_decay": 1e-6,
            "max_epochs": 50,
            "grad_clip": 1.0,
            "num_workers": 0,
            "log_every": 50,
        },
    }
    return config
