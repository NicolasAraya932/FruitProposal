# fruit_proposal/method_configs.py
from nerfstudio.configs.method_configs import method_configs, TrainerConfig
from nerfstudio.configs.pipeline import VanillaPipelineConfig
from nerfstudio.configs.data import VanillaDataManagerConfig
from nerfstudio.configs.viewer import ViewerConfig
from nerfstudio.configs.optimizers import AdamOptimizerConfig, ExponentialDecaySchedulerConfig
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackLocation

# Import your custom pieces
from fruit_proposal.data.fruit_proposal_dataparser import FruitProposalDataParserConfig
from fruit_proposal.fruit_proposal import FruitProposalModelConfig
from fruit_proposal.callbacks import SemanticStageCallback

method_configs["fruit-proposal"] = TrainerConfig(
    method_name="fruit-proposal",
    steps_per_eval_batch=20,
    steps_per_save=500,
    max_num_iterations=100_000,
    mixed_precision=True,
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=FruitProposalDataParserConfig(),
        ),
        model=FruitProposalModelConfig(),
        callbacks=[
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=SemanticStageCallback(stop_step=800),
            ),
        ],
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=5e-3, eps=1e-8),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=100_000),
        },
        "nerfacto_fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-8),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-4, max_steps=100_000),
        },
        "fruit_proposal_fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-8),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-4, max_steps=100_000),
        },
        "camera_opt": {
            "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-8),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=10_000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 12),
    vis="viewer",
)
