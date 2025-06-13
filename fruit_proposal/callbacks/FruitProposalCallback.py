import torch
from nerfstudio.engine.callbacks import TrainingCallbackAttributes

class FruitEarlyStopCallback:
    def __init__(self, stop_step: int = 800, save_path: str = "./fruit_nerfacto.pt"):
        self.stop_step = stop_step
        self.save_path = save_path
        self.triggered = False

    def __call__(self, attrs: TrainingCallbackAttributes, step: int):
        # Only fire once we hit stop_step
        if self.triggered or step < self.stop_step:
            return
        self.triggered = True

        model = attrs.pipeline.model  # your Pipeline → Model reference:contentReference[oaicite:4]{index=4}

        # Freeze fruit field
        for p in model.fruit_proposal_field.parameters():
            p.requires_grad = False

        ray_bundle, batch = attrs.pipeline.datamanager.next_train(0)

        # Snapshot model outputs (e.g. semantic labels, RGB)
        outputs = model.get_outputs(ray_bundle)

        torch.save({
            "step": step,
            "semantic_labels": outputs["semantic_labels"].cpu(),
            "rgb": outputs["rgb"].cpu(),
        }, self.save_path)

        print(f"[FruitEarlyStop] step={step}: frozen & saved → {self.save_path}")
