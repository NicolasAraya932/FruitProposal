from nerfstudio.engine.callbacks import TrainingCallbackAttributes

class SemanticStageCallback:
    def __init__(self, stop_step: int):
        self.stop_step = stop_step
        self.triggered = False

    def __call__(self, attrs: TrainingCallbackAttributes, step: int):
        # Only trigger once after stop_step
        if step < self.stop_step or self.triggered:
            return

        # 1) Freeze semantic field parameters
        model = attrs.pipeline.model   # ← correct access to your FruitProposalModel
        for p in model.fruit_proposal_field.parameters():
            p.requires_grad = False

        # 2) Remove its optimizer & scheduler
        # Nerfstudio’s Optimizers creates one optimizer per param group name
        removed_opt = attrs.optimizers.optimizers.pop("fruit_proposal_fields", None)
        removed_sched = attrs.optimizers.schedulers.pop("fruit_proposal_fields", None)
        if removed_opt:
            print("Removed semantic optimizer")
        if removed_sched:
            print("Removed semantic scheduler")

        self.triggered = True
        print(f"SemanticStageCallback: frozen FruitProposalField at step {step}")