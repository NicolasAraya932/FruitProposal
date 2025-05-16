from nerfstudio.engine.callbacks import TrainingCallbackAttributes

class SemanticStageCallback:
    def __init__(self, stop_step: int):
        self.stop_step = stop_step
        self.triggered = False

    def __call__(self, attrs: TrainingCallbackAttributes, step: int):
        # Only trigger once
        #print(f"SemanticStageCallback: step {step}, triggered {self.triggered}, attrs {attrs}")

        if step < self.stop_step or self.triggered:
            return

        model = attrs.trainer.model
        optim = attrs.optimizers.optimizer

        # 1) Freeze all FruitProposalField parameters
        for p in model.fruit_proposal_field.parameters():
            p.requires_grad = False

        # 2) Remove its param group by name
        new_groups = [
            g for g in optim.param_groups
            if not g.get("name", "").startswith("fruit_proposal_fields")
        ]
        optim.param_groups = new_groups

        self.triggered = True
        print(f"SemanticStageCallback: frozen FruitProposalField at step {step}")
