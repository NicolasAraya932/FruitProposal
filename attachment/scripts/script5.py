import torch

ckpt = torch.load("/workspace/FruitProposal/outputs/SynthBinRGBData/fruit-proposal/2025-05-27_010158/nerfstudio_models/step-000000799.ckpt", map_location="cpu")

print(ckpt["pipeline"]["_model.fruit_proposal_field.aabb"])