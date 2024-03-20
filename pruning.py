import torch
import torch.nn as nn
import yaml
import os
import argparse

from models.yolo import Model
from utils.pruning import eval_l1_sparsity, prune, transfer_weights
from utils.torch_utils import intersect_dicts

def pruning(checkpoint_path, output_directory="pruned_model", prune_ratio, compute_device="cuda:0"):
    '''
    Prune the model based on L1 norm sparsity and prune_ratio.
    
    Args:
        checkpoint_path (str): Path to the model checkpoint.
        output_directory (str): Directory to save the pruned model and its configuration.
        prune_ratio (float): Ratio of weights to prune.
        compute_device (str): Device.
    '''
    # load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=compute_device)
    # Initialize model from checkpoint
    model = Model(checkpoint['model'].yaml).to(compute_device)
    # Convert to FP32 and align state dicts
    state_dict = checkpoint['model'].float().state_dict()
    new_state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=[])
    model.load_state_dict(new_state_dict, strict=False)
    # Evaluate current sparsity and prune
    current_sparsity = eval_l1_sparsity(model)
    %print(current_sparsity)
    pruning_mask, updated_cfg = prune(model, prune_ratio)

    # define directory for saving pruned model
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    config_filename = "pruned_{}_config.yaml".format(prune_ratio)

    with open(os.path.join(output_directory, config_filename), "w") as file:
        yaml.safe_dump(updated_cfg, file, sort_keys=False)

    # Initialize pruned model and transfer weights
    pruned_model = Model(updated_cfg).to(compute_device)
    pruned_model = transfer_weights(model, pruned_model, pruning_mask)

    # forward pass to test
    test_input = torch.rand(1, 3, 640, 640).to(compute_device)
    pruned_model.eval()
    test_output = pruned_model(test_input)
    print(test_output[0].shape)

    checkpoint["model"] = pruned_model
    checkpoint["best_fitness"] = 0.0
    pruned_model_name = "pruned_{}_model.pt".format(prune_ratio)
    torch.save(checkpoint, os.path.join(output_directory, pruned_model_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prune YOLOv5 model.")
    parser.add_argument('--weight', type=str, help='Path to YOLOv5 checkpoint.')
    parser.add_argument('--save_dir', type=str, default="pruned_model", help='Directory to save pruned model and config.')
    parser.add_argument('--prune_ratio', type=float, default=0.7)
    args = parser.parse_args()
    pruning(args.weight, args.save_dir, args.prune_ratio)