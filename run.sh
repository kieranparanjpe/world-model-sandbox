#!/usr/bin/env bash

COMMAND=$1

if [ "$COMMAND" = "dataset_from_mdp" ]; then
    uv run python -m src.datasets.dataset_from_mdp \
        --environment="LunarLander-v3" \
        --timesteps=300000 \
        --policy="categorical" \
        --weights="/home/kieran/coding-projects/My-RL-Impl/saved_policies/LunarLander-v3/LunarLander-v3@2026-07-12-17-18-42/policy_1999999.pth"
elif [ "$COMMAND" = "train_encoder" ]; then
    uv run python -m src.train \
        --environment="LunarLander-v3" \
        --algorithm="jepa" \
        --dataset="datasets/LunarLander-v3/state-action/dataset_2026-08-25-20-31-29.pt" \
        --grid="hyperparameters/jepa_lunar_lander_grid.json" \
        --normalise-obs \
        -dt="sa" -l -s
elif [ "$COMMAND" = "dataset_from_encoder" ]; then
    uv run python -m src.datasets.dataset_from_encoder \
        --environment="LunarLander-v3" \
        --dataset="datasets/LunarLander-v3/state-action/dataset_2026-08-25-20-31-29.pt" \
        --model="saved_networks/jepa/model/LunarLander-v3/LunarLander-v3@2026-08-25-20-46-18/LunarLander-v3@2026-08-25-20-46-18_RUN-0/model_083.pt"
elif [ "$COMMAND" = "train_decoder" ]; then
    uv run python -m src.train \
        --environment="LunarLander-v3" \
        --algorithm="jepa_decoder" \
        --dataset="datasets/LunarLander-v3/decoder/dataset_2026-08-25-21-02-25.pt" \
        --hyperparameters="hyperparameters/jepa_decoder_lunar_lander.json" \
        --dataset_type="encoder" -l -s
elif [ "$COMMAND" = "visualise" ]; then
    uv run python -m src.visualise \
        --environment="LunarLander-v3" \
        --model="saved_networks/jepa/model/LunarLander-v3/LunarLander-v3@2026-08-25-20-46-18/LunarLander-v3@2026-08-25-20-46-18_RUN-0/model_083.pt" \
        --decoder="saved_networks/jepa/decoder/LunarLander-v3/LunarLander-v3@2026-08-25-21-02-50/decoder_89.pt" \
        --policy="categorical" \
        --weights="/home/kieran/coding-projects/My-RL-Impl/saved_policies/LunarLander-v3/LunarLander-v3@2026-07-12-17-18-42/policy_1999999.pth" \
        --sync \
        --sync_timesteps=5
elif [ "$COMMAND" = "evaluate" ]; then
    uv run python -m src.evaluate \
        --environment="<gymnasium environment id>" \
        --algorithm="jepa" \
        --dataset="path/to/dataset.pt" \
        --model="path/to/saved/jepa/model.pt" \
        --hyperparameters="path/to/hyperparameters.json" \
        --dataset_type="sa"
else
    echo "Usage: $0 <command>"
    echo "Available commands:"
    echo "  dataset_from_mdp      - Generate a dataset from the MDP"
    echo "  train_encoder         - Train the JEPA encoder"
    echo "  dataset_from_encoder  - Generate an encoder dataset (for decoder training)"
    echo "  train_decoder         - Train the JEPA decoder"
    echo "  visualise             - Visualise a trained model"
    echo "  evaluate              - Evaluate a trained model"
    exit 1
fi