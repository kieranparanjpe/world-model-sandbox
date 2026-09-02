#!/usr/bin/env bash

COMMAND=$1

if [ "$COMMAND" = "dataset_from_mdp" ]; then
    uv run python -m src.datasets.dataset_from_mdp \
        --environment="Walker2d-v5" \
        --timesteps=300000 \
        --policy="single_beta" \
        --weights="/home/kieran/coding-projects/My-RL-Impl/saved_policies/Walker2d-v5/Walker2d-v5@2026-06-24-17-23-44/policy_1999999.pth"
#        --weights="/home/kieran/coding-projects/My-RL-Impl/saved_policies/Humanoid-v5/Humanoid-v5@2026-08-18-00-08-30/policy_5999999.pth"
#        --policy="categorical" \
#        --weights="/home/kieran/coding-projects/My-RL-Impl/saved_policies/LunarLander-v3/LunarLander-v3@2026-07-12-17-18-42/policy_1999999.pth"
elif [ "$COMMAND" = "train_encoder" ]; then
    uv run python -m src.train \
        --environment="Walker2d-v5" \
        --algorithm="jepa" \
        --dataset="datasets/Walker2d-v5/state-action/dataset_2026-08-30-22-06-41.pt" \
        --grid="hyperparameters/jepa_walker2d_grid.json" \
        --normalise-obs \
        -dt="sa" -l -s
elif [ "$COMMAND" = "dataset_from_encoder" ]; then
    uv run python -m src.datasets.dataset_from_encoder \
        --environment="Humanoid-v5" \
        --dataset="datasets/Humanoid-v5/state-action/dataset_2026-08-30-22-05-51.pt" \
        --model="saved_networks/jepa/model/Humanoid-v5/Humanoid-v5@2026-08-30-22-10-49/Humanoid-v5@2026-08-30-22-10-49_RUN-2/model_149.pt"
elif [ "$COMMAND" = "train_decoder" ]; then
    uv run python -m src.train \
        --environment="Humanoid-v5" \
        --algorithm="jepa_decoder" \
        --dataset="datasets/Humanoid-v5/decoder/dataset_2026-08-31-22-45-25.pt" \
        --hyperparameters="hyperparameters/jepa_decoder_lunar_lander.json" \
        --normalise-obs \
        --dataset_type="encoder" -l -s
elif [ "$COMMAND" = "visualise" ]; then
# Walker2d
    uv run python -m src.visualise \
        --environment="Walker2d-v5" \
        --model="saved_networks/jepa/model/Walker2d-v5/Walker2d-v5@2026-08-30-22-11-39/Walker2d-v5@2026-08-30-22-11-39_RUN-2/model_149.pt" \
        --decoder="saved_networks/jepa/decoder/Walker2d-v5/Walker2d-v5@2026-08-31-21-44-10/decoder_89.pt" \
        --policy="single_beta" \
        --weights="/home/kieran/coding-projects/My-RL-Impl/saved_policies/Walker2d-v5/Walker2d-v5@2026-06-24-17-23-44/policy_1999999.pth" \
        --sync
# Humanoid
#        --environment="Humanoid-v5" \
#        --model="saved_networks/jepa/model/Humanoid-v5/Humanoid-v5@2026-08-30-22-10-49/Humanoid-v5@2026-08-30-22-10-49_RUN-2/model_149.pt" \
#        --decoder="saved_networks/jepa/decoder/Humanoid-v5/Humanoid-v5@2026-08-31-22-46-02/decoder_89.pt" \
#        --policy="single_beta" \
#        --weights="/home/kieran/coding-projects/My-RL-Impl/saved_policies/Humanoid-v5/Humanoid-v5@2026-08-18-00-08-30/policy_5999999.pth" \
#        --sync
# Lunar Lander
#        --environment="StaticLunarLander-v0" \
#        --model="saved_networks/jepa/model/LunarLander-v3/LunarLander-v3@2026-08-28-18-43-46/LunarLander-v3@2026-08-28-18-43-46_RUN-2/model_149.pt" \
#        --decoder="saved_networks/jepa/decoder/LunarLander-v3/LunarLander-v3@2026-08-30-14-55-41/decoder_89.pt" \
#        --policy="categorical" \
#        --weights="/home/kieran/coding-projects/My-RL-Impl/saved_policies/LunarLander-v3/LunarLander-v3@2026-07-12-17-18-42/policy_1999999.pth" \
#        --sync
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

#        --model="saved_networks/jepa/model/LunarLander-v3/LunarLander-v3@2026-08-28-18-43-46/LunarLander-v3@2026-08-28-18-43-46_RUN-1/model_149.pt" \
#        --decoder="saved_networks/jepa/decoder/LunarLander-v3/LunarLander-v3@2026-08-30-14-54-29/decoder_89.pt" \