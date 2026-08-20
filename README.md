# world-model-sandbox
Experimenting with world models

## Running

#### 1. Generate a dataset from the MDP
```
uv run python -m src.datasets.dataset_from_mdp \
    --environment="<gymnasium environment id>" \
    --timesteps="<number of samples to collect>"
```

#### 2. Train the JEPA encoder
```
uv run python -m src.train \
    --environment="<gymnasium environment id>" \
    --algorithm="jepa" \
    --dataset="path/to/state-action/dataset.pt" \
    --hyperparameters="path/to/hyperparameters.json" \
    -dt="sa" -l -s
```

#### 3. Generate an encoder dataset (for decoder training)
```
uv run python -m src.datasets.dataset_from_encoder \
    --environment="<gymnasium environment id>" \
    --dataset="path/to/state-action/dataset.pt" \
    --model="path/to/saved/jepa/model.pt"
```

#### 4. Train the JEPA decoder
```
uv run python -m src.train \
    --environment="<gymnasium environment id>" \
    --algorithm="jepa_decoder" \
    --dataset="path/to/encoder/dataset.pt" \
    --hyperparameters="path/to/hyperparameters.json" \
    --dataset_type="encoder" -l -s
```

#### 5. Visualise a trained model
```
uv run python -m src.visualise \
    --environment="<gymnasium environment id>" \
    --model="path/to/saved/jepa/model.pt" \
    --decoder="path/to/saved/jepa/decoder.pt" \
    --sync
```

#### 6. Evaluate a trained model
```
uv run python -m src.evaluate \
    --environment="<gymnasium environment id>" \
    --algorithm="jepa" \
    --dataset="path/to/dataset.pt" \
    --model="path/to/saved/jepa/model.pt" \
    --hyperparameters="path/to/hyperparameters.json" \
    --dataset_type="sa"
```
