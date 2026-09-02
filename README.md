# world-model-sandbox

## Project Description

The goal of this project is to experiment with world models, specifically action conditioned JEPA models. This builds
on a [previous project I did](https://github.com/kieranparanjpe/My-RL-Impl), where I implemented PPO and trained some
policies to complete various [gymnasium](https://gymnasium.farama.org/index.html) tasks. This project uses the same environments and policies, but it flips 
the goal. The RL project learned a policy to map from S -> A (state to action), and then evaluated performance by 
taking those actions in the physics simulator backed environment mdp. This project, aims to replace the physics 
simulator backed environment with a learned representation of the environment. So to compare:

**RL (1):**
```
s_t -> Policy(s_t) -> a_t -> Take a_t in physics sim backed environment -> s_{t+1} -> repeat
                             └───────────── learned/focus ────────────┘
```

**World Model (2):**
```
s_t -> Policy(s_t) -> a_t -> World Model(s_t, a_t) -> s_{t+1} -> repeat
                             └── learned/focus ──┘
```

### Why do we do this?

Consider a policy running on a robot in the real world. In this case, the simulator in (1) is replaced by the real 
world. You can see that the only way to find s_{t+1} is to actually take the action and see what happens. This could 
potentially be problematic, because the robot has no way of assessing the consequences of its actions before taking 
them. In contrast, with the world model, we can take some candidate action a_t, plug it into our world model, and 
get an estimation of the next state, s_{t+1}, without taking an action in the real world. 

### Technical Overview

I used JEPA (Joint Embedding Predictive Architecture) for this world model. JEPA is made up of 2 parts: the encoder 
and the predictor. Given some state `s`, we first encode it to be `z = E(s)`. We then pass this encoding and the 
action to the predictor to get the encoding of the next state `z' = P(E(s), a)`. The idea is that the encoder will 
figure out which features of the state are important, and represent it in the latent space. Then, we predict only in 
this latent space. Finally, to do visualisation, we train a decoder which allows us to go from encoding space to 
state space: `s' = D(z')`. 

This project draws heavily from [LeWorldModel](https://arxiv.org/abs/2603.19312), which similarly uses JEPA world 
models. In the paper, their state representation is always an image. This highlights one of the key benefits of JEPA,
which is to encode high dimensional state (like images) into a lower dimension embedding. 

For this project, I do not use images for state. Instead, I use the Markovian states supplied by the MDPs. To be honest,
this project doesn't really gain a lot from using JEPA, as the encoding is actually higher dimensional than the 
Markovian state. I'm not using images because I don't have access to large amounts of compute, required for higher 
dimension states, and I wanted to keep the project simple. However, I still wanted to implement JEPA to better 
understand it.


## Logging

I use Weights & Biases to log training. The dashboard I used can be found [here](https://wandb.ai/kieranparanjpe-mcgill-university/World-Model-Sandbox).

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

**1. Install uv** (if you don't have it):
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**2. Clone the repository:**
```
git clone https://github.com/kieranparanjpe/world-model-sandbox.git
cd world-model-sandbox
```

**3. Create the virtual environment and install dependencies:**
```
uv sync
```
This automatically creates a `.venv` and installs all pinned dependencies.

**4. Activate the virtual environment:**
```
source .venv/bin/activate
```


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

I typically use `run.sh` to run my experiments, because there is less to copy & paste each time.
