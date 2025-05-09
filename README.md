# Mario Brain: Deep Reinforcement Learning with PPO

Mario Brain trains an AI agent to play Super Mario Bros using deep reinforcement learning.

![Mario Gameplay](./videos/mario_gameplay.gif)

## Installation

Clone the repository and create a virtual environment, and install the required packages.

```bash
git clone --recurse-submodules https://github.com/akanto/mario-brain.git
cd mario-brain
python3 -m venv .venv
source .venv/bin/activate
```

Install requirements if you have MacOS wih mps:

```bash
pip install -r requirements.txt
```

Install reqirements if you have Nvidia GPU (e.g with CUDA 12.6):

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
```

## Hugging Face Model Hub

The `models` directory is submodule and it is linked to [mario-rl-model](https://huggingface.co/akantox/mario-rl-model) repository on Hugging Face. The `git clone --recurse-submodules https://github.com/akanto/mario-brain.git` command will automatically clone the model repository as well.

## Training

Train the agent from scratch using the following command, the trained model will be saved in the `models/` directory:

```bash
python -m mario_brain.train
```

Or if you wwish to launch in the background, you can use the following command:

```bash
nohup python -m mario_brain.train --parallel 4 --timesteps 50_000_000 > train.log 2>&1 &
```

## Evaluation

Evaluate the trained model to see how well it performs, it will load the model from the `models/` directory. The evaluation also renders the gameplay, so you can watch the AI play:

```bash
python -m mario_brain.evaluate
```

## Random Play

If you want to see some gamplay without AI, then you can run the random play script:

```bash
python -m mario_brain.random_play
```

## Human Play

If you want to play the game yourself, you can use the human play script. It will allow you to control the game using your keyboard:

```bash
python -m mario_brain.human_play
```

## Benchmarking

Benchmark contains a few scripts to test PyTorch and Gymasium performance on your machine. It does not provied any useful information about the training process, but it can be used to test the performance of your machine or test wether cuda or mps is working properly.

```bash
python -m mario_brain.benchmark
```

## Logging and Monitoring

Training logs and metrics are stored in the `../logs/` directory. Launch TensorBoard to monitor progress:

```bash
tensorboard --logdir logs/PPO_1
```

Open the link in your browser (http://localhost:6006/) to view real-time metrics.

## References

This was inspired by the following resources:

- [Reinforcement Learning for Gaming](https://youtu.be/dWmJ5CXSKdw)
- [Proximal Policy Optimization](https://youtu.be/5P7I-xPq8u8)
- [Policy Gradient Theorem Explained](https://youtu.be/cQfOQcpYRzE)

## Compatibility Issues

Some of the libraries used in this project are not compatible with the latest versions of NumPy and OpenAI's Gymnasium, therfore those libraries have been forked and the git repos were added to the `requirements.txt`.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
