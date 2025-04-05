# Mario Brain: Deep Reinforcement Learning with PPO

Mario Brain trains an AI agent to play Super Mario Bros using deep reinforcement learning.

![Mario Gameplay](./videos/mario_gameplay.gif)

## Installation

Clone the repository and create a virtual environment, and install the required packages.

```bash
git clone https://github.com/akanto/mario-brain.git
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

You can download the trained model from the Hugging Face Model Hub by adding the submodule to your project:

```bash
 git submodule add https://huggingface.co/akantox/mario-rl-model models
```

## Training

Train the agent from scratch using the following command, the trained model will be saved in the `models/` directory:

```bash
python mario_brain/train.py
```

Or if you wwish to launch in the background, you can use the following command:

```bash
nohup python mario_brain/train.py --parallel 4 --timesteps 50_000_000 > train.log 2>&1 &
```

## Evaluation

Evaluate the trained model to see how well it performs, it will load the model from the `models/` directory. The evaluation also renders the gameplay, so you can watch the AI play:

```bash
python mario_brain/evaluate.py
```

## Random Play

If you want to see some gamplay without AI, then you can run the random play script:

```bash
python mario_brain/random_play.py
```

## Human Play

If you want to play the game yourself, you can use the human play script. It will allow you to control the game using your keyboard:

```bash
python mario_brain/human_play.py
```

## Benchmarking

Benchmark contains a few scripts to test PyTorch and Gymasium performance on your machine. It does not provied any useful information about the training process, but it can be used to test the performance of your machine or test wether cuda or mps is working properly.

```bash
python mario_brain/benchmark.py
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

Some of the libraries used in this project are not compatible with the latest versions of NumPy and OpenAI's Gymnasium, therfore some of the libs are pinned to an old versions in the `requirements.txt` file.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
