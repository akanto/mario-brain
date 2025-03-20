# Mario Brain: Deep Reinforcement Learning with PPO

Mario Brain trains an AI agent to play Super Mario Bros using deep reinforcement learning.

![Mario Gameplay](./videos/mario_gameplay.gif)

## Installation

Clone the repository and create a virtual environment, and install the required packages.

```bash
git clone git@github.com:akanto/mario-brain.git
cd mario-brain
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Change the working directory to the `src` folder:

```bash
cd src
```

Train the agent from scratch using the following command, the trained model will be saved in the `../models/` directory:

```bash
python train.py
```

Evaluate the trained model to see how well it performs, it will load the model from the `../models/` directory. The evaluation also renders the gameplay, so you can watch the AI play:

```bash
python evaluate.py
```

## Logging and Monitoring

Training logs and metrics are stored in the `../logs/` directory. Launch TensorBoard to monitor progress:

```bash
tensorboard --logdir ../logs/PPO_1
```

Open the link in your browser (http://localhost:6006/) to view real-time metrics.

## References

This repo was inspired by [Reinforcement Learning for Gaming](https://youtu.be/dWmJ5CXSKdw).

## Compatibility Issues

Some of the libraries used in this project are not compatible with the latest versions of NumPy and OpenAI's Gymnasium, therfore some of the libs are pinned to an old versions in the `requirements.txt` file.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
