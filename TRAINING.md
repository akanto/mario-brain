# PPO Training Step Summary

## Initialization

- The model weights are randomly initialized at the start of training
- The environment is created using the SuperMarioBros-v0 environment from the OpenAI Gym library.
- The environment is wrapped with the following wrappers:
  - GrayScaleObservation: Converts the RGB image to grayscale
  - FrameStack: Stacks four consecutive frames to form a single observation

Note: consider to use SuperMarioBros-v3 and with ResizingWrapper

## Observation Processing

- The environment outputs a game frame, which is a 240 (height) x 256 (width) x 3 (RGB) image. This results in a total of 184,320 pixels (240 x 256 x 3).
- The RGB image is transformed into grayscale, reducing the number of channels from 3 to 1. After transformation, the grayscale frame has 61,440 pixels (240 x 256 x 1).
- The grayscale transformation is applied using the GrayScaleObservation wrapper.
- Four consecutive grayscale frames are stacked together to form a single observation, allowing the model to see both the past and present states of the game. The resulting stacked frames have a shape of 240 x 256 x 4.
- Stacking frames helps the model understand motion and dynamics, which are crucial for decision-making in fast-paced games like Super Mario Bros.

## Model Prediction

- The model takes the stacked frames as input and predicts:
  - An action to take
  - The estimated state value
  - The log probability of the chosen action
- Initially, due to random weights, the predictions are often meaningless

## Step Execution and Reward Collection

- The predicted action is executed in the environment using:
  ```
  obs, reward, done, info = env.step(action)
  ```
- The environment calculates the reward for each step.
- This process repeats for 512 steps.

## Reward Calculation

Reward Function is described in here (https://github.com/Kautenja/gym-super-mario-bros?tab=readme-ov-file#reward-function), code is available [\_get_reward()](https://github.com/Kautenja/gym-super-mario-bros/blob/master/gym_super_mario_bros/smb_env.py#L395)

- The reward function is evaluated 512 times, once per step
- The total accumulated reward forms a vector of length 512

## Data Collection

- After 512 steps, the model has collected:
  - 512 observations (stacked frames)
  - 512 actions taken
  - 512 rewards received
  - 512 value estimates
  - 512 log probabilities of the actions

## Weight Update

- Learning (weight adjustment) only happens after collecting the entire batch of 512 steps.
