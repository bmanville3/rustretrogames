# Retro Rust Games + Custom RL Framework

This project contains a collection of retro-style games written entirely in Rust, accompanied by custom deep learning and reinforcement learning libraries. It exists both as a playground for experimenting with RL algorithms and as a long-term Rust learning project.

## Running the Project

For best performance (especially for larger neural network models), run:

`cargo run --release`

## Project Goals
### Primary Goals

- Strengthen and showcase my skills in Rust.

- Gain a deeper understanding of reinforcement learning and how modern game-playing agents are built.

### Additional Learning Objectives

- Develop a more robust understanding of deep learning by building components from scratch.
  - This has greatly increased my appreciation for frameworks like PyTorch.

- Learn efficient multithreading patterns in Rust.

- Explore GUI development using Rust libraries such as Iced.

## Implemented Games
### Snake
#### Implemented Bot Types

- Random movement bots

- BFS-based bots that search for specific goals

- Double Deep Q-Learning (DDQN) agent trained using the custom RL library

## Roadmap / TODO

- Add convolutional layers to the neural network module.

- Explore using an autoencoder to reduce the size of the game state.

- Continue training and evaluating RL agents.
  - The DDQN snake bot is quite bad right now.

- Improve end-of-game messaging and UI polish.

- Fully conform to Rust formatting, clippy lints, and general best practices.
  - Project is in a "wip" state right now and is majorly behind on this.

- Eventually add more games like PacMan
