# Snake-Game-AI

# AI-Driven Snake Game

## Overview
This project is an implementation of the classic Snake game enhanced with artificial intelligence. Using Python and reinforcement learning techniques, the game features an AI that learns to navigate and maximize points by avoiding walls and its own tail while eating food.

## Features
- **AI Gameplay**: Utilizes reinforcement learning for decision making.
- **Graphical Interface**: Built with Pygame, providing a visual representation of the game, including the snake's movements and interactions.
- **Customizable Settings**: Adjustable parameters for training the AI, including the number of training iterations and learning rates.

## Requirements
Ensure you have Python and the necessary libraries installed:
- `pygame`
- `numpy`

Install the required libraries using:
```
bash
pip install pygame numpy
```
## Installation
Clone the repository to your local machine:

```
bash
Copy code
git clone https://github.com/yourusername/ai-snake-game.git
```
Navigate to the project directory:

```
bash
Copy code
cd ai-snake-game
```
## Usage
Run the game by executing:
```
bash
Copy code
python game.py
```
You can adjust the training settings and game parameters by modifying the helper.py configuration or passing command-line arguments to game.py.

## Command Line Arguments
Adjust AI settings like training iterations, display iterations, learning parameters, etc., through command-line arguments as documented in helper.py.

## Game Mechanics
- `Snake Movement: Control the snake using AI, where the snake must eat food to grow and gain points`
- `Game End Conditions: The game ends if the snake hits the wall or itself.`
- `Scoring: Points are awarded for each piece of food eaten by the snake.`

## Contributing
Contributions to improve the game or enhance the AI algorithm are welcome. Please fork the repository and open a pull request with your proposed changes.

License
This project is open-source and available under the MIT License. See the LICENSE file in the repository for more information.
