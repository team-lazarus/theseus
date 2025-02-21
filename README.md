# Thesus

The one who traverses the Labrynth, this is a reinforcement learning agent & pipeline to play the game [Labrynth](https://github.com/team-lazarus/labrynth)

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Best Practices](#best-practices)
- [Contributing](#contributing)
- [License](#license)

## Installation

Follow these steps to set up the project:

1. **Set up a virtual environment** (recommended for dependency isolation):
   ```sh
   python3 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   venv\Scripts\activate     # On Windows
   ```

2. **Create a soft link for the root directory in the virtual environment's `lib` folder:**
   ```sh
   ln -s /absolute/path/to/the/root/of/thesus /absolute/path/to/python/environment/lib/
   ```

3. **Install dependencies from `requirements.txt`:**
   ```sh
   pip install -r requirements.txt
   ```

## Usage

Run the program using:
```sh
python3 main.py
```

## Project Structure

```
.
├── main.py           # Entry point of the application
├── agent.py          # Reinforcement learning (RL) agent implementation
├── models/           # Contains all machine learning models
├── utils/            # Utility functions for logging, tracking metrics, graphing, etc.
│   ├── network/      # Networking utilities (TCPClient, environment abstraction, etc.)
├── tests/            # Unit tests for all modules
│   ├── utils/        # Tests for utilities
│   ├── models/       # Tests for models
│   ├── test_agent.py # Tests for the RL agent
```

## Best Practices

To ensure code quality and maintainability, follow these guidelines:

- **Commit messages should follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)**.
- **Work on a separate branch** and create pull requests (PRs) against the `master` branch.
- **Write unit tests** for any core functionality you add (`tests/` folder).
- **Format your code using `black` before committing**:
  ```sh
  black .
  ```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request following the best practices.

