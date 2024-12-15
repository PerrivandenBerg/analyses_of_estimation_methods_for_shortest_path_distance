# README

## Overview
This project recreates and modifies the estimation algorithm from the paper: *"Fast Shortest Path Distance Estimation in Large Networks"*. It is part of the course **Social Network Analyses for Computer Scientists** at Leiden University.

The primary goal is to evaluate the four suggested estimation methods in the context of landmark selection strategies to determine which method performs best. The project focuses on the following landmark selection methods:
- **Random**
- **Degree**
- **Closeness Centrality**
- **Deg-1** (Degree-based with a minimum distance of 1 node between landmarks)
- **CC-1** (Closeness Centrality-based with a minimum distance of 1 node between landmarks)

## Datasets
The following datasets were used for the experiments:
- **Amazon**: [https://snap.stanford.edu/data/com-Amazon.html](https://snap.stanford.edu/data/com-Amazon.html)
- **Enron**: [http://snap.stanford.edu/data/email-Enron.html](http://snap.stanford.edu/data/email-Enron.html)
- **Facebook**: [https://snap.stanford.edu/data/ego-Facebook.html](https://snap.stanford.edu/data/ego-Facebook.html)
- **Net-PA**: [https://snap.stanford.edu/data/roadNet-PA.html](https://snap.stanford.edu/data/roadNet-PA.html)
- **Twitch**: [https://snap.stanford.edu/data/twitch_gamers.html](https://snap.stanford.edu/data/twitch_gamers.html)

Ensure the dataset files are downloaded and placed in the appropriate directory before running the experiments.

## Prerequisites
To use this application, you need:
1. **Python**: Version 3.6.8
2. The following Python libraries (install using `pip`):
   - `networkx`
   - `matplotlib`
   - `numpy`
   - `seaborn`
   - `pandas`

## Installation and Setup
1. Download the dataset files from the links provided above and place them in the working directory.
2. Create a virtual environment:
   ```bash
   python3 -m venv env
   source env/bin/activate  # On Windows, use `env\Scripts\activate`
   ```
3. Install the required libraries:
   ```bash
   pip install networkx matplotlib numpy seaborn pandas
   ```
4. Ensure the datasets are properly named and located in the correct directory.

## Usage
1. Run the program using the correct name:
   ```bash
   python snacs_algo_<name>.py
   ```
2. The results of the experiments will be saved in the working directory as `.eps` files.
3. After reading the dataset, network statistics will be printed on the screen.
4. Debugging statements will provide updates on the program's progress during execution.

### Running on a Server
If you are running this program on a server and need to disconnect from your session, use the `screen` command to prevent the program from stopping:
1. Start a new screen session:
   ```bash
   screen
   ```
   This opens a new terminal session within your current terminal.
2. Run the program within the screen session:
   ```bash
   python snacs_algo_<name>.py
   ```
3. Detach from the screen session using the key combination `Ctrl+A`, followed by `D`.
4. Reattach to the screen session later using:
   ```bash
   screen -r
   ```
   **Note**: Forgetting the `-r` flag will create a new screen session instead of reattaching to the existing one.

## Results
The output of the experiments is saved as `.eps` files, which can be analyzed using your preferred visualization tools. Ensure that you review the debugging statements for any issues during execution.

## Contributing
Contributions are welcome! If you would like to contribute to this project, please fork the repository, create a feature branch, and submit a pull request.

### Possible Future Work
   - Look into different landmark selection strategies.
   - We stored all the landmark calculations in RAM, but consider reducing storage using BFS for nodes which are close to a landmark. This way you could reduce RAM usage.

## License
This project is licensed under the MIT License.
