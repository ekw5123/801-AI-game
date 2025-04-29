# 801-AI-game
Penn State, Spring 2025
Eric Williams, David Kleppang, Maegan Soebel

Designing a minesweeper game and an AI that plays it

## Description

This project uses a combindation of Constraint SAtisfaction Problems, the Markov Decision Process and Deep Learning to solve a game of minesweeper.  Minesweeper was chosen as the game is played with incomplete information but can be won by employing learned strategies.  

Each revealed number represents a constraint on a neighboring cell. This enabled us to utilize the Constraint Satisfaction Problem (CSP) to help train our AI how to solve for the winning strategy.​

The Markov Decision Process (MDP) is used when the AI has to make a decision when given uncertain parameters. Given that Minesweeper is played with incomplete information, this is a great method to implement to help train our AI to make the best decision.​

With our best run, we achieved a win ratio of 42.4% over the course of 10,000 training episodes.

All of our data showed an initial performance increase, but an ultimate plateau​.

The plateau in performance shortly after epsilon decay could point to diminished returns after the training period, settling on a sub-optimal policy, and reduced ability to test after the training period and continue to develop winning strategies.​

We could potentially resolve this by extending the training period and slowing the decay of our epsilon value, allow for milestone increases in the epsilon value which, in turn, allows for more risk during testing, or continuing to tune the parameters like our penalty/reward values as we saw some opportunities to do so in the cells revealed/reward totals visuals.

## Getting Started

### Dependencies
Required python modules
* tensorflow
* numpy
* shutil
* pandas
* matplotlib
* mplcursors
* collections

### Installing

* Clone the repo
* Install above prerequisites

### Executing program

* How to run the program
* Run main.py
* For ai mode, type 1
* For interactive mode: type 2
* ---interactive mode user input: type 1
* ---interactive mode random agent: type 2

To Postprocess generated data into plots, run metrics_visInteractive.py 


## Help

Constants for number of mines, grid dimensions, and number of episodes can be modified in constants.py

## Acknowledgments

Inspiration, code snippets, etc.
* Sinha, Y. P., Malviya, P., & Nayak, R. K. (2021). Fast constraint satisfaction problem and learning-based algorithm for solving Minesweeper. arXiv. https://doi.org/10.48550/arXiv.2105.04120 
