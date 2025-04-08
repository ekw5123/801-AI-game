from constants import *
from MineSweeperEnv import MinesweeperEnv
import airand as aiRand

def SimOrInteractive(choice=None):
    if choice is None:
        """Choose between simulation or interactive mode."""
        print("Choose mode:")
        print("1. Simulation")
        print("2. Interactive")
        choice = input("Enter choice: ")

    if str(choice) == "1":
        return True
    elif str(choice) == "2":
        return False
    else:
        print("Invalid choice. Please enter 1 or 2.")
        return SimOrInteractive()
    
def randomOrChoice(choice=None):
    if choice is None:
        """Choose between user select or random mode."""
        print("Choose mode:")
        print("1. user select")
        print("2. Random")
        choice = input("Enter choice: ")

    if str(choice) == "1":
        return True
    elif str(choice) == "2":
        return False
    else:
        print("Invalid choice. Please enter 1 or 2.")
        return SimOrInteractive()
    

def runGame(agent=False):
    if not agent:
        game = MinesweeperEnv(rows=NUM_ROWS, cols=NUM_COLS, num_mines=NUM_MINES)
        game.reset()
        while not game.done:
            game.render()
            action = input("Enter action (row, col): ")
            row, col = map(int, action.split(","))
            game.get_available_actions()
            next_state, reward, done, info = game.step((row, col))
        if game.won:
            print("You won!")
        else:
            print("You lost!")
            game.render()
    else:
        game = MinesweeperEnv(rows=NUM_ROWS, cols=NUM_COLS, num_mines=NUM_MINES)
        game.reset()
        while not game.done:
            game.render()
            row, col = aiRand.selectRowCol(game.display)
            game.get_available_actions()
            next_state, reward, done, info = game.step((row, col))
        if game.won:
            print("You won!")
        else:
            print("You lost!")
            game.render()



###############################################################
#  MAIN (DEMO)
###############################################################
if __name__ == "__main__": # hyperparameters need to be tuned in final version
    decision=SimOrInteractive()
    if not decision:
        input
    
    if decision:
        from deepQLearner import DeepQLearner
        dql = DeepQLearner(
            rows=NUM_ROWS,
            cols=NUM_COLS,
            num_mines=NUM_MINES,
            sub_state_size=3,
            gamma=0.99,
            lr=1e-3,
            epsilon_start=1.0,
            epsilon_min=0.1,
            epsilon_decay=1e-4,
            buffer_capacity=3000,
            batch_size=32
        )

        dql.train_episodes(csv_output="metrics_output.csv")
        # DEMO TESTING ON 1 ITERATION with 200 EPISODES (WinRatio may change when repeat testing this demo except for 0% mine density)
        # NUM_MINES/NUM_ROWS/NUM_COLS = 10/10/10, I got 0 wins out of 200
        # NUM_MINES/NUM_ROWS/NUM_COLS = 5/10/10, I got 10 wins out of 200
        # NUM_MINES/NUM_ROWS/NUM_COLS = 2/10/10, I got 112 wins out of 200
        # NUM_MINES/NUM_ROWS/NUM_COLS = 1/10/10, I got 157 wins out of 200
        # NUM_MINES/NUM_ROWS/NUM_COLS = 0/10/10, I got 200 wins out of 200

        # NUM_MINES/NUM_ROWS/NUM_COLS = 8/8/8, I got 0 wins out of 200    
        # NUM_MINES/NUM_ROWS/NUM_COLS = 5/8/8, I got 13 wins out of 200
        # NUM_MINES/NUM_ROWS/NUM_COLS = 2/8/8, I got 90 wins out of 200
        # NUM_MINES/NUM_ROWS/NUM_COLS = 1/8/8, I got 146 wins out of 200
        # NUM_MINES/NUM_ROWS/NUM_COLS = 0/8/8, I got 200 wins out of 200

        # NUM_MINES/NUM_ROWS/NUM_COLS = 12/12/12, I got 0 wins out of 200    
        # NUM_MINES/NUM_ROWS/NUM_COLS = 5/12/12, I got 19 wins out of 200
        # NUM_MINES/NUM_ROWS/NUM_COLS = 2/12/12, I got 126 wins out of 200
        # NUM_MINES/NUM_ROWS/NUM_COLS = 1/12/12, I got 169 wins out of 200
        # NUM_MINES/NUM_ROWS/NUM_COLS = 0/12/12, I got 200 wins out of 200

        # AlSO GETTING 0 wins out of 200 if NUM_MINES > NUM_ROWS|NUM_COLS
        
        # Not great learning here in the epsilon value, but I believe it is because the main issue is that is no 
        # CSP solver implemented yet, so the dummy probs (e.g. 0.5 for covered cells) is still implying a random selection.  Also squares revealed looks good, 
        # sometimes but it is most likely because of the def _flood fill_ (BFS) in SECTION 1 revealing lots of good squares due to low mine density. 
        # Once CSP is implemented, then we should see better learning gains.  If still bad after that, then we can adjust DQN architecture, episodes, 
        # penalties, alpha, hyperparameters etc.

        dql.calculate_stratified_metrics(metrics_csv="metrics_output.csv", 
                                    #strat_csv="metrics_stratified.csv",
                                    strat_interval=100)
        # NOTE: CAN CALL THIS METHOD WITH A SPECIFIED ARGUMENT OF strat_csv = 'Name_of_file.csv', OTHERWISE WILL ADD BOARD PARAMS TO FILE NAME
        # NOW THAT DEMO TESTING WITH CSP SOLVER IS IN PLACE AND PROBS CALCULATED, I SEE A SIGNIFICANT IMPROVEMENT IN WIN RATIO/100 EPISODES
        # STILL NEEDS WORK ON CSP SOLVER.

    else:
            
            if randomOrChoice():
                runGame(False)
            else:
                runGame(True)
            