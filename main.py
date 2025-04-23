from constants import NUM_MINES, NUM_ROWS, NUM_COLS, NUM_EPISODES
from MineSweeperEnv import MinesweeperEnv
import airand as aiRand
import shutil

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
if __name__ == "__main__":
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
            epsilon_min=0.05,
            epsilon_decay=1e-4,
            buffer_capacity=3000,
            batch_size=32
        )

        dql.train_episodes(csv_output="metrics_output_current.csv")

        # Construct new file name with board parameters and number of episodes.
        new_metrics_filename = f"metrics_output_{NUM_ROWS}x{NUM_COLS}_{NUM_MINES}mines_{NUM_EPISODES}eps.csv"
    
        # Copy the dynamically generated metrics_output.csv to the new file name.
        shutil.copy("metrics_output_current.csv", new_metrics_filename)
    
        print(f"Copied metrics_output_current.csv to {new_metrics_filename}")

        dql.calculate_stratified_metrics(metrics_csv="metrics_output_current.csv", 
                                    strat_interval=100)

    else:
            
            if randomOrChoice():
                runGame(False)
            else:
                runGame(True)
            