###############################################################
#  SECTION 1: Minesweeper Environment (CSP + MDP)
# CSP = Constraint Satisfaction Problem, MDP = Markov Decision Process
# ERIC WORKING ON THIS ------------
# All FUNCTIONS HERE ARE MDP, EXCEPT DEF apply_csp_solver()
###############################################################
from constants import *
import numpy as np


class MinesweeperEnv: # This is just now for demo - ERIC setting up environment)
    """
    Environment that integrates:
      - CSP hook (apply_csp_solver) for deterministic safe squares
      - BFS expansions for zero-adjacency - Could be DFS too but need to discuss with team
      - Weighted sub-state (alpha-based) for the "modified" Q-learning approach mentioned in paper
    """

    def __init__(self, rows=NUM_ROWS, cols=NUM_COLS, num_mines=NUM_MINES, sub_state_size=3):
        self.rows = rows
        self.cols = cols
        self.num_mines = num_mines
        self.sub_state_size = sub_state_size

        # Board:   1=mine, 0=safe
        # Display: -1=hidden, else 0..8 for adjacency
        self.board = np.zeros((rows, cols), dtype=int)
        self.display = np.full((rows, cols), -1, dtype=int)

        # We do NOT place mines yet; we wait for the first move since this is always safe
        self.first_move_done = False  # ensures the first move is always safe

        self.done = False
        self.won = False
        self.revealed_count = 0
        self.safe_cells = (rows*cols) - num_mines

        # Compute alpha once (paper's eq. (10)) using minimal coefficients
        # We can tune this or regress these coefficients to maximize performance
        mine_ratio = self.num_mines / float(self.rows * self.cols)
        # For demo purposes using: alpha = 0.25 * p + 0.25 * q + 0.2 * mine_ratio + 0.05 [SEE EQ. 10]
        self.alpha = 0.25*self.rows + 0.25*self.cols + 0.2*mine_ratio + 0.05
        # We clamp alpha between [0,1] for safety:
        self.alpha = max(0, min(self.alpha, 1))

        # Run initial CSP logic to uncover guaranteed squares
        self.apply_csp_solver() # THIS IS WHAT ERIC IS WORKING ON!  JUST CALL AND PASS JUST NOW...

    def render(self):

        print("-"*50)
        for row in range(0,self.rows):
            print(" ", end = " | ")
            for col in range(0,self.cols):
                if self.display[row][col]==-1:
                    print(" ", end = " | ")
                else:
                    print(self.display[row][col],end=" | ")
            print("")
            print("-"*50)

    def reset(self):
        """Reset for a new episode."""
        self.board[:] = 0
        self.display[:] = -1
        self.first_move_done = False  # ensures the first move is always safe

        self.done = False
        self.won = False
        self.revealed_count = 0
        self.safe_cells = (self.rows*self.cols) - self.num_mines

        # Recompute alpha in case mine ratio changes, etc.
        mine_ratio = self.num_mines / float(self.rows * self.cols)
        self.alpha = 0.25*self.rows + 0.25*self.cols + 0.2*mine_ratio + 0.05 # SAME AS IN DEF_INIT (Can Tune this!)
        self.alpha = max(0, min(self.alpha, 1))

        self.apply_csp_solver()
        return self.display

    def _place_mines_excluding(self, row_exclude, col_exclude):
            """
            Place mines randomly on the board, excluding (row_exclude, col_exclude).
            This ensures the first move is always safe.
            """
            self.board[:] = 0
            placed = 0
            while placed < self.num_mines:
                r = np.random.randint(0, self.rows)
                c = np.random.randint(0, self.cols)
                # skip if it's the excluded cell or if a mine is already placed
                if (r == row_exclude and c == col_exclude):
                    continue
                if self.board[r, c] == 0:
                    self.board[r, c] = 1
                    placed += 1

    def apply_csp_solver(self):
        """
        A basic implementation of DSScsp + DSS based on the paper 
        (NOTE: This probably won't scale well for larger boards due to computational 
        cost of backtracking approach... Discuss with team!):
        1) Gather constraints from all uncovered cells
        2) Use DSS to prune obvious safe/mined cells
        3) Randomly backtrack (DSScsp) to find partial solutions
        4) Compute average mine probability for each covered cell

        Storing results in self.mine_probabilities for retrieval by get_probability_of_mine method.
        Usage:
          - Identify squares with 0% chance of being mine => uncover them
          - Identify squares with 100% chance => (flag them or skip in action space)
        """
        # Step 0: Prepare storage for probabilities.
        self.mine_probabilities = {}
        covered_cells = []   # list of (row, col)
        index_map = {}       # maps (row, col) -> index in the solution vector

        idx = 0
        for r in range(self.rows):
            for c in range(self.cols):
                if self.display[r, c] == -1:
                    covered_cells.append((r, c))
                    index_map[(r, c)] = idx
                    idx += 1

        if not covered_cells:
            return

        # Step 1: Gather constraints.
        constraints = []
        for rr in range(self.rows):
            for cc in range(self.cols):
                if self.display[rr, cc] >= 0:
                    adj_mines = self.display[rr, cc]
                    covered_neighbors = []
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            r2, c2 = rr + dr, cc + dc
                            if 0 <= r2 < self.rows and 0 <= c2 < self.cols:
                                if self.display[r2, c2] == -1:
                                    covered_neighbors.append(index_map[(r2, c2)])
                    if covered_neighbors:
                        constraints.append((covered_neighbors, adj_mines))

        # Step 2: Apply DSS to force obvious assignments.
        forced_mines = set()
        forced_safes = set()

        def dss_pass():
            changed = False
            for nbrs, val in constraints:
                # Only consider those not yet forced.
                covered_nbrs = [n for n in nbrs if n not in forced_mines and n not in forced_safes]
                flagged_count = sum(1 for n in nbrs if n in forced_mines)
                leftover = val - flagged_count
                if leftover == 0:
                    for x in covered_nbrs:
                        if x not in forced_safes:
                            forced_safes.add(x)
                            changed = True
                elif leftover == len(covered_nbrs):
                    for x in covered_nbrs:
                        if x not in forced_mines:
                            forced_mines.add(x)
                            changed = True
            return changed

        while dss_pass():
            pass

        # Step 3: DSScsp backtracking with memoization.
        solutions = []
        max_solutions = 20  # key parameter – adjust as needed
        current = [None] * len(covered_cells)
    
        # Pre-assign forced variables.
        for i in range(len(covered_cells)):
            if i in forced_mines:
                current[i] = 1
            elif i in forced_safes:
                current[i] = 0

        # --- New: Compute variable ordering by constraint frequency ---
        # Count frequency for each variable (i.e., how many constraints it appears in)
        freq = [0] * len(covered_cells)
        for nbrs, _ in constraints:
            for n in nbrs:
                freq[n] += 1
        # Create an ordering: variables with higher frequency come first.
        ordering = sorted(range(len(covered_cells)), key=lambda i: -freq[i])
        # Create a mapping: position -> variable index in the ordering.
        order_map = { pos: var for pos, var in enumerate(ordering) }

        def check_constraints():
            """Return True if the current partial assignment is consistent with constraints."""
            for nbrs, val in constraints:
                assigned = 0
                unknown = 0
                for n in nbrs:
                    if current[n] == 1:
                        assigned += 1
                    elif current[n] is None:
                        unknown += 1
                if assigned > val or assigned + unknown < val:
                    return False
            return True

        solutions_found = 0
        memo = {}  # memoization dictionary

        def backtrack(idx=0):
            nonlocal solutions_found
            # Use the ordering to select the next variable:
            if idx >= len(current):
                # Fully assigned: check constraints.
                if check_constraints():
                    solutions_found += 1
                    solutions.append(current[:])
                    memo[(idx, tuple(current))] = True
                else:
                    memo[(idx, tuple(current))] = False
                return

            key = (idx, tuple(current))
            if key in memo:
                return memo[key]
            if solutions_found >= max_solutions:
                memo[key] = None
                return

            # Instead of iterating over sequential indices, use ordering:
            next_var = ordering[idx]  # pick the variable at position idx in the ordered list
            found_solution = False
            for val in [0, 1]:
                current[next_var] = val
                if check_constraints():
                    backtrack(idx + 1)
                    if any(sol is not None for sol in solutions):
                        found_solution = True
                current[next_var] = None
            memo[key] = found_solution
            return

        backtrack()

        # Step 4: Compute probabilities.
        if not solutions:
            for i, (r, c) in enumerate(covered_cells):
                if i in forced_mines:
                    self.mine_probabilities[(r, c)] = 1.0
                elif i in forced_safes:
                    self.mine_probabilities[(r, c)] = 0.0
                else:
                    self.mine_probabilities[(r, c)] = 0.5
            return

        counts = [0] * len(covered_cells)
        for sol in solutions:
            for i, val in enumerate(sol):
                counts[i] += val

        for i, (r, c) in enumerate(covered_cells):
            if i in forced_mines:
                p = 1.0
            elif i in forced_safes:
                p = 0.0
            else:
                p = counts[i] / float(len(solutions))
            self.mine_probabilities[(r, c)] = p

    def check_adjacent_mines(self, row, col):
        """
        Return how many mines are adjacent to (row,col).
        """
        count = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                rr, cc = row+dr, col+dc
                if 0 <= rr < self.rows and 0 <= cc < self.cols:
                    count += self.board[rr, cc]
        return count

    def _flood_fill(self, row, col):
        """
        If adjacency=0, BFS expansion reveals neighbors.
        I am using BFS here but possibly use 
        DFS expansion here (discuss with Team!)
        """
        stack = [(row, col)]
        revealed_positions = []

        while stack:
            r, c = stack.pop()
            if self.display[r, c] == -1:
                adj = self.check_adjacent_mines(r, c)
                self.display[r, c] = adj
                revealed_positions.append((r, c))
                self.revealed_count += 1

                if adj == 0:  # blank => expand neighbors
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            rr, cc = r+dr, c+dc
                            if 0 <= rr < self.rows and 0 <= cc < self.cols:
                                if self.display[rr, cc] == -1:
                                    stack.append((rr, cc))
        return revealed_positions

    def step(self, action):
        """
        Perform an action (uncover cell).
        Return: next_state, reward, done, info
        """
        row, col = action

        # If this is the first move, place mines except this cell
        if not self.first_move_done:
            self._place_mines_excluding(row, col)
            self.first_move_done = True
        
        if self.display[row, col] != -1:
            # Cell already revealed => small negative penalty of 0.1
            return self.display.copy(), -0.1, self.done, {}

        # If it's a mine => lose
        if self.board[row, col] == 1:
            self.done = True
            self.won = False
            # Cell contains mine => large negative penalty of 1.0
            return self.display.copy(), -1.0, self.done, {"info": "Hit a mine"}

        # Otherwise reveal
        revealed_before = self.revealed_count
        self._flood_fill(row, col)
        revealed_now = self.revealed_count
        delta = revealed_now - revealed_before

        # Reward proportional to fraction of new safe squares uncovered
        reward = float(delta) / float(self.safe_cells)

        # Check for win
        if self.revealed_count == self.safe_cells:
            self.done = True
            self.won = True
            reward += 2.0  # big bonus for winning reward of 2.0

        # re-run CSP after new reveals and log probabilities to CSV for debugging csp_solver
        if not self.done:
            self.apply_csp_solver()

        # NOTE: This can cause a slow down since we are calculating probs after each step.
        # For debugging purposes, log the probabilities to a CSV file
        #    with open("csp_probabilities_log.csv", "a") as f:
        #        for (rc, prob) in self.mine_probabilities.items():
        #            r2, c2 = rc
        #            f.write(f"{r2},{c2},{prob}\n")        

        return self.display.copy(), reward, self.done, {"info": f"Revealed {delta} squares"}

    def get_available_actions(self):
        """Return all currently hidden cells."""
        actions = []
        for r in range(self.rows):
            for c in range(self.cols):
                if self.display[r, c] == -1:
                    actions.append((r, c))
        return actions

    # ----------- BEGIN "MODIFIED" SUB-STATE EXTRACTION (Alpha Weighted) ERIC WORKING ON THESE BUT USING DEMO VALUES JUST NOW-----------
    # the alpha here is a weighting of mine_probability and location score (review of equation 10) but using dummy alpha just now
    # We should tune or regress to get better values of alpha.

    def get_probability_of_mine(self, r, c): # ERIC WORKING ON THIS
        """
        Placeholder: This is what we would get from CSP
        eq. (3) from the paper. For now, I am just going to return 0.5 for demo.
        Returns the probability of cell (r, c) containing a mine.
        This is calculated from the CSP solver (DSScsp _ DSS).
        Note that probability falls back to 0.5 if no information
        """
        return self.mine_probabilities.get((r, c), 0.5) 

    def get_location_score(self, r, c):
        """
        Compute the location score for cell (r, c) using Equation (4).
        
        formula_dict = {'corner': 4, 'edge': 6, 'middle': 8}
        The formula for location score is formula_dict.key():
          - If (r,c) is a dict.key:
              score = 1 - (((m - f) / l) ** formula_dict.value())
        
        where:
          m = total number of mines (self.num_mines)
          f = number of flags used (0, because we don't flag)
          l = number of covered cells remaining = (total cells - revealed_count)
        
        Returns a value between 0 and 1.
        """
        m = self.num_mines
        f = 0  # No flags are used in our application.
        l = (self.rows * self.cols) - self.revealed_count
        # To avoid division by zero
        if l <= 0:
            return 1.0
    
        # Determine cell position:
        is_corner = ((r == 0 and c == 0) or 
                     (r == 0 and c == self.cols - 1) or 
                     (r == self.rows - 1 and c == 0) or 
                     (r == self.rows - 1 and c == self.cols - 1))
        
        if is_corner:
            exponent = 4
        elif r == 0 or c == 0 or r == self.rows - 1 or c == self.cols - 1:
            exponent = 6
        else:
            exponent = 8
    
        score = 1 - (((m - f) / l) ** exponent)
        # Return restricted score between 0 and 1
        return max(0, min(score, 1))
    
    def extract_sub_state(self, row, col):
        """
        Instead of directly returning display[row,col],
        we compute S_{c_{i,j}} = alpha * p_mine + (1-alpha) * location_score
        for hidden cells. If a cell is revealed, we store the adjacency number [0..8].
        If out of bounds => sentinel = -2.
        """
        half = self.sub_state_size // 2
        sub = []
        for rr in range(row - half, row + half + 1):
            row_vals = []
            for cc in range(col - half, col + half + 1):
                if 0 <= rr < self.rows and 0 <= cc < self.cols:
                    if self.display[rr, cc] == -1:
                        # Covered square => use alpha-weighted approach
                        p_mine = self.get_probability_of_mine(rr, cc)
                        loc_score = self.get_location_score(rr, cc)
                        val = (self.alpha * p_mine) + ((1 - self.alpha) * loc_score)
                        row_vals.append(val)
                    else:
                        # Revealed => just store the adjacency number
                        row_vals.append(float(self.display[rr, cc]))
                else:
                    row_vals.append(-2.0)  # sentinel for out-of-bounds
            sub.append(row_vals)
        return np.array(sub, dtype=np.float32)
