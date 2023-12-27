import numpy as np
from queue import PriorityQueue

# Define the goal state
goal_state = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])

# Define a function to get the position of a tile in a state
def get_position(tile, state):
    return np.where(state == tile)

# Define the Manhattan distance heuristic function
def manhattan_distance(state):
    distance = 0
    for i in range(state.shape[0]):
        for j in range(state.shape[1]):
            tile = state[i][j]
            if tile != 0:
                goal_position = get_position(tile, goal_state)
                distance += abs(i - goal_position[0][0]) + abs(j - goal_position[1][0])
    return distance

def euclidean_distance(state):
    distance = 0
    for i in range(state.shape[0]):
        for j in range(state.shape[1]):
            tile = state[i][j]
            if tile != 0:
                goal_position = get_position(tile, goal_state)
                distance += np.sqrt((i - goal_position[0][0])*2 + (j - goal_position[1][0])*2)
    return distance

def misplaced_tiles(state):
    misplaced = 0
    for i in range(state.shape[0]):
        for j in range(state.shape[1]):
            if state[i][j] != goal_state[i][j]:
                misplaced += 1
    return misplaced

# Define the Node class to represent each state in the search tree
class Node:
    def _init_(self, state, parent=None, h = 'm'):
        self.state = state
        self.parent = parent
        self.g = 0 if not parent else parent.g + 1
        if h == 'm':
            self.h = manhattan_distance(self.state)
        elif h == 'e':
            self.h = euclidean_distance(self.state) 
        else:
            self.h = misplaced_tiles(self.state) 
    
    def _lt_(self, other):
        return self.g + self.h < other.g + other.h
    
    def get_path(self):
        path = []
        node = self
        while node:
            path.append(node)
            node = node.parent
        path.reverse()
        return path

# Define the main search function
def solve_puzzle(initial_state):
    # Initialize the open and closed sets
    open_set = PriorityQueue()
    closed_set = set()
    
    # Add the initial state to the open set
    initial_node = Node(initial_state)
    open_set.put(initial_node)
    
    # Start the search
    while not open_set.empty():
        # Get the node with the lowest f score from the open set
        current_node = open_set.get()
        
        # Check if the current node is the goal state
        if np.array_equal(current_node.state, goal_state):
            return current_node.get_path()
        
        # Add the current node to the closed set
        closed_set.add(tuple(current_node.state.flatten()))
        
        # Generate all possible successor states by moving one tile in each of the four directions
        zero_pos = np.argwhere(current_node.state == 0)[0]
        for move in [((0, 1)), ((0, -1)), ((1, 0)), ((-1, 0))]:
            new_pos = zero_pos + np.array(move)
            if np.all(new_pos >= 0) and np.all(new_pos < 3):
                new_state = current_node.state.copy()
                new_state[zero_pos[0], zero_pos[1]] = current_node.state[new_pos[0], new_pos[1]]
                new_state[new_pos[0], new_pos[1]] = 0
                
                # Check if the new state has already been visited
                if tuple(new_state.flatten()) not in closed_set:
                    # Add the new state to the open set with its f score
                    new_node = Node(new_state, parent=current_node)
                    open_set.put(new_node)
    
    # If the open set is empty and no solution was found, return None
    return None

# Test the function with an example initial state
initial_state = np.random.permutation(np.arange(9)).reshape((3,3))
solution = solve_puzzle(initial_state)

if solution:
    print("Solution found!")
    for i, node in enumerate(solution):
        print(f"Step {i}\n{node.state}\n")
else:
    print("No solution found.")