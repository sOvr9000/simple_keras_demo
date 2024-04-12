
# This script provides a function to generate a dataset for the neural network that uses TimeDistributed effectively.

# Generated data consists of Sudoku puzzles and their solutions.

import numpy as np
from fischer.classicgames.sudoku.sudoku_io import read_puzzles
from fischer.classicgames.sudoku.su import number_swap, row_swap, column_swap, box_row_swap, box_column_swap, transpose



def format_puzzles(puzzles: np.ndarray, solutions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Initialize an array to store `samples` boards that are partitioned into nine 3x3 blocks containing integers 0-9.
    # 9 x 9 x 9:
    # - 9 of the 3x3 squares
    # - 9 cells in each 3x3 square.
    # - 10 possible values for each cell. (zero means empty)
    X = np.zeros((len(puzzles), 9, 9, 10))
    # Initialize an array to store `samples` solutions that are partitioned into nine 3x3 blocks containing integers 1-9.
    Y = np.zeros((len(puzzles), 9, 9, 9))

    for i, (puzzle, solution) in enumerate(zip(puzzles, solutions)):
        # Convert the puzzle to a 9 x 9 x 10 array, and the solution to a 9 x 9 x 9 array.
        for row in range(9):
            for col in range(9):
                # Index of the 3x3 square that contains the cell at (row, col)
                box = row // 3 * 3 + col // 3
                # Row and column indices relative to the 3x3 square
                rel_row = row % 3
                rel_col = col % 3
                # Cell index relative to the 3x3 square
                rel_cell = rel_row * 3 + rel_col

                # If the cell is empty, set the first element of the 10-element array to 1.  Otherwise, set the element corresponding to the value of the cell to 1.
                # The rest of the elements are zero to indicate that the cell is not that integer.
                # For example, if the cell is empty, the array will be [1, 0, 0, 0, 0, 0, 0, 0, 0, 0].
                # If the cell contains a 5, the array will be [0, 0, 0, 0, 0, 1, 0, 0, 0, 0].
                # (this is known as one-hot encoding)
                if puzzle[row, col] is None:
                    X[i, box, rel_cell, 0] = 1
                else:
                    X[i, box, rel_cell, puzzle[row, col]] = 1

                # Set the element corresponding to the value of the cell to 1 in the 9-element array.
                # There is no need to check if the cell is empty because the solution will not contain empty cells.
                # That said, the cell value is decremented by 1 to index the array.
                Y[i, box, rel_cell, solution[row, col] - 1] = 1
    
    return X, Y

def generate_data(samples: int) -> tuple[np.ndarray, np.ndarray]:
    # Read puzzles from a file.
    puzzles, solutions = read_puzzles('./timedist/puzzles_reduced.sudk')
    print(f'Loaded {len(puzzles)} puzzles.')

    # Augment dataset with Sudoku symmetries
    puzzles, solutions = augment_by_symmetries(puzzles, solutions,
        factor = max(1, int(.5 + samples / len(puzzles)))
    )

    # Format the puzzles and solutions into arrays that can be used by the neural network.
    return format_puzzles(puzzles, solutions)

def augment_by_symmetries(puzzles: np.ndarray, solutions: np.ndarray, factor: int = 8) -> tuple[np.ndarray, np.ndarray]:
    # Augment the dataset by applying random transformations to the puzzles and solutions.  At least 1.2 trillion distinct puzzles can be made from one puzzle.
    new_puzzles = np.empty((len(puzzles) * factor, 9, 9), dtype=np.int8)
    new_solutions = np.empty((len(puzzles) * factor, 9, 9), dtype=np.int8)
    for i, (puzzle, solution) in enumerate(zip(puzzles, solutions)):
        fi = i * factor
        for j in range(factor):
            fij = fi + j
            ps = np.concatenate([[puzzle], [solution]], axis=0)
            perm = np.arange(1, 10)
            np.random.shuffle(perm)
            ps = number_swap(ps, perm)
            ps = row_swap(ps, np.random.randint(0, 6, 3))
            ps = column_swap(ps, np.random.randint(0, 6, 3))
            ps = box_row_swap(ps, np.random.randint(0, 6))
            ps = box_column_swap(ps, np.random.randint(0, 6))
            if np.random.randint(0, 2):
                ps = transpose(ps)
            new_puzzles[fij], new_solutions[fij] = ps
    return new_puzzles, new_solutions


if __name__ == '__main__':
    # View some training data.
    # Input puzzles are one-hot encoded, whereas solutions are not (sparse encoding).
    puzzles, solutions = generate_data(10)
    indices = np.random.randint(0, puzzles.shape[0], 2)
    print(puzzles[indices])
    print(solutions[indices])

