from ic_routing_board_generation.board_generator.numpy_data_model.abstract_board import AbstractBoard
from ic_routing_board_generation.board_generator.jax_utils.post_processor_utils_jax import extend_wires_jax
from dataclasses import dataclass
from ic_routing_board_generation.board_generator.numpy_data_model.abstract_board import AbstractBoard
from ic_routing_board_generation.board_generator.jax_utils.grid_utils import optimise_wire
from ic_routing_board_generation.board_generator.jax_utils.post_processor_utils_jax import extend_wires_jax, \
    training_board_from_solved_board_jax
from ic_routing_board_generation.board_generator.numpy_utils.post_processor_utils_numpy import count_detours
#import numpy as np

from jax import Array
from jax.random import PRNGKey
import jax.numpy as jnp
import jax

from jumanji.environments.routing.connector.constants import EMPTY, PATH, POSITION, TARGET
STARTING_POSITION = POSITION  # Resolve ambiguity of POSITION constant

#@jax.disable_jit()
#@jax.jit
class RandomSeedBoard(AbstractBoard):
    """ The boards are 2D arrays of wiring routed on a printed circuit board.

    The coding of the boards is as follows:
    Empty cells are coded 0.
    Positions (starting positions) are encoded starting from 2 in multiples of 3: 2, 5, 8, ...
    Targets are encoded starting from 3 in multiples of 3: 3, 6, 9, ...
    Wiring paths connecting the position/target pairs are encoded starting
            at 1 in multiples of 3: 1, 4, 7, ...

    Args:
        rows, cols (int, int) : Dimensions of the board.
        num_agents (int) : Number of wires to attempt to add to the board

    """
    def __init__(self, rows: int, cols: int, num_agents: int = 0):
        super().__init__(rows, cols, num_agents)

        self._rows = rows
        self._cols = cols
        self.grid_size = jax.lax.select(rows > cols, rows, cols)
        self._num_agents = num_agents  # Number of wires requested
        # Limit the number of wires to number_cells/3 to ensure we can fit them all
        max_num = jax.lax.select(num_agents > rows*cols//3, jnp.array(rows*cols//3, int), num_agents)
        self._wires_on_board = max_num  # Actual number of wires on the board

    #@jax.disable_jit()
    #@jax.jit
    def return_seeded_board(self, key: PRNGKey) -> Array:
        """ Generate and return an array of the board with the connecting wires encoded.

        Args:
            key (PRNGKey) : Random number generator key

        Returns:
            (Array) : 2D layout of the board with all wirings encoded
        """
        def generate_indices(key: jax.random.PRNGKey):
            """Create a list of agents with random initial starting and ending (as neighbours) locations."""
            row_indices, col_indices = jax.numpy.meshgrid(jax.numpy.arange(1, self._rows, 2), jax.numpy.arange(1, self._cols, 2), indexing='ij')
            index_choice = jax.numpy.stack((row_indices, col_indices), axis=-1).reshape(-1, 2)
            head_indices = jax.random.choice(key, index_choice, (self._num_agents,), replace=False)
            randomness_type = jax.random.randint(key, (), 0, 2)
            offset_array = jax.lax.select(randomness_type == 0, jax.numpy.array([[0, 1], [1, 0]]), jax.numpy.array([[1, 0], [0, 1]]))
            tail_offsets = jax.random.choice(key, offset_array, (self._num_agents,))
            tail_indices = head_indices + tail_offsets
            return head_indices - 1, tail_indices - 1

        xs, ys = generate_indices(key)
        board = jax.numpy.zeros((self.grid_size, self.grid_size), int)
        board = board.at[(xs[:, 0], xs[:, 1])].set(jnp.arange(len(xs))*3 + TARGET)
        board = board.at[(ys[:, 0], ys[:, 1])].set(jnp.arange(len(ys))*3 + POSITION)
        return board

    def return_solved_board(self, key: PRNGKey, randomness: float = 0.0, two_sided: bool = True,
                            extension_iterations: int = 1, extension_steps: int = 1e23) -> Array:
        """ Generate and return an array of the board with the connecting wires zeroed out.

        Args:
            key (PRNGKey) : Random number generator key
            randomness (float): How randomly to extend the wires, 0=>Keep same direction if possible, 1=>Random
            two_sided (bool): True => Wire extension extends both heads and targets.  False => Only targets.
            extension_iterations (int) : Number of iterations of wire-extension/BFS optimization. (Default = 1)
            extension_steps (int): Max number of extension loops to perform during each iteration.  (Default = no limit)
                                Note that sometimes wires might extend one cell or multiple cells per steps.

        Returns:
            (Array) : 2D layout of the board with all wirings encoded
        """
        from numpy import array as np_array
        from copy import deepcopy
        key, seedkey = jax.random.split(key)

        board_layout = self.return_seeded_board(seedkey)
        # print("SEEDED BOARD")
        # print(board_layout)

        # While Loop: Loop through specified num of extension/optimisation iterations
        # as long as convergence hasn't occurred
        def ext_iterations_cond(carry):
            board_layout, key, iteration_num, converged = carry
            return ~converged & (iteration_num < extension_iterations)

        def ext_iteration_body(carry):
            board_layout, key, iteration_num, converged = carry
            iteration_num = iteration_num + 1

            # print("Extension iteration ", iteration_num + 1)
            key, extkey, optkey = jax.random.split(key, 3)
            board_layout = extend_wires_jax(board_layout, extkey, randomness, two_sided, extension_steps)
            # print(board_layout)
            optkeys = jax.random.split(optkey, self._wires_on_board)
            #board_layout_save = deepcopy(board_layout)

            """ TEMPORARILY DISABLE OPTIMISATION
            ## Optimise each wire individually
            #def optimise_wire_loop_func(wire_num, carry):
            #    board_layout, keys = carry
            #    board_layout = optimise_wire(keys[wire_num], board_layout, wire_num)
            #    carry = (board_layout, keys)
            #    return carry
            #carry = (board_layout, optkeys)
            #board_layout, _ = jax.lax.fori_loop(0, self._wires_on_board, optimise_wire_loop_func, carry)
            ## print("Optimization")
            ## print(board_layout)
            """

            #   WHILE LOOP
            carry = (board_layout, key, iteration_num, converged)
            return carry
        
        # For ext_iterations while loop
        converged = False
        iteration_num = 0
        carry = (board_layout, key, iteration_num, converged)
        board_layout, key, iteration_num, converged = jax.lax.while_loop(ext_iterations_cond, ext_iteration_body, carry)
        # END WHILE LOOP
    
        return board_layout



    #@jax.jit
    def return_training_board(self, key: PRNGKey, randomness: float = 0.0, two_sided: bool = True,
                              extension_iterations: int = 1, extension_steps: int = 1e23) -> Array:
        """ Generate and return an array of the board with the connecting wires zeroed out.

        Args:
            key (PRNGKey) : Random number generator key
            randomness (float): How randomly to extend the wires, 0=>Keep same direction if possible, 1=>Random
            two_sided (bool): True => Wire extension extends both heads and targets.  False => Only targets.
            extension_iterations (int) : Number of iterations of wire-extension/BFS optimization.
            extension_steps (int): Max number of extension loops to perform during each iteration.
                                Note that sometimes wires may extend multiple cells per extension loop.
        Returns:
            (Array) : 2D layout of the board with all wirings encoded
        """
        board_layout = self.return_solved_board(key, randomness, two_sided, extension_iterations, extension_steps)
        training_board = training_board_from_solved_board_jax(board_layout)
        return training_board

    #@jax.jit
    def generate_starts_ends(self, key: PRNGKey, randomness: float = 0.0, two_sided: bool = True,
                             extension_iterations: int = 1, extension_steps: int = 1e23) -> Array:
        """  Call generate, take the first and last cells of each wire

        Args:
            key (PRNGKey) : Random number generator key
            randomness (float): How randomly to extend the wires, 0=>Keep same direction if possible, 1=>Random
            two_sided (bool): True => Wire extension extends both heads and targets.  False => Only targets.
            extension_iterations (int) : Number of iterations of wire-extension/BFS optimization.
            extension_steps (int): Max number of extension loops to perform during each iteration.
                                Note that sometimes wires may extend multiple cells per extension loop.
        Returns:
            (Array) : List of starting points, dimension 2 x num_agents
            (Array) : List of target points, dimension 2 x num_agents
        """
        board_layout = self.return_solved_board(key, randomness, two_sided, extension_iterations, extension_steps)

        def find_positions(wire_id):
            wire_positions = board_layout == 3 * wire_id + POSITION
            wire_targets = board_layout   == 3 * wire_id + TARGET

            # Compute indices where wire_positions and wire_targets are True
            start_indices = jnp.argwhere(wire_positions, size=2)
            end_indices   = jnp.argwhere(wire_targets,   size=2)

            # Take the first valid index (row)
            start = start_indices[0]
            end = end_indices[0]
            return start, end

        wire_ids = jnp.arange(self._num_agents)
        starts_ends = jax.vmap(find_positions)(wire_ids)
        #jax.debug.print("{x}", x=starts_ends)
        #jax.debug.print("{x}", x=type(starts_ends))
        starts, ends = starts_ends[0], starts_ends[1]

        # Want starts to be a tuple of arrays, first array is x coords, second is y coords
        starts = (starts[:, 0], starts[:, 1])
        ends = (ends[:, 0], ends[:, 1])
        return starts, ends


if __name__ == '__main__':
    board_object = RandomSeedBoard(3, 3, 3)
