from dataclasses import dataclass
from functools import partial
import timeit
import time
from chex import PRNGKey, Array

# import random
from typing import Tuple
import jax.numpy as jnp
import jax
from jumanji.environments.routing.connector.constants import POSITION, TARGET, \
    PATH

from ic_routing_board_generation.board_generator.jax_data_model.wire import Wire, create_wire, \
    stack_push

STARTING_POSITION = POSITION  # My internal variable to disambiguate the word "position"
# Also available to import from constants NOOP, LEFT, LEFT, UP, RIGHT, DOWN

class JaxRandomWalk:
    def __init__(self, rows: int, cols: int, num_agents: int = 3):
        # super().__init__(rows, cols, num_agents)
        self._rows = rows
        self._cols = cols
        self._num_agents = num_agents

    def return_blank_board(self) -> Array:
        return jnp.zeros((self._rows, self._cols), dtype=int)

    def pick_start(self, key: PRNGKey, layout: Array, wire_id: int, max_length: int) -> Wire:
        """Create a wire and populate start and end points randomly."""
        key, subkey = jax.random.split(key)
        grid_size = len(layout) * len(layout[0])
        # See if there is any empty space in the grid
        empty_space = jnp.any(layout == 0)
        def inner_create_wire(key, layout=layout, wire_id=wire_id):
            # Pick a random coordinate which is empty (i.e. 0)
            coordinate_flat = jax.random.choice(
                key=key,
                a=jnp.arange(grid_size),
                shape=(),
                replace=False,
                p=jnp.where(layout.flatten() == 0, 1, 0),
            )

            # Create a 2D coordinate from the flat array.
            coordinate = jnp.divmod(coordinate_flat, self._rows)
            new_wire = create_wire(grid_size, coordinate, (-1, -1), wire_id)
            new_wire = stack_push(new_wire, coordinate)
            layout = layout.at[coordinate[0], coordinate[1]].set(3 * wire_id + POSITION)

            return new_wire, layout
        # Use jax.lax.cond to only run the inner function if there is empty space
        dummy_wire = create_wire(grid_size, (-1, -1), (-1, -1), wire_id)
        new_wire, layout = jax.lax.cond(empty_space, inner_create_wire, lambda x: (dummy_wire, layout), key)
        return (key, layout, new_wire, wire_id, max_length), empty_space

    def adjacent_cells(self, cell: int):
        available_moves = jnp.full(4, cell)
        direction_operations = jnp.array([-1 * self._rows, self._rows, -1, 1])

        cells_to_check = available_moves + direction_operations
        #jax.debug.print("cells to check are: {c}", c=cells_to_check)
        is_id_in_grid = cells_to_check < self._rows * self._cols
        is_id_positive = 0 <= cells_to_check
        mask = is_id_positive & is_id_in_grid
        #jax.debug.print("mask is: {mask}", mask=mask)

        # Need to add further checks to make sure that the move doesn't go off the grid
        # Can maybe use the divmod function to check one of the row or column is the same
        # as the current cell
        unflatten_available = jnp.divmod(cells_to_check, self._rows)
        unflatten_current = jnp.divmod(cell, self._rows)
        #print("unflatten is: ", unflatten_available[0])
        is_same_row = unflatten_available[0] == unflatten_current[0]
        is_same_col = unflatten_available[1] == unflatten_current[1]
        row_col_mask = is_same_row | is_same_col
        mask = mask & row_col_mask
        #jax.debug.print("what I want: {x}", x=jnp.where(mask == 0, -1, cells_to_check))
        return jnp.where(mask == 0, -1, cells_to_check)

    def available_cells(self, layout: Array, cell: int):
        # TODO: make sure this outputting -1 for the end of the array
        adjacent_cells = self.adjacent_cells(cell)
        #jax.debug.print("adjacent cells are: {adj}", adj=adjacent_cells)
        #print("internal adjacent cells are: ", adjacent_cells)
        _, available_cells_mask = jax.lax.scan(self.is_cell_free, layout, adjacent_cells)
        # Also want to check if the cell is touching itself more than once
        _, touching_cells_mask = jax.lax.scan(self.is_cell_touching_self, layout, adjacent_cells)
        # Combine masks
        available_cells_mask = available_cells_mask & touching_cells_mask
        # Want the boolean masking to leave available cells as they are and
        # change the unavailable cells to -1
        available_cells = jnp.where(available_cells_mask == 0, -1, adjacent_cells)
        #print("available cells are: ", available_cells)
        return jnp.hstack((available_cells,
                           jnp.full(self._rows - len(available_cells) + 1,
                                    -1)))

    def is_cell_free(self, layout: Array, cell: int):
        coordinate = jnp.divmod(cell, self._rows)
        return layout, jax.lax.select(cell == -1, False, layout[coordinate[0], coordinate[1]] == 0)
    
    def is_cell_touching_self(self, layout: Array, cell: int):
        """
        Check if the cell is touching any of the wire's own cells more than once.
        This means looking for surrounding cells of value 3 * wire_id + POSITION or
        3 * wire_id + PATH.
        """
        coordinate = jnp.divmod(cell, self._rows)
        # get the wire id from the layout
        # Taking one away so head tail and path are all 0, 1, 2
        wire_id = (layout[coordinate[0], coordinate[1]] -1 ) // 3
        return layout, jax.lax.select(cell == -1, False, (layout[coordinate[0], coordinate[1]] == 3 * wire_id + POSITION) | (layout[coordinate[0], coordinate[1]] == 3 * wire_id + PATH))

    def one_step(self, random_walk_tuple: Tuple[PRNGKey, Array, Wire]):
        # jax.debug.print("we do be stepping")
        key, layout, wire, wire_id = random_walk_tuple
        key, subkey = jax.random.split(key)
        cell = wire.path[wire.insertion_index - 1][0] * self._cols + wire.path[wire.insertion_index - 1][1]
        #jax.debug.print("cell is: {cell}", cell=cell)
        available_cells = self.available_cells(layout=layout, cell=cell)
        #jax.debug.print("available cells are: {available_cells}", available_cells=available_cells)
        step_coordinate_flat = jax.random.choice(
            key=subkey,
            a=available_cells,
            shape=(),
            replace=False,
            p=available_cells != -1,
        )
        coordinate = jnp.divmod(step_coordinate_flat, self._rows)
        new_wire = stack_push(wire, coordinate)
        # Add new coordinate
        layout = layout.at[coordinate[0], coordinate[1]].set(3 * wire_id + TARGET) # TODO: change to wire constant
        # Change old coordinate to be part of wire, TODO: if not a head
        layout = layout.at[wire.path[wire.insertion_index - 1][0], wire.path[wire.insertion_index - 1][1]].set(3 * wire_id + PATH)
        return key, layout, new_wire, wire_id

    def can_step(self, random_walk_tuple: Tuple[PRNGKey, Array, Wire]):
        key, layout, wire, _ = random_walk_tuple
        cell = wire.path[wire.insertion_index - 1][0] * self._cols + wire.path[wire.insertion_index - 1][1]
        available_cells = self.available_cells(layout=layout, cell=cell)
        return jnp.any(available_cells != -1)

    def walk_randomly(self, random_walk_tuple: Tuple[PRNGKey, Array, Wire, int, int]):
        max_length = random_walk_tuple[4]
        random_walk_tuple = random_walk_tuple[:4]
        moved = False

        def body_fun(i, carry):
            random_walk_tuple, moved = carry
            can_step = self.can_step(random_walk_tuple)
            moved = moved | can_step
            random_walk_tuple = jax.lax.cond(can_step, self.one_step, lambda x: x, random_walk_tuple)
            return (random_walk_tuple, moved)

        random_walk_tuple, moved = jax.lax.fori_loop(0, max_length, body_fun, (random_walk_tuple, moved))

        # Afterwards, change the first cell to be a head
        key, layout, wire, wire_id = random_walk_tuple
        layout = layout.at[wire.path[0][0], wire.path[0][1]].set(3 * wire_id + POSITION)
        random_walk_tuple = (key, layout, wire, wire_id, max_length)
        return random_walk_tuple, moved

    def add_agents(self, key: PRNGKey, board: Array, max_length: int):
        # Random walk n times
        random_walk_tuple = (key, board, 0, 0, max_length)
        success = True
        for wire_id in range(self._num_agents):
            #jax.debug.print("wire id is: {wire_id}", wire_id=wire_id)
            #jax.debug.print("board is: {board}", board=board)
            key, board, _, _, _  = random_walk_tuple
            key, subkey = jax.random.split(key)
            # Try to start the wire at a random location
            random_walk_tuple, can_start = self.pick_start(subkey, board, wire_id, max_length)

            # If successful, try to walk randomly
            # Do this using jax.lax.cond
            random_walk_tuple, moved = jax.lax.cond(can_start, self.walk_randomly, lambda x: (x, False), random_walk_tuple)
            #jax.debug.print("moved is: {moved}", moved=moved)
            # Track if we have been successful
            success = can_start & moved & success
        return random_walk_tuple[1], success

    def generate(self, key):
        def body_fun(self, max_length_int, i, state):
            board, success = state

            def update_board_and_success(_):
                empty_board = jnp.zeros((self._rows, self._cols))
                new_board, new_success = self.add_agents(key, empty_board, max_length_int - i)
                return (new_board, new_success)

            def keep_current_state(_):
                return (board, success)

            # Call self.add_agents only if success is False
            updated_board, updated_success = jax.lax.cond(
                jnp.logical_not(success),
                None,
                update_board_and_success,
                None,
                keep_current_state
            )

            def true_fun(_):
                return (updated_board, updated_success)

            def false_fun(_):
                return (updated_board, False)

            return jax.lax.cond(updated_success, None, true_fun, None, false_fun)

        start_max_length = self._rows * self._cols // self._num_agents
        max_length_int = jax.lax.convert_element_type(start_max_length, jnp.int32).astype(int)

        board = jnp.zeros((self._rows, self._cols))
        init_state = (board, False)

        # Create a partial function with the first two arguments fixed
        body_fun_partial = partial(body_fun, self, max_length_int)

        final_board, success = jax.lax.fori_loop(1, max_length_int + 1, body_fun_partial, init_state)

        # Use jax.lax.cond to replace the if-else statement in a JIT-compatible way
        def true_fun(_):
            return final_board

        def false_fun(_):
            return jnp.zeros((self._rows, self._cols))

        return jax.lax.cond(success, None, true_fun, None, false_fun)
    
    def generate_until_non_empty(self, key):
        def cond_fun(state):
            _, board, success, _ = state
            return jnp.logical_and(jnp.logical_not(success), jnp.sum(board) == 0)

        def body_fun(state):
            key, _, _, _ = state
            new_key, sub_key = jax.random.split(key)
            new_board = self.generate(sub_key)
            new_success = jnp.sum(new_board) > 0
            return new_key, new_board, new_success, sub_key

        init_key, init_sub_key = jax.random.split(key)
        init_board = jnp.zeros((self._rows, self._cols))
        init_state = (init_key, init_board, False, init_sub_key)

        _, final_board, _, _ = jax.lax.while_loop(cond_fun, body_fun, init_state)
        return final_board
    
    def generate_starts_ends(self, key):
        """
        Call generate, take the first and last cells of each wire

        Return two arrays of dimension 2 x num_agents
        """
        board = self.generate_until_non_empty(key)
        #jax.debug.print("{x}", x=board)

        def find_positions(wire_id):
            wire_positions = board == 3 * wire_id + POSITION
            wire_targets = board == 3 * wire_id + TARGET

            # Compute indices where wire_positions and wire_targets are True
            start_indices = jnp.argwhere(wire_positions, size=2)
            end_indices = jnp.argwhere(wire_targets, size=2)

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



if __name__ == "__main__":
    board_generator = JaxRandomWalk(10, 10, 5)
    key = jax.random.PRNGKey(54)
    # jit generate
    #board_generator_jit = jax.jit(board_generator.generate)
    #print(board_generator_jit(key))
    #print(board_generator.generate(key))
    board_generator_starts_ends = board_generator.generate_starts_ends
    print(board_generator_starts_ends(key))

    # jit generate_starts_ends
    board_generator_starts_ends_jit = jax.jit(board_generator.generate_starts_ends)
    print(board_generator_starts_ends_jit(key))
    board_generate_until_non_empty_jit = jax.jit(board_generator.generate_until_non_empty)
    # Time how quick the generation is and vary the key
    # start the timer
    # start = time.time()

    # for i in range(10000):
    #     key = jax.random.PRNGKey(i)
    #     _, _ = board_generator_starts_ends_jit(key)
    # # stop the timer
    # end = time.time()
    # print("Time taken: ", end - start)
