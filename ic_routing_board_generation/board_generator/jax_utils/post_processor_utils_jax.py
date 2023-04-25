import random
from typing import List, Tuple
import numpy as np
from copy import deepcopy
from ic_routing_board_generation.board_generator.numpy_board_generation.bfs_board import BFSBoard
from ic_routing_board_generation.board_generator.numpy_board_generation.board_generator_random_walk_rb import RandomWalkBoard
from ic_routing_board_generation.board_generator.numpy_board_generation.board_generator_wfc_oj import WFCBoard
from ic_routing_board_generation.board_generator.numpy_board_generation.lsystems_numpy import LSystemBoardGen
from ic_routing_board_generation.board_generator.numpy_data_model.board_generator_data_model import Position
from ic_routing_board_generation.board_generator.numpy_data_model.abstract_board import AbstractBoard


# EMPTY, PATH, POSITION, TARGET = 0, 1, 2, 3  # Ideally should be imported from Jumanji
from jumanji.environments.routing.connector.constants import EMPTY, PATH, POSITION, TARGET
STARTING_POSITION = POSITION  # Resolve ambiguity of POSITION constant


# Define exceptions for the board validity checks
class IncorrectBoardSizeError(Exception):
    """ Raised when a board size does not match the specified dimensions."""
    pass


class NumAgentsOutOfRangeError(Exception):
    """ Raised when self._wires_on_board is negative."""
    pass


class EncodingOutOfRangeError(Exception):
    """ Raised when one or more cells on the board have an invalid index."""
    pass


class DuplicateHeadsTailsError(Exception):
    """ Raised when one of the heads or tails of a wire is duplicated."""
    pass


class MissingHeadTailError(Exception):
    """ Raised when one of the heads or tails of a wire is missing."""
    pass


class InvalidWireStructureError(Exception):
    """ Raised when one or more of the wires has an invalid structure, e.g. looping or branching."""
    pass


class PathNotFoundError(Exception):
    """ Raised when a path cannot be found between a head and a target."""
    pass



####################################################################################################
# JAX UTILS ########################################################################################
from jax import Array
from jax.random import PRNGKey
import jax.numpy as jnp
import jax
INVALID = jnp.array((-999, -999))  # Encoding for an invalid 2D position for lists that must be of constant length
INVALID4 = jnp.array([(-999, -999), (-999, -999), (-999, -999), (-999, -999)]) # Encoding for list of four invalid positions


#@jax.disable_jit()
#@jax.jit
def extend_wires_jax(board_layout: Array, key: PRNGKey, randomness: float = 0.0,
                     two_sided: bool = True, extension_steps: int = 1e23) -> Array:
    """ Extend the heads and targets of each wire as far as they can go, preference given to current direction.
        The implementation is done in-place on self.board_layout

        Args:
            board_layout (Array): 2D layout of the encoded board (before wire extension)
            key( PRNGKey): a random key to use for randomly selecting directions (as needed)
            randomness (float): How randomly to extend the wires, 0=>Keep same direction if possible, 1=>Random
            two_sided (bool): True => £xtend both heads and targets.  False => Only extend targets.

        Returns:
            (Array): 2D layout of the encoded board (after wire extension)
    """
    rows, cols = board_layout.shape
    # Assuming a valid board, we find the number of wires from the value of the TARGET cell of the last wire
    num_wires = jnp.max(board_layout) // 3
    num_extendables = jax.lax.select(two_sided, 2 * num_wires, num_wires)
    prev_layout = board_layout.at[0, 0].add(1)  # Make prev_layout != board_layout
    # Continue as long as the algorithm is still changing the board
    step_num = 0
    def while_cond(carry):
        prev_layout, board_layout, _, step_num = carry
        return ~jnp.array_equal(prev_layout, board_layout) & (step_num < extension_steps)

    def while_body(carry):
        prev_layout, board_layout, key, step_num = carry
        step_num = step_num + 1
        # print("Extension Step ", step_num)

        # Randomly flip/flop the board between iterations to remove any directional bias.
        key, flipkey, flopkey = jax.random.split(key, 3)
        do_flip = jax.random.choice(flipkey, jnp.array([True, False]))
        do_flop = jax.random.choice(flopkey, jnp.array([True, False]))
        board_layout = jax.lax.select(do_flip, board_layout[::-1, :], board_layout)
        board_layout = jax.lax.select(do_flop, board_layout[:, ::-1], board_layout)

        # Make a copy of the board_layout to check if it changes
        prev_layout = deepcopy(board_layout)





        """
        # TRY JUST SAMPLING THE EXTENDABLES
        targets_bool = (((board_layout // 3) == TARGET) & (board_layout != 0))
        targets_pos = jnp.argwhere(targets_bool, size=num_extendables, fill_value=(-999, -999))
        print("targets=", targets_pos)
        starts_targets_bool = (((board_layout // 3) != PATH) & (board_layout != 0))
        starts_targets_pos = jnp.argwhere(starts_targets_bool, size=num_extendables, fill_value=(-999, -999))
        print("starts_targets=", starts_targets_pos)
        extendables_pos = jax.lax.select(two_sided, starts_targets_pos, targets_pos)
        print("extendables_pos=", extendables_pos)
        """






        def row_func(row, carry):
            board_layout, key = carry
            def col_func(col, carry):
                board_layout, key = carry

                # Get the list of neighbors available to extend to.
                current_pos = jnp.array((row, col))
                poss_extension_list = get_open_adjacent_cells_jax(current_pos, board_layout)

                current_wire_num = position_to_wire_num_jax(current_pos, board_layout)
                # For each possible cell, mark it INVALID if it already touches part of the same wire.
                def invalidate_func(cell_pos, current_wire_num, board_layout):
                    num_adjacencies = num_wire_adjacencies_jax(cell_pos, current_wire_num, board_layout)
                    cell_or_invalid = jax.lax.select(num_adjacencies > 1, INVALID, cell_pos)
                    return cell_or_invalid
                poss_extension_list = jax.vmap(invalidate_func, in_axes=(0, None, None))(poss_extension_list, current_wire_num, board_layout)

                """  WHY DOESN'T THIS GIVE THE SAME RESULT??!?!!!?!?
                def invalidate_extension_list_func(i, carry):
                    poss_extension_list, current_wire_num, board_layout = carry
                    cell_pos = poss_extension_list[i]
                    num_adjacencies = num_wire_adjacencies_jax(cell_pos, current_wire_num, board_layout)
                    cell_or_invalid = jax.lax.select(num_adjacencies > 1, INVALID, cell_pos)
                    poss_extension_list = poss_extension_list.at[i].set(cell_or_invalid)
                    carry = (poss_extension_list, current_wire_num, board_layout)
                    return carry

                carry = (poss_extension_list_save, current_wire_num, board_layout)
                poss_extension_list2, _, _ = jax.lax.fori_loop(0, 3, invalidate_extension_list_func, carry)
                """

                # Get the position of the wire that led into this cell and prioritize continuing the same direction
                prev_neighbor = get_previous_neighbor_jax(current_pos, board_layout)
                priority_neighbor = (row + (row-prev_neighbor[0]), col + (col-prev_neighbor[1]))
                priority_neighbor = jnp.array(priority_neighbor)

                # If the current cell is not a head or target, it's not extendable,
                # so change everything in the list to INVALID
                cell_type = position_to_cell_type_jax(current_pos, board_layout)
                is_extendable = (two_sided & ((cell_type == STARTING_POSITION) | (cell_type == TARGET))) | \
                                (~two_sided & (cell_type == TARGET))
                poss_extension_list = jax.lax.select(is_extendable, poss_extension_list, INVALID4)

                # If there are no valid options to extend to, stop extending
                stop_extension = jnp.array_equal(poss_extension_list, INVALID4)
                poss_extension_list = jax.lax.select(stop_extension,
                                                     jnp.array([current_pos, current_pos, current_pos, current_pos]),
                                                     poss_extension_list)

                # Pick a random selection from the possible extensions until we pick a valid one.
                def choice_body_func(carry):
                    poss_extension_list, extension_pos, key = carry
                    key, choice_key = jax.random.split(key)
                    extension_pos = jax.random.choice(choice_key, poss_extension_list)
                    carry = (poss_extension_list, extension_pos, key)
                    return carry

                def choice_cond_func(carry):
                    _, extension_pos, _ = carry
                    return jnp.array_equal(extension_pos, INVALID)

                # Is the prioity neighbor available?
                extension_pos = INVALID
                _, extension_pos, _ = jax.lax.while_loop(choice_cond_func, choice_body_func,
                                                         (poss_extension_list, extension_pos, key))

                def priority_found_func(cell_pos):
                    item_match = jnp.array_equal(cell_pos, priority_neighbor)
                    return item_match
                priority_neighbor_available = jnp.any(jax.vmap(priority_found_func)(poss_extension_list))

                # Use random choice if we are in random mode.
                # Otherwise rioritize extending away from the previous neighbor if possible, random if not available.
                key, random_key = jax.random.split(key)
                use_random = (randomness > jax.random.uniform(random_key))
                extension_pos = jax.lax.select(priority_neighbor_available & ~use_random,
                                               priority_neighbor, extension_pos)

                # Extend head/target into new cell
                board_layout = board_layout.at[extension_pos[0], extension_pos[1]]\
                                            .set(board_layout[current_pos[0], current_pos[1]])
                cell_type = position_to_cell_type_jax(current_pos, board_layout)
                # Convert old head/target cell to a wire (unless stop_extension is True)
                cell_type_offset = jax.lax.select(stop_extension, 0, PATH - cell_type)
                board_layout = board_layout.at[current_pos[0], current_pos[1]].add(cell_type_offset)


                return board_layout, key
            # THESE FORI LOOPS DON'T IMPROVE COMPILATION TIME MUCH COMPARED TO NESTED FOR LOOPS
            carry = (board_layout, key)
            board_layout, key = jax.lax.fori_loop(0, cols, col_func, carry)
            return board_layout, key
        carry = (board_layout, key)
        board_layout, key = jax.lax.fori_loop(0, rows, row_func, carry)

        """ THIS FLATTENED FORI LOOP WORKS BIT IT MAKES COMPILE TIME LONGER?!
        carry = (board_layout, key)
        board_layout, key = jax.lax.fori_loop(0, rows*cols, row_col_func, carry)
        """

        # Undo random flip/flopping to compare to prev_layout
        board_layout = jax.lax.select(do_flip, board_layout[::-1, :], board_layout)
        board_layout = jax.lax.select(do_flop, board_layout[:, ::-1], board_layout)
        # For jax.lax.while_loop
        carry = (prev_layout, board_layout, key, step_num)
        return carry

    # For jax.lax.while_loop
    carry = (prev_layout, board_layout, key, step_num)
    _, board_layout, _, _ = jax.lax.while_loop(while_cond, while_body, carry)
    return board_layout


#@jax.jit
# This method is used by the extend_wires_jax method
def get_previous_neighbor_jax(input_pos: jnp.array, board_layout: Array) -> jnp.array:
    """ Returns the position of an adjacent cell of the same wire as the current cell.

        Note that if the input cell is a head or target, then there will only be one adjacent cell in the wire.
        If the current cell is a PATH, it will return one of the two adjacent cells in the wire.
        If the current cell is EMPTY, then it will return something INVALID

        Args:
            input_pos (jnp.array): 2D position of cell in board_layout
            board_layout (Array): 2D layout of board with wires encoded

        Returns:
            (jnp.array) : 2D position of a cell of the same wire.
    """
    row, col = input_pos[0], input_pos[1]
    input_wire_num = position_to_wire_num_jax(input_pos, board_layout)
    neighbors_positions = jnp.array([(row-1, col), (row+1, col), (row, col-1), (row, col+1)])
    # Check each of four neighbors to see if it is part of the same wire
    def matching_wire_func(i, output_pos):
        neighbor_position = neighbors_positions[i]
        neighbor_wire_num = position_to_wire_num_jax(neighbor_position, board_layout)
        is_prev_neighbor = (neighbor_wire_num == input_wire_num) & \
                   (neighbor_position[0] >= 0) & (neighbor_position[0] < board_layout.shape[0]) & \
                   (neighbor_position[1] >= 0) & (neighbor_position[1] < board_layout.shape[1])
        output_pos = jax.lax.select(is_prev_neighbor, neighbor_position, output_pos)
        return output_pos
    matching_pos = jax.lax.fori_loop(0, 3, matching_wire_func, INVALID)
    return matching_pos


#@jax.jit
# This method is used by the extend_wires_jax method
def get_neighbors_same_wire_jax(input_pos: jnp.array, board_layout: Array) -> jnp.array:
    """ Returns a list of adjacent cells belonging to the same wire.

        Args:
            input_pos (jnp.array): 2D position of cell in board_layout
            board_layout (Array): 2D layout of board with wires encoded

        Returns:
            (jnp.array) : a list of cells (2D positions) adjacent to the queried cell which belong to the same wire
                            The output will always have four entries but some will be INVALID.
    """
    output_list = []
    wire_num = position_to_wire_num_jax(input_pos, board_layout)
    row, col = input_pos[0], input_pos[1]
    pos_up = jnp.array((row - 1, col))
    pos_down = jnp.array((row + 1, col))
    pos_left = jnp.array((row, col - 1))
    pos_right = jnp.array((row, col + 1))
    # Check each of the four directions
    adjacent_cell = jax.lax.select(position_to_wire_num_jax(pos_up, board_layout) == wire_num,
                                   pos_up, INVALID)
    output_list.append(adjacent_cell)
    adjacent_cell = jax.lax.select(position_to_wire_num_jax(pos_down, board_layout) == wire_num,
                                   pos_down, INVALID)
    output_list.append(adjacent_cell)
    adjacent_cell = jax.lax.select(position_to_wire_num_jax(pos_left, board_layout) == wire_num,
                                   pos_left, INVALID)
    output_list.append(adjacent_cell)
    adjacent_cell = jax.lax.select(position_to_wire_num_jax(pos_right, board_layout) == wire_num,
                                   pos_right, INVALID)
    output_list.append(adjacent_cell)
    #  There will always be four items in the list but some of them might be 'INVALID'
    output_list = jnp.array(output_list)
    return output_list


#@jax.jit
# This method is used by the extend_wires_jax method
def num_wire_adjacencies_jax(input_pos: jnp.array, wire_num: Array, board_layout: Array) -> Array:
    """ Returns the number of cells adjacent to cell which belong to the wire specified by wire_num.

        Args:
            input_pos (tuple): 2D position in board_layout
            wire_num (int): Count adjacent contacts with this specified wire.
            board_layout (Array): 2D array of board with wires encoded

        Returns:
            (Array) : The number of adjacent cells belonging to the specified wire
    """
    num_adjacencies = 0
    # Check above
    wire_adjacent = position_to_wire_num_jax(jnp.array((input_pos[0] - 1, input_pos[1])), board_layout)
    num_adjacencies = num_adjacencies + jax.lax.select(wire_adjacent == wire_num, 1, 0)
    # Check below
    wire_adjacent = position_to_wire_num_jax(jnp.array((input_pos[0] + 1, input_pos[1])), board_layout)
    num_adjacencies = num_adjacencies + jax.lax.select(wire_adjacent == wire_num, 1, 0)
    # Check left
    wire_adjacent = position_to_wire_num_jax(jnp.array((input_pos[0], input_pos[1] - 1)), board_layout)
    num_adjacencies = num_adjacencies + jax.lax.select(wire_adjacent == wire_num, 1, 0)
    # Check right
    wire_adjacent = position_to_wire_num_jax(jnp.array((input_pos[0], input_pos[1] + 1)), board_layout)
    num_adjacencies = num_adjacencies + jax.lax.select(wire_adjacent == wire_num, 1, 0)
    return num_adjacencies


#@jax.jit
# This method is used by the extend_wires_jax method
def get_open_adjacent_cells_jax(input_pos: jnp.array, board_layout: Array) -> jnp.array:
    """ Returns a list of open cells adjacent to the input cell.

        Args:
            input_pos (Array, Tuple[int, int]): The 2D position of the input cell to search adjacent to.
            board_layout (Array): 2D layout of the board with wires encoded

        Returns:
            (jnp.array): List of 2D integer tuples, up to four available cells adjacent to the input cell.
    """
    rows, cols = board_layout.shape
    adjacent_list = []
    # Check above, below, to the left and the right and add those cells to the list if available.
    # Check above
    adjacent_x, adjacent_y = input_pos[0] - 1, input_pos[1]
    adjacent_cell = jax.lax.select((adjacent_x >= 0) & (board_layout[adjacent_x, adjacent_y] == EMPTY),
                                   jnp.array((adjacent_x, adjacent_y)), INVALID)
    adjacent_list.append(adjacent_cell)
    # Check left
    adjacent_x, adjacent_y = input_pos[0], input_pos[1] - 1
    adjacent_cell = jax.lax.select((adjacent_y >= 0) & (board_layout[adjacent_x, adjacent_y] == EMPTY),
                                   jnp.array((adjacent_x, adjacent_y)), INVALID)
    adjacent_list.append(adjacent_cell)
    # Check below
    adjacent_x, adjacent_y = input_pos[0] + 1, input_pos[1]
    adjacent_cell = jax.lax.select((adjacent_x < rows) & (board_layout[adjacent_x, adjacent_y] == EMPTY),
                                   jnp.array((adjacent_x, adjacent_y)), INVALID)
    adjacent_list.append(adjacent_cell)
    # Check right
    adjacent_x, adjacent_y = input_pos[0], input_pos[1] + 1
    adjacent_cell = jax.lax.select((adjacent_y < cols) & (board_layout[adjacent_x, adjacent_y] == EMPTY),
                                   jnp.array((adjacent_x, adjacent_y)), INVALID)
    adjacent_list.append(adjacent_cell)
    #  There will always be four items in the list but some of them might be 'INVALID'
    adjacent_list = jnp.array(adjacent_list)
    return adjacent_list


#@jax.jit
def position_to_wire_num_jax(position: jnp.array, board_layout: Array) -> Array:
    """ Returns the wire number of the given cell position

        Args:
            position (jnp.array[int, int]): 2D tuple of the [row, col] position
            board_layout (Array): 2D layout of the board with wires encoded

        Returns:
            (int) : The wire number that the cell belongs to.
                    Returns -1 if not part of a wire or out-of-bounds.
    """
    row, col = position[0], position[1]
    rows, cols = board_layout.shape
    cell_encoding = board_layout[row, col]
    wire_num = jax.lax.select((0 <= row) & (row < rows) & (0 <= col) & (col < cols),
                              cell_encoding_to_wire_num_jax(cell_encoding), -1)
    return wire_num


#@jax.jit
def cell_encoding_to_wire_num_jax(cell_encoding: Array) -> Array:
    """ Returns the wire number of the given cell value

        Args:
            cell_encoding (int) : the value of the cell in self.layout

        Returns:
            (int) : The wire number that the cell belongs to. Returns -1 if not part of a wire.
    """
    #output = jax.lax.select(cell_encoding == 0, -1, int((cell_encoding - 1) // 3))
    output = jax.lax.select(cell_encoding == 0, -1, (cell_encoding - 1) // 3)
    return jax.lax.convert_element_type(output, np.int32)


#@jax.jit
def position_to_cell_type_jax(position: jnp.array, board_layout: Array) -> Array:
    """ Return the type of cell at position (row, col) in board_layout
        0 = empty
        1 = path
        2 = position (starting position)
        3 = target

        Args:
            position (jnp.array) : The 2D position of the cell
            board_layout (Array): 2D layout of the encoded board

        Returns:
            (int) : The type of cell (0-3) as detailed above.
    """
    cell_encoding = board_layout[position[0], position[1]]
    cell_type = jax.lax.select(cell_encoding == 0, cell_encoding, ((cell_encoding - 1) % 3) + 1)
    return cell_type


#@jax.jit
def training_board_from_solved_board_jax(board_layout: Array) -> Array:
    """ Zero out and PATH cells to convert from a solved board to a training board.

        Args:
            board_layout (Array): 2D layout of the encoded solved board

        Returns:
            (Array): 2D layout of the encoded training board
    """
    board_path_bool = ((board_layout % 3) == PATH)
    board_layout_paths_removed = board_layout * ~board_path_bool
    return board_layout_paths_removed
    """  OLD CODE THAT WAS REPLACED BY MORE EFFICIENT CODE ABOVE
    rows, cols = board_layout.shape
    for row in range(rows):
        for col in range(cols):
            old_encoding = board_layout[row, col]
            new_encoding = jax.lax.select((old_encoding % 3) == PATH, 0, old_encoding)
            board_layout = board_layout.at[row, col].set(new_encoding)
    return board_layout
    """



#TODO JAXIFY
def count_detours_jax(board_layout: Array, count_current_wire: bool = False) -> jnp.array:
    """ Returns the number of wires that have to detour around a head or target cell.

        Args:
            board_layout (Array): 2D layout of the board with wires encoded
            count_current_wire (bool): Should we count wires that wrap around their own heads/targets?
                                            (default = False)

        Returns:
            (jnp.array(int)) : The number of wires that have to detour around a head or target cell.
    """
    rows, cols = board_layout.shape
    num_detours = 0
    for x in range(rows):
        for y in range(cols):
            cell_type = position_to_cell_type(Position(x, y), board_layout)
            if (cell_type != STARTING_POSITION) and (cell_type != TARGET):
                continue
            current_wire = position_to_wire_num(Position(x, y), board_layout)
            #
            above = board_layout[:x, y]
            above = [cell_label_to_wire_num(cell_label) for cell_label in above if cell_label != 0]
            if not count_current_wire:
                above = [wire_num for wire_num in above if wire_num != current_wire]
            below = board_layout[x + 1:, y]
            below = [cell_label_to_wire_num(cell_label) for cell_label in below if cell_label != 0]
            if not count_current_wire:
                below = [wire_num for wire_num in below if wire_num != current_wire]
            common = (set(above) & set(below))
            num_detours += len(common)
            #
            left = board_layout[x, :y].tolist()
            left = [cell_label_to_wire_num(cell_label) for cell_label in left if cell_label != 0]
            if not count_current_wire:
                left = [wire_num for wire_num in left if wire_num != current_wire]
            right = board_layout[x, y+1:].tolist()
            right = [cell_label_to_wire_num(cell) for cell in right if cell != 0]
            if not count_current_wire:
                right = [wire_num for wire_num in right if wire_num != current_wire]
            common = (set(right) & set(left))
            num_detours += len(common)
    return num_detours

