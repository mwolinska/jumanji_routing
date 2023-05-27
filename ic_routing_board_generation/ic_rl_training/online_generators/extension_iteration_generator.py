# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
from typing import Any, Tuple
from typing import NamedTuple, Optional

import chex
import jax
from jax import numpy as jnp
from jax import Array
from jax.random import PRNGKey

from jumanji.environments.routing.connector.constants import (
    DOWN,
    EMPTY,
    LEFT,
    NOOP,
    PATH,
    POSITION,
    RIGHT,
    TARGET,
    UP,
)
from jumanji.environments.routing.connector.generator import Generator
from jumanji.environments.routing.connector.types import Agent, State
from jumanji.environments.routing.connector.utils import (
    get_agent_grid,
    get_correction_mask,
    get_position,
    get_target,
    move_agent,
    move_position,
)


class Wire(NamedTuple):
    """Define a stack usable with Jax transformations.

    - data: array of fixed-shape, each row up to insertion_index containing an element of the stack.
        Rows after insertion_index should be ignored, they only contain padding to make sure data
        is of fixed shape and can be used with Jax transformations.
        The width of the data is the number of features in an element, the height is the maximum
        number of elements the stack can contain.
    - insertion_index: the index of the row at which to insert the next element in data. Should be
        0 for an empty stack.
    """
    path: chex.Array
    insertion_index: int
    wire_id: int
    start: Tuple[int, int]
    end: Tuple[int, int]



class ExtensionIterationGenerator(Generator):
    """Randomly generates `Connector` grids that are guaranteed be solvable. This generator places
    start positions randomly on the grid and performs a random walk from each.  Targets are placed
    at their terminuses.
    """

    def __init__(self, grid_size: int, num_agents: int, extension_iterations: int = 2) -> None:
        """Instantiates a `ParallelRandomWalkGenerator.

        Args:
            grid_size: size of the square grid to generate.
            num_agents: number of agents/paths on the grid.
        """
        super().__init__(grid_size, num_agents)
        self.cols = grid_size
        self.rows = grid_size
        self.num_iterations = extension_iterations

    def __call__(self, key: chex.PRNGKey) -> State:
        """Generates a `Connector` state that contains the grid and the agents' layout.

        Returns:
            A `Connector` state.
        """
        key, _ = jax.random.split(key)

        grid = jnp.zeros((self.grid_size, self.grid_size), dtype=jnp.int32)
        starts, targets, _ = self.generate_board(key)
        starts = tuple(starts)
        targets = tuple(targets)
        agent_position_values = jax.vmap(get_position)(jnp.arange(self.num_agents))
        agent_target_values = jax.vmap(get_target)(jnp.arange(self.num_agents))

        # Transpose the agent_position_values to match the shape of the grid.
        # Place the agent values at starts and targets.
        grid = grid.at[starts].set(agent_position_values)
        grid = grid.at[targets].set(agent_target_values)

        # Create the agent pytree that corresponds to the grid.

        agents = jax.vmap(Agent)(
            id=jnp.arange(self.num_agents),
            start=jnp.stack(starts, axis=1),
            target=jnp.stack(targets, axis=1),
            position=jnp.stack(starts, axis=1),
        )

        step_count = jnp.array(0, jnp.int32)

        return State(key=key, grid=grid, step_count=step_count, agents=agents)

    def generate_board(
        self, key: chex.PRNGKey
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """Generates solvable board using iterated random walks.

        Args:
            key: random key.

        Returns:
            Tuple containing head and targets positions for each wire and the solved board
            generated in the random walk.
        """
        grid = self._return_blank_board()
        key, step_key = jax.random.split(key)
        grid, agents = self._initialise_agents(key, grid)

        stepping_tuple = (step_key, grid, agents)
        _, grid, agents = jax.lax.while_loop(
            self._continue_stepping, self._step, stepping_tuple
        )
        # Convert heads and targets to format accepted by generator
        #agents.start.T, arents.position.T = agents.position.T, agents.start.T  # Swap heads and tails so we alternate extensions
        heads = agents.start.T
        targets = agents.position.T
        heads, targets = targets, heads
        solved_grid = self.update_solved_board_with_head_target_encodings(
            grid, tuple(heads), tuple(targets)
        )
        jax.debug.print("BEFORE BFS OPT\n{x}", x=solved_grid)


        #  Iteratively apply extensions/optimizations
        #num_iterations = jnp.int32((self.grid_size * self.grid_size / self.num_agents ) / 3)
        num_iterations = 2
        #num_iterations = 1
        for iteration in range(num_iterations):
            key, optkey, extkey = jax.random.split(key, 3)
            # Optimise each wire individually
            optkeys = jax.random.split(optkey, self.num_agents)
            def optimize_wire_loop_func(wire_num, carry):
                #jax.debug.print("wire_num={x}", x=wire_num)
                board_layout, keys = carry
                board_layout = self.optimize_wire(keys[wire_num], board_layout, wire_num)
                carry = (board_layout, keys)
                return carry
            carry = (solved_grid, optkeys)
            solved_grid, _ = jax.lax.fori_loop(0, self.num_agents, optimize_wire_loop_func, carry)
            jax.debug.print("AFTER BFS OPT {i}\n{x}", i=iteration, x=solved_grid)


            key, step_key = jax.random.split(key)
            grid = solved_grid
            stepping_tuple = (step_key, grid, agents)
            _, grid, agents = jax.lax.while_loop(
                self._continue_stepping, self._step, stepping_tuple
            )
            # Convert heads and targets to format accepted by generator
            heads = agents.start.T
            targets = agents.position.T
            heads, targets = targets, heads
            solved_grid = self.update_solved_board_with_head_target_encodings(
                grid, tuple(heads), tuple(targets)
            )
            jax.debug.print("AFTER EXTENSION {i}\n{x}", i=iteration, x=solved_grid)




        return heads, targets, solved_grid

    ##############################################################################################
    # Methods for random walk

    def _step(
        self, stepping_tuple: Tuple[chex.PRNGKey, chex.Array, Agent]
    ) -> Tuple[chex.PRNGKey, chex.Array, Agent]:
        """Takes one step for all agents."""
        key, grid, agents = stepping_tuple
        key, next_key = jax.random.split(key)
        agents, grid = self._step_agents(key, grid, agents)

        #jax.debug.print("{x}", x=grid)
        return next_key, grid, agents

    def _step_agents(
        self, key: chex.PRNGKey, grid: chex.Array, agents: Agent
    ) -> Tuple[Agent, chex.Array]:
        """Steps all agents at the same time correcting for possible collisions.

        If a collision occurs we place the agent with the lower `agent_id` in its previous position.
        This method is equivalent in function to _step_agents from 'Connector' environment.

        Returns:
            Tuple of agents and grid after having applied each agents' action
        """
        agent_ids = jnp.arange(self.num_agents)
        keys = jax.random.split(key, num=self.num_agents)

        # Randomly select action for each agent
        actions = jax.vmap(self._select_action, in_axes=(0, None, 0))(
            keys, grid, agents
        )

        # Step all agents at the same time (separately) and return all of the grids
        new_agents, grids = jax.vmap(self._step_agent, in_axes=(0, None, 0))(
            agents, grid, actions
        )
        #jax.debug.print("Stepping_grid =\n{x}", x=grid)

        # Get grids with only values related to a single agent.
        # For example: remove all other agents from agent 1's grid. Do this for all agents.
        agent_grids = jax.vmap(get_agent_grid)(agent_ids, grids)
        joined_grid = jnp.max(agent_grids, 0)  # join the grids

        # Create a correction mask for possible collisions (see the docs of `get_correction_mask`)
        correction_fn = jax.vmap(get_correction_mask, in_axes=(None, None, 0))
        correction_masks, collided_agents = correction_fn(grid, joined_grid, agent_ids)
        correction_mask = jnp.sum(correction_masks, 0)
        #jax.debug.print("correction_mask={x}", x=correction_mask)
        #jax.debug.print("collided_agents={x}", x=collided_agents)

        # Correct state.agents
        # Get the correct agents, either old agents (if collision) or new agents if no collision
        agents = jax.vmap(
            lambda collided, old_agent, new_agent: jax.lax.cond(
                collided,
                lambda: old_agent,
                lambda: new_agent,
            )
        )(collided_agents, agents, new_agents)
        # Create the new grid by fixing old one with correction mask and adding the obstacles
        return agents, joined_grid + correction_mask

    def _initialise_agents(
        self, key: chex.PRNGKey, grid: chex.Array
    ) -> Tuple[chex.Array, Agent]:
        """Initialises agents using random starting point and places heads on grid.

        Args:
            key: random key.
            grid: empty grid.

        Returns:
            Tuple of grid with populated starting points and agents initialised with
            the same starting points.
        """
        starts_flat = jax.random.choice(
            key=key,
            a=jnp.arange(self.rows * self.cols),
            shape=(self.num_agents,),
            # Start positions for all agents
            replace=False,
        )

        # Create 2D points from the flat arrays.
        starts = jnp.divmod(starts_flat, self.cols)
        # Fill target with default value as targets will be assigned aftert random walk
        targets = jnp.full((2, self.num_agents), -1)

        # Initialise agents
        agents = jax.vmap(Agent)(
            id=jnp.arange(self.num_agents),
            start=jnp.stack(starts, axis=1),
            target=jnp.stack(targets, axis=1),
            position=jnp.stack(starts, axis=1),
        )
        grid = jax.vmap(self._place_agent_heads_on_grid, in_axes=(None, 0))(
            grid, agents
        )
        grid = grid.max(axis=0)
        grid = jnp.array(grid, dtype=int)
        ### IGNORE
        # jax.debug.print("board just heads {x}", x=grid)
        # keys = jax.random.split(key, num=self.num_agents)
        # actions = jax.vmap(self._select_action, in_axes=(0, None, 0))(
        #     keys, grid, agents
        # )
        # targets = jax.vmap(self._find_target, in_axes=(0, 0))(agents, actions)
        # agents.target = targets
        # grid = jax.vmap(self._place_agent_targets_on_grid, in_axes=(None, 0))(
        #     grid, agents
        # )
        # grid = grid.max(axis=0)
        # grid = jnp.array(grid, dtype=int)
        ### END IGNORE

        return grid, agents

    def _place_agent_heads_on_grid(self, grid: chex.Array, agent: Agent) -> chex.Array:
        """Updates grid with agent starting positions."""
        return grid.at[tuple(agent.start)].set(get_position(agent.id))

    def _place_agent_targets_on_grid(self, grid: chex.Array, agent: Agent) -> chex.Array:
        """Updates grid with agent starting positions."""
        return grid.at[tuple(agent.target)].set(get_target(agent.id))

    def _continue_stepping(
        self, stepping_tuple: Tuple[chex.PRNGKey, chex.Array, Agent]
    ) -> chex.Array:
        """Determines if agents can continue taking steps."""
        _, grid, agents = stepping_tuple
        dones = jax.vmap(self._no_available_cells, in_axes=(None, 0))(grid, agents)
        return ~dones.all()

    def _no_available_cells(self, grid: chex.Array, agent: Agent) -> chex.Array:
        """Checks if there are no moves are available for the agent."""
        cell = self._convert_tuple_to_flat_position(agent.position)
        return (self._available_cells(grid, cell) == -1).all()

    def _select_action(
        self, key: chex.PRNGKey, grid: chex.Array, agent: Agent
    ) -> chex.Array:
        """Selects action for agent to take given its current position.

        Args:
            key: random key.
            grid: current state of the grid.
            agent: the agent.

        Returns:
            Integer corresponding to the action the agent will take in its next step.
            Action indices match those in connector.constants.
        """
        cell = self._convert_tuple_to_flat_position(agent.position)
        available_cells = self._available_cells(grid=grid, cell=cell)
        #jax.debug.print("available cells {x}", x=available_cells)
        step_coordinate_flat = jax.random.choice(
            key=key,
            a=available_cells,
            shape=(),
            replace=True,
            p=available_cells != -1,
        )

        action = self._action_from_positions(cell, step_coordinate_flat)
        return action

    def _convert_flat_position_to_tuple(self, position: chex.Array) -> chex.Array:
        return jnp.array([(position // self.cols), (position % self.cols)], dtype=int)

    def _convert_tuple_to_flat_position(self, position: chex.Array) -> chex.Array:
        return jnp.array((position[0] * self.cols + position[1]), int)

    def _action_from_positions(
        self, position_1: chex.Array, position_2: chex.Array
    ) -> chex.Array:
        """Compares two positions and returns action id to get from one to the other."""
        position_1 = self._convert_flat_position_to_tuple(position_1)
        position_2 = self._convert_flat_position_to_tuple(position_2)
        action_tuple = position_2 - position_1
        return self._action_from_tuple(action_tuple)

    def _action_from_tuple(self, action_tuple: chex.Array) -> chex.Array:
        """Returns integer corresponding to taking action defined by action_tuple."""
        action_multiplier = jnp.array([UP, DOWN, LEFT, RIGHT, NOOP])
        actions = jnp.array(
            [
                (action_tuple == jnp.array([-1, 0])).all(axis=0),
                (action_tuple == jnp.array([1, 0])).all(axis=0),
                (action_tuple == jnp.array([0, -1])).all(axis=0),
                (action_tuple == jnp.array([0, 1])).all(axis=0),
                (action_tuple == jnp.array([0, 0])).all(axis=0),
            ]
        )
        actions = jnp.sum(actions * action_multiplier, axis=0)
        return actions

    def _adjacent_cells(self, cell: int) -> chex.Array:
        """Returns chex.Array of adjacent cells to the input.

        Given a cell, return a chex.Array of size 4 with the flat indices of
        adjacent cells. Padded with -1's if less than 4 adjacent cells (if on the edge of the grid).

        Args:
            cell: the flat index of the cell to find adjacent cells of.

        Returns:
            A chex.Array of size 4 with the flat indices of adjacent cells
            (padded with -1's if less than 4 adjacent cells).
        """
        available_moves = jnp.full(4, cell)
        direction_operations = jnp.array([-1 * self.rows, self.rows, -1, 1])
        # Create a mask to check 0 <= index < total size
        cells_to_check = available_moves + direction_operations
        is_id_in_grid = cells_to_check < self.rows * self.cols
        is_id_positive = 0 <= cells_to_check
        mask = is_id_positive & is_id_in_grid

        # Ensure adjacent cells doesn't involve going off the grid
        unflatten_available = jnp.divmod(cells_to_check, self.cols)
        unflatten_current = jnp.divmod(cell, self.cols)
        is_same_row = unflatten_available[0] == unflatten_current[0]
        is_same_col = unflatten_available[1] == unflatten_current[1]
        row_col_mask = is_same_row | is_same_col
        # Combine the two masks
        mask = mask & row_col_mask
        return jnp.where(mask, cells_to_check, -1)

    def _available_cells(self, grid: chex.Array, cell: chex.Array) -> chex.Array:
        """Returns list of cells that can be stepped into from the input cell's position.

        Given a cell and the grid of the board, see which adjacent cells are available to move to
        (i.e. are currently unoccupied) to avoid stepping over exisitng wires.

        Args:
            grid: the current layout of the board i.e. current grid.
            cell: the flat index of the cell to find adjacent cells of.

        Returns:
            A chex.Array of size 4 with the flat indices of adjacent cells.
        """
        adjacent_cells = self._adjacent_cells(cell)
        # Get the wire id of the current cell
        value = grid[jnp.divmod(cell, self.cols)]
        wire_id = (value - 1) // 3

        available_cells_mask = jax.vmap(self._is_cell_free, in_axes=(None, 0))(
            grid, adjacent_cells
        )
        # Also want to check if the cell is touching itself more than once
        touching_cells_mask = jax.vmap(
            self._is_cell_doubling_back, in_axes=(None, None, 0)
        )(grid, wire_id, adjacent_cells)
        available_cells_mask = available_cells_mask & touching_cells_mask
        available_cells = jnp.where(available_cells_mask, adjacent_cells, -1)
        return available_cells

    def _is_cell_free(
        self,
        grid: chex.Array,
        cell: chex.Array,
    ) -> chex.Array:
        """Check if a given cell is free, i.e. has a value of 0.

        Args:
            grid: the current grid of the board.
            cell: the flat index of the cell to check.

        Returns:
            Boolean indicating whether the cell is free or not.
        """
        coordinate = jnp.divmod(cell, self.cols)
        return (cell != -1) & (grid[coordinate] == 0)

    def _is_cell_doubling_back(
        self,
        # grid_wire_id: Tuple[chex.Array, int],
        grid: chex.Array,
        wire_id: int,
        cell: int,
    ) -> chex.Array:
        """Checks if moving into an adjacent position results in a wire doubling back on itself.

        Check if the cell is touching any of the wire's own cells more than once.
        This means looking for surrounding cells of value 3 * wire_id + POSITION or
        3 * wire_id + PATH.
        """
        # grid, wire_id = grid_wire_id
        # Get the adjacent cells of the current cell
        adjacent_cells = self._adjacent_cells(cell)

        def is_cell_doubling_back_inner(
            grid: chex.Array, cell: chex.Array
        ) -> chex.Array:
            coordinate = jnp.divmod(cell, self.cols)
            cell_value = grid[tuple(coordinate)]
            touching_self = (
                (cell_value == 3 * wire_id + POSITION)
                | (cell_value == 3 * wire_id + PATH)
                | (cell_value == 3 * wire_id + TARGET)
            )
            return (cell != -1) & touching_self

        # Count the number of adjacent cells with the same wire id
        doubling_back_mask = jax.vmap(is_cell_doubling_back_inner, in_axes=(None, 0))(
            grid, adjacent_cells
        )
        # If the cell is touching itself more than once, return False
        return jnp.sum(doubling_back_mask) <= 1

    def _step_agent(
        self,
        agent: Agent,
        grid: chex.Array,
        action: int,
    ) -> Tuple[Agent, chex.Array]:
        """Moves the agent according to the given action if it is possible.

        This method is equivalent in function to _step_agent from 'Connector' environment.

        Returns:
            Tuple of (agent, grid) after having applied the given action.
        """
        new_pos = move_position(agent.position, action)

        new_agent, new_grid = jax.lax.cond(
            self._is_valid_position(grid, agent, new_pos) & (action != NOOP),
            move_agent,
            lambda *_: (agent, grid),
            agent,
            grid,
            new_pos,
        )

        return new_agent, new_grid

    def _find_target(self, agent: Agent,
        action: int,
    ) -> chex.Array:
        return move_position(agent.position, action)

    def _is_valid_position(
        self,
        grid: chex.Array,
        agent: Agent,
        position: chex.Array,
    ) -> chex.Array:
        """Checks to see if the specified agent can move to `position`.

        This method is mirrors the use of to is_valid_position from the 'Connector' environment.

        Args:
            grid: the environment state's grid.
            agent: the agent.
            position: the new position for the agent in tuple format.

        Returns:
            bool: True if the agent moving to position is valid.
        """
        row, col = position
        grid_size = grid.shape[0]

        # Within the bounds of the grid
        in_bounds = (0 <= row) & (row < grid_size) & (0 <= col) & (col < grid_size)
        # Cell is not occupied
        open_cell = (grid[row, col] == EMPTY) | (grid[row, col] == get_target(agent.id))
        # Agent is not connected
        not_connected = ~agent.connected

        return in_bounds & open_cell & not_connected

    def _return_blank_board(self) -> chex.Array:
        """Return empty grid of correct size."""
        return jnp.zeros((self.rows, self.cols), dtype=int)

    def update_solved_board_with_head_target_encodings(
        self,
        solved_grid: chex.Array,
        heads: Tuple[Any, ...],
        targets: Tuple[Any, ...],
    ) -> chex.Array:
        """Updates grid array with all agent encodings."""
        agent_position_values = jax.vmap(get_position)(jnp.arange(self.num_agents))
        agent_target_values = jax.vmap(get_target)(jnp.arange(self.num_agents))
        # Transpose the agent_position_values to match the shape of the grid.
        # Place the agent values at starts and targets.
        solved_grid = solved_grid.at[heads].set(agent_position_values)
        solved_grid = solved_grid.at[targets].set(agent_target_values)
        return solved_grid

    ############################################################################################
    # Methods for optimize_wire

    def optimize_wire(self, key: PRNGKey, board: Array, wire_num: int) -> Array:
        """Essentially the same as thin_wire but also uses empty spaces to optimize the wire
        Args:
            key: key to use for random number generation
            board: board to thin the wire on
            wire_num: wire number

        Returns:
            board with the wire optimized
            """
        # # Initialize the grid
        # grid = Grid(board.shape[0], board.shape[1], fill_num=wire_num)
        # max_size = board.shape[0] * board.shape[1]
        #
        # start_num = 3 * wire_num + POSITION
        # end_num = 3 * wire_num + TARGET
        #
        # # Find start and end positions
        # wire_start = jnp.argwhere(board == start_num)[0]
        # wire_end = jnp.argwhere(board == end_num)[0]
        #
        # # Initialize the wire
        # wire = create_wire(max_size=max_size, wire_id=wire_num, start=wire_start, end=(wire_end[0], wire_end[1]))
        #
        # # Set start and end in board to the same value as wire
        # board = board.at[wire_start[0], wire_start[1]].set(grid.fill_num * 3 + 1)
        #
        # board = board.at[wire_end[0], wire_end[1]].set(grid.fill_num * 3 + 1)
        #
        # queue = jnp.zeros(max_size, dtype=jnp.int32)
        # visited = -1 * jnp.ones(max_size, dtype=jnp.int32)
        #
        # # Update queue to reflect the start position
        # queue = queue.at[grid.convert_tuple_to_int(wire.start)].set(1)
        #
        # for _ in range(max_size):
        #     bfs_stack = (key, wire, board, queue, visited)
        #     board = bfs_stack[2]
        #     queue = bfs_stack[3]
        #     visited = bfs_stack[4]
        #
        #     queue, visited, board = grid.update_queue_and_visited(queue, visited, board, key, use_empty=True)
        #
        #     if grid.check_if_end_reached(wire, visited):
        #         break
        #
        # curr_pos = grid.convert_tuple_to_int(wire.end)
        # wire = grid.get_path((wire, visited, curr_pos))
        # board = grid.remove_path(board, wire)
        #
        # board = grid.jax_fill_grid(wire, board)
        #
        # return board

        #grid = Grid(self.rows, self.cols, fill_num=wire_num)
        #jax.debug.print("optimize_wire wire_num={x}", x=wire_num)
        max_size = self.rows * self.cols

        start_num = 3 * wire_num + POSITION
        end_num = 3 * wire_num + TARGET

        flat_start = jnp.argmax(jnp.where(board == start_num, board, 0))
        wire_start = self.convert_int_to_tuple(flat_start)

        flat_end = jnp.argmax(jnp.where(board == end_num, board, 0))
        wire_end = self.convert_int_to_tuple(flat_end)

        # Initialize the wire
        wire = self.create_wire(max_size=max_size, wire_id=wire_num, start=wire_start, end=(wire_end[0], wire_end[1]))

        # Set start and end in board to the same value as wire
        board = board.at[wire_start[0], wire_start[1]].set(wire_num * 3 + 1)

        board = board.at[wire_end[0], wire_end[1]].set(wire_num * 3 + 1)

        queue = jnp.zeros(max_size, dtype=jnp.int32)
        visited = -1 * jnp.ones(max_size, dtype=jnp.int32)

        # Update queue to reflect the start position
        queue = queue.at[self.convert_tuple_to_int(wire.start)].set(1)
        #jax.debug.print("optimize_wire_boardA = \n{x}", x=board)

        def loop_body(full_stack):
            i, bfs_stack, end_reached, wire_num = full_stack
            #jax.debug.print("i={x}", x=i)
            key, wire, board, queue, visited = bfs_stack
            # Update the queue and visited arrays
            queue, visited, board = self.update_queue_and_visited(queue, visited, board, wire_num, key, use_empty=True)
            # Check if the end has been reached
            end_reached = self.check_if_end_reached(wire, visited)
            #jax.debug.print("end_reached={x}", x=end_reached)
            # Return the loop condition and the updated bfs_stack
            i += 1
            return i, (key, wire, board, queue, visited), end_reached, wire_num
        def loop_cond(full_stack):
            i, bfs_stack, end_reached, wire_num = full_stack
            return jnp.logical_and(i < (self.cols * self.rows),jnp.logical_not(end_reached))
        i = 0
        #loop_cond = lambda full_stack: jnp.logical_and(full_stack[0] < (self.cols * self.rows),
        #                                               jnp.logical_not(full_stack[-2]))
        #jax.debug.print("visitedB={x}", x=visited)
        end_reached = False
        bfs_stack = (key, wire, board, queue, visited)
        full_stack = (i, bfs_stack, end_reached, wire_num)
        #.print("optimize_wire_boardB = \n{x}", x=board)
        #jax.debug.print("i={x}  {y}", x=full_stack[0], y=(full_stack[0] < (self.cols * self.rows)))
        #jax.debug.print("wire_num={x} {y}", x=full_stack[-1], y=jnp.logical_not(full_stack[-1]))
        final_i, final_bfs_stack, end_reached, wire_num = jax.lax.while_loop(loop_cond, loop_body, full_stack)
        #jax.debug.print("optimize_wire_boardC = \n{x}", x=board)
        _, wire, board, _, visited = final_bfs_stack
        #jax.debug.print("visitedC={x}", x=visited)
        #jax.debug.print("wire.path={x}", x=wire.path)
        #jax.debug.print("optimize_wire_boardC2 = \n{x}", x=board)

        curr_pos = self.convert_tuple_to_int(wire.end)
        wire = self.get_path((wire, visited, curr_pos))
        board = self.remove_path(board, wire)
        #jax.debug.print("wire.path={x}", x=wire.path)
        #jax.debug.print("optimize_wire_boardD = \n{x}", x=board)
        board = self.jax_fill_grid(wire, board)
        #jax.debug.print("optimize_wire_boardE= \n{x}", x=board)
        return board

    def convert_tuple_to_int(self, position: Tuple[int, int]) -> Array:
        """Converts a tuple to an integer format but in a jax array"""
        return jnp.array(position[0] * self.cols + position[1], dtype=jnp.int32)

    def convert_int_to_tuple(self, position: int) -> Array:
        """Converts an integer to a tuple format but in a jax array"""
        return jnp.array([position // self.cols, position % self.cols], dtype=jnp.int32)

    def check_if_end_reached(self, wire: Wire, visited: Array) -> bool:
        """Check if the end of the wire has been reached"""
        # Convert wire.end to int
        end_int = self.convert_tuple_to_int(wire.end)
        return visited[end_int] != -1

    def remove_path(self, board: Array, wire: Wire) -> Array:
        """Removes a wire path from the board before repopulating the new path
        paths are encoded as 3 * wire_num + PATH
        starts are encoded as 3 * wire_num + POSITION
        ends are encoded as 3 * wire_num + TARGET

        Args:
            board: board to remove the wire from
            wire: wire to remove from the board
        Returns:
            board with the wire removed from it
            """
        # Get the wire number
        wire_fill_num = 3 * wire.wire_id + PATH
        # Populate the cells in the board that have the wire_fill_num to 0
        board = jnp.where(board == wire_fill_num, 0, board)
        return board

    def jax_fill_grid(self, wire: Wire, board: Array) -> Array:
        """Places a wire path on the board in a Jax way
                paths are encoded as 3 * wire_num + 1
                starts are encoded as 3 * wire_num + 2
                ends are encoded as 3 * wire_num + 3

            Args:
                wire: wire to place on the board
                board: board to place the wire on
            Returns:
                board with the wire placed on it
        """
        #jax.debug.print("wire.path={x}", x=wire.path)
        board_flat = board.ravel()
        # Need to populate the start and end positions and the path positions using a jax while loop.
        # The loop will stop when -1 is reached in the path
        def cond_fun(i_board_tuple):
            i, board_flat = i_board_tuple
            return wire.path[i] != -1

        def body_fun(i_board_tuple):
            i, board_flat = i_board_tuple
            board_flat = board_flat.at[wire.path[i]].set(3 * wire.wire_id + PATH)
            return i + 1, board_flat

        i, board_flat = jax.lax.while_loop(cond_fun, body_fun, (0, board_flat))
        # update the start and end positions
        board_flat = board_flat.at[wire.path[i -1]].set(3 * wire.wire_id + POSITION)
        board_flat = board_flat.at[wire.path[0]].set(3 * wire.wire_id + TARGET)
        # reshape the board
        board = board_flat.reshape(board.shape)
        return board

    def get_path(self, wire_visited_tuple: Tuple[Wire, Array, int]) -> Wire:
        """Populates wire path using wire operations and visited array

        Args:
            wire_visited_tuple (Tuple[Wire, Array, int]): Tuple containing wire, visited array and current position

        Returns:
            Wire: Wire with path populated

        """
        wire, visited, cur_poss = wire_visited_tuple
        #jax.debug.print("Awire.path={x}", x=wire.path)
        #jax.debug.print("Awire={x}", x=wire)
        #jax.debug.print("Avisited={x}",x=visited)

        wire_start = self.convert_tuple_to_int(wire.start)

        # Using a jax while loop update the wire path. The loop will stop when the wire start is reached
        def cond_fun(wire_visited_tuple):
            wire, visited, _ = wire_visited_tuple
            # Check that last position in wire path is not the wire start
            # Check the previous input which will be max of 0 and insertion index - 1
            index_to_check = jnp.max(jnp.array([0, wire.insertion_index - 1]))
            return wire.path[index_to_check] != wire_start

        def body_fun(wire_visited_tuple):
            wire, visited, cur_poss = wire_visited_tuple
            # Get the next position
            next_pos = visited[cur_poss]
            wire = self.stack_push(wire, cur_poss)
            #jax.debug.print("loopwire.path={x}", x=wire.path)
            wire_visited_tuple = (wire, visited, next_pos)
            return wire_visited_tuple

        wire_visited_tuple = jax.lax.while_loop(cond_fun, body_fun, wire_visited_tuple)
        wire, visited, _ = wire_visited_tuple
        # wire = stack_reverse(wire)
        #jax.debug.print("Bwire.path={x}", x=wire.path)
        return wire

    def create_wire(self,
        max_size: int,
        start: Tuple[int, int],
        end: Tuple[int, int],
        wire_id: int
    ) -> Wire:
        """Create an empty stack.

        Args:
            max_size: maximum number of elements the stack can contain. GRID SIZE


        Returns:
            stack: the created stack of size grid size x 2 - i.e. rows and columns
        """
        return Wire(jnp.full((max_size), fill_value=-1, dtype=int), 0, wire_id=wire_id, start=start, end=end)


    def update_queue_and_visited(self, queue: Array, visited: Array, board: Array, wire_num: int, key: Optional[PRNGKey] = 100,
                                 use_empty: Optional[bool] = False) -> Tuple[Array, Array, Array]:
        """Updates the queue and visited arrays

        Args:
            queue (Array): Array indicating the next positions to visit
            visited (Array): Array indicating the previous state visited from each position
            board (Array): Array indicating the current state of the board
            key (Optional[PRNGKey]): Key used for random number generation
            use_empty (Optional[bool]): Boolean indicating whether to use empty spaces as valid positions

        Returns:
            Tuple[Array, Array, Array]: Updated queue, visited and board arrays
        """
        #jax.debug.print("updateQandV_wire_num={x}", x=wire_num)
        # Current position is the lowest value in the queue that is greater than 0
        curr_int = jnp.argmin(jnp.where((queue > 0), queue, jnp.inf))
        # Convert current position to tuple
        curr_pos = self.convert_int_to_tuple(curr_int)
        # Define possible movements
        row = [-1, 0, 1, 0]
        col = [0, 1, 0, -1]
        # Shuffle row and col in the same way
        perm = jax.random.permutation(key, jnp.arange(4), independent=True)
        row = jnp.array(row, dtype=jnp.int32)[perm]
        col = jnp.array(col, dtype=jnp.int32)[perm]
        # Do a jax while loop of the update_queue_visited_loop
        def qv_loop_cond(full_qv_stack):
            j, *_ = full_qv_stack
            return j < 4

        def qv_loop_body(full_qv_stack):
            j, curr_pos, curr_int, row, col, visited, queue, board, wire_num = full_qv_stack
            j, curr_pos, curr_int, row, col, visited, queue, board, wire_num = self.update_queue_visited_loop(j, curr_pos,
                                                                                                    curr_int, row, col,
                                                                                                    visited, queue,
                                                                                                    board, wire_num,
                                                                                                    use_empty)
            return j, curr_pos, curr_int, row, col, visited, queue, board, wire_num

        j = 0
        full_qv_stack = (j, curr_pos, curr_int, row, col, visited, queue, board, wire_num)
        full_qv_stack = jax.lax.while_loop(qv_loop_cond, qv_loop_body, full_qv_stack)
        *_, visited, queue, board, _ = full_qv_stack
        # remove current position from queue
        queue = queue.at[curr_int].set(0)
        return queue, visited, board

    def update_queue_visited_loop(self, j, curr_pos, curr_int, row, col, visited, queue, board, wire_num, use_empty=False):
        # Calculate new position
        #jax.debug.print("{x}", x=j)
        new_row = jnp.array(curr_pos, dtype=jnp.int32)[0] + jnp.array(row, dtype=jnp.int32)[j]
        new_col = jnp.array(curr_pos, dtype=jnp.int32)[1] + jnp.array(col, dtype=jnp.int32)[j]
        pos_int = self.convert_tuple_to_int((new_row, new_col))
        # Check value of new position index in visited
        size_cond = jnp.logical_and(jnp.logical_and(0 <= new_row, new_row < self.rows),
                                    jnp.logical_and(0 <= new_col, new_col < self.cols))
        cond_1 = (visited[pos_int] == -1)
        cond_2 = (queue[pos_int] == 0)
        cond_3 = jax.lax.cond(use_empty, lambda _: jnp.logical_or((board[new_row, new_col] == 3 * wire_num + PATH),
                                                                  (board[new_row, new_col] == EMPTY)),
                              lambda _: (board[new_row, new_col] == 3 * wire_num + PATH), None)
        condition = jax.lax.cond((size_cond & cond_1 & cond_2 & cond_3), lambda _: True, lambda _: False, None)
        curr_val = jnp.max(jnp.where((queue > 0), queue, -jnp.inf))
        queue = jax.lax.cond(
            condition, lambda _: queue.at[pos_int].set(curr_val + 1), lambda _: queue, None)
        visited = jax.lax.cond(condition, lambda _: visited.at[pos_int].set(curr_int), lambda _: visited, None)
        return j + 1, curr_pos, curr_int, row, col, visited, queue, board, wire_num


    def stack_push(self, stack: Wire, element: chex.Array) -> Wire:
        """Push an element on top of the stack.

        Args:
            stack: the stack on which to push element.
            element: the element to push on the stack.

        Returns:
            stack: the stack containing the new element.
        """
        return Wire(start=stack.start, end = stack.end,
                    wire_id=stack.wire_id,
                    path=stack.path.at[stack.insertion_index].set(element),
                    insertion_index=stack.insertion_index + 1)


########################################################################################################

if __name__ == '__main__':
    """
    board_gen = ExtensionIterationGenerator(5, 3)
    board = board_gen._return_blank_board()
    key = jax.random.PRNGKey(1)
    grid, agents = board_gen._initialise_agents(key=key, grid=board)
    #print(grid)
    board = board_gen.generate_board(key)
    print(board[2])
    """

    board_gen = ExtensionIterationGenerator(10,5)
    board = board_gen._return_blank_board()
    key = jax.random.PRNGKey(1)
    grid, agents = board_gen._initialise_agents(key=key, grid=board)
    #print(grid)
    board = board_gen.generate_board(key)
    print(board[2])
