import argparse
import jax
import matplotlib.pyplot as plt

from ic_routing_board_generation.interface.board_generator_interface_numpy import BoardGenerator, BoardName

from jumanji.environments.routing.connector.viewer import ConnectorViewer

parser = argparse.ArgumentParser()

board_choices = [board.value for board in BoardName]

parser.add_argument(
    "--board_type",
    default="offline_parallel_rw",
    type=str,
    choices=BoardGenerator.board_generator_dict.keys()
)
parser.add_argument(
    "--board_size",
    default=10,
    type=int
)
parser.add_argument(
    "--num_agents",
    default=5,
    type=int
)
parser.add_argument(
    "--seed",
    default=0,
    type=int
)


if __name__ == "__main__":
    args = parser.parse_args()
    key = jax.random.PRNGKey(args.seed)

    board_generator = BoardGenerator.get_board_generator(BoardName(args.board_type))
    initial_board = board_generator(args.board_size, args.board_size, args.num_agents)
    # Generate depending on board type
    if args.board_type == "random_walk":
        board = initial_board.return_training_board()
    elif args.board_type == "offline_parallel_rw":
        board = initial_board.generate_board(key)
    elif args.board_type == "bfs_base":
        board = initial_board.generate_boards(1)
    elif args.board_type == "bfs_min_bend" or args.board_type == "bfs_fifo" or args.board_type == "bfs_short" or args.board_type == "bfs_long":
        board = initial_board.return_training_board()
    elif args.board_type == "lsystems_standard":
        board = initial_board.return_training_board()
    elif args.board_type == "wfc":
        board = initial_board.return_training_board()
    elif args.board_type == "numberlink":
        board = initial_board.return_training_board()

    print(board)


    # Render the board using jumanji's method
    viewer = ConnectorViewer("Ben", args.num_agents)
    # Make sure the board is a jnp array
    board = jax.numpy.array(board)
    viewer.render(board)
    plt.show()
