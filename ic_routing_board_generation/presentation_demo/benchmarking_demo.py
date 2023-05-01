from ic_routing_board_generation.benchmarking.benchmark_data_model import \
    BoardGenerationParameters
from ic_routing_board_generation.benchmarking.benchmark_utils import \
    board_generation_params_from_grid_params
from ic_routing_board_generation.benchmarking.empty_board_evaluation import \
    evaluate_generator_outputs_averaged_on_n_boards
from ic_routing_board_generation.interface.board_generator_interface_numpy import \
    BoardName

if __name__ == '__main__':

    # Option 1: run for all generators
    grid_params = [(10, 10, 5)]
    board_list = board_generation_params_from_grid_params(grid_params)

    # Option 2:specify board generation parameters
    # board_list = [
    #     BoardGenerationParameters(rows=10, columns=10, number_of_wires=5,
    #                               generator_type=BoardName.RANDOM_SEED),
    #     # BoardGenerationParameters(rows=10, columns=10, number_of_wires=5,
    #     #                           generator_type=BoardName.BFS_SHORTEST),
    #     # BoardGenerationParameters(rows=10, columns=10, number_of_wires=5,
    #     #                           generator_type=BoardName.JAX_SEED_EXTENSION),
    #     # BoardGenerationParameters(rows=10, columns=10, number_of_wires=5,
    #     #                           generator_type=BoardName.LSYSTEMS),
    #     # BoardGenerationParameters(rows=10, columns=10, number_of_wires=5,
    #     #                           generator_type=BoardName.WFC),
    # ]
    evaluate_generator_outputs_averaged_on_n_boards(board_list, number_of_boards=5000, plot_individually=False)