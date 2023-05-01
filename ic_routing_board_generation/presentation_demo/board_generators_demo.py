import argparse
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pickle
import datetime
from hydra import compose, initialize



from ic_routing_board_generation.interface.board_generator_interface_numpy import BoardGenerator, BoardName

from jumanji.training.setup_train import setup_agent, setup_env
from jumanji.training.utils import first_from_device
from jumanji.env import Environment
from jumanji.environments.routing.connector.env import Connector
from jumanji.environments.routing.connector.types import Agent, Observation, State
from jumanji.types import TimeStep, restart, termination, transition


from jumanji.environments.routing.connector.viewer import ConnectorViewer


def board_to_env(board: jnp.ndarray) -> Environment:
    """Converts a board to a jumanji environment"""
    # get agent
    board_env = Connector()
    state = board
    action_mask = jax.vmap(board_env._get_action_mask, (0, None))(
        state.agents, state.grid
    )
    observation = Observation(
        grid=board_env._obs_from_grid(state.grid),
        action_mask=action_mask,
        step_count=state.step_count,
    )
    extras = board_env._get_extras(state)
    timestep = restart(
        observation=observation, extras=extras, shape=(board_env.num_agents,)
    )
    return state, timestep
    



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
       training_board= initial_board.return_training_board()
    elif args.board_type == "offline_parallel_rw":
       training_board= initial_board.generate_board(key)
    elif args.board_type == "bfs_base":
        training_board = initial_board.generate_boards(1)
    elif args.board_type == "bfs_min_bend" or args.board_type == "bfs_fifo" or args.board_type == "bfs_short" or args.board_type == "bfs_long":
        training_board = initial_board.return_training_board()
    elif args.board_type == "lsystems_standard":
        training_board = initial_board.return_training_board()
    elif args.board_type == "wfc":
        training_board = initial_board.return_training_board()
    elif args.board_type == "numberlink":
        training_board = initial_board.return_training_board()

    print( training_board )


    # Render the board using jumanji's method
    viewer = ConnectorViewer("Ben", args.num_agents)
    # Make sure the board is a jnp array
    training_board = jax.numpy.array(training_board)
    #viewer.render(training_board)
    #plt.show()

    # TODO: add argument for showing a trained agent on the board
    
    # Show a trained agent trying to solve the board
    # TODO: change file path once rebased
    file = "examples/trained_agent_10x10_5_uniform/19-27-36/training_state_10x10_5_uniform"
    with open(file,"rb") as f:
        training_state = pickle.load(f)
    
    with initialize(version_base=None, config_path="../../jumanji/training/configs"):
        cfg = compose(config_name="config.yaml", overrides=["env=connector", "agent=a2c"])

    params = first_from_device(training_state.params_state.params)
    #print(params)
    env = setup_env(cfg).unwrapped
    #print(env)
    agent = setup_agent(cfg, env)
    #print(agent)
    policy = agent.make_policy(params.actor, stochastic = False)
    #print(params.num_agents)


    step_fn = env.step  # Speed up env.step
    GRID_SIZE = 10

    states = []
    key = jax.random.PRNGKey(cfg.seed)

    connections = []
    key, reset_key = jax.random.split(key)
    state, timestep = board_to_env(training_board)
    
    # Set the initial state
    # state = training_board


    while not timestep.last():
        print("gooseeyyyy")
        key, action_key = jax.random.split(key)
        observation = jax.tree_util.tree_map(lambda x: x[None], timestep.observation)
        # Two implementations for calling the policy, about equivalent speed
        action, _ = policy(observation, action_key)
        #action, _ = jax.jit(policy)(observation, action_key)
        # Three implementations for updating the state/timestep.  The third is much faster.
        #state, timestep = jax.jit(env.step)(state, action.squeeze(axis=0)) # original jit = 0.32, 52sec/10
        state, timestep = env.step(state, action.squeeze(axis=0)) # no jit = 0.13, 26sec/10
        #state, timestep = step_fn(state, action.squeeze(axis=0)) # jit function = 0.003 5 sec/10, 49sec/100d
        states.append(state.grid)
    
    # Render the animation
    animation = viewer.animate(states)
    # Animation is a matplotlib.animation.FuncAnimation object
    # save the animation
    animation.save('animation.mp4')

    import cv2

    from matplotlib.animation import writers

    if not writers.is_available('ffmpeg'):
        print("Error: FFmpeg is not available. Please install it to save the animation.")
    else:
        # Save the animation
        animation.save('animation.mp4', writer='ffmpeg')

        import cv2

        def play_video(filename):
            cap = cv2.VideoCapture(filename)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                cv2.imshow('Video', frame)

                # Wait for a key press
                key = cv2.waitKey(0) & 0xFF

                # Press 'q' to exit the video window
                if key == ord('q'):
                    break
                # Press spacebar to advance to the next frame
                elif key == ord(' '):
                    continue

            cap.release()
            cv2.destroyAllWindows()

        video_filename = 'animation.mp4'
        play_video(video_filename)
    

    
