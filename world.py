#!/python

import os
import cv2

import numpy as np

import matplotlib.pyplot as plt

from ai_economist import foundation
import matplotlib.animation as animation

from IPython import display
import plotting

# Define the configuration of the environment that will be built

env_config = {
    # ===== SCENARIO CLASS =====
    # Which Scenario class to use: the class's name in the Scenario Registry (foundation.scenarios).
    # The environment object will be an instance of the Scenario class.
    'scenario_name': 'layout_from_file/simple_wood_and_stone',

    # ===== COMPONENTS =====
    # Which components to use (specified as list of ("component_name", {component_kwargs}) tuples).
    #   "component_name" refers to the Component class's name in the Component Registry (foundation.components)
    #   {component_kwargs} is a dictionary of kwargs passed to the Component class
    # The order in which components reset, step, and generate obs follows their listed order below.
    'components': [
        # (1) Building houses
        ('Build', {'skill_dist': "pareto", 'payment_max_skill_multiplier': 3}),
        # (2) Trading collectible resources
        ('ContinuousDoubleAuction', {'max_num_orders': 5}),
        # (3) Movement and resource collection
        ('Gather', {}),
    ],

    # ===== SCENARIO CLASS ARGUMENTS =====
    # (optional) kwargs that are added by the Scenario class (i.e. not defined in BaseEnvironment)
    #'env_layout_file': 'quadrant_25x25_20each_30clump.txt',
    'starting_agent_coin': 10,
    'fixed_four_skill_and_loc': True,

    # ===== STANDARD ARGUMENTS ======
    # kwargs that are used by every Scenario class (i.e. defined in BaseEnvironment)
    'n_agents': 4,          # Number of non-planner agents (must be > 1)
    'world_size': [25, 25], # [Height, Width] of the env world
    'episode_length': 20, # Number of timesteps per episode

    # In multi-action-mode, the policy selects an action for each action subspace (defined in component code).
    # Otherwise, the policy selects only 1 action.
    'multi_action_mode_agents': False,
    'multi_action_mode_planner': True,

    # When flattening observations, concatenate scalar & vector observations before output.
    # Otherwise, return observations with minimal processing.
    'flatten_observations': False,
    # When Flattening masks, concatenate each action subspace mask into a single array.
    # Note: flatten_masks = True is required for masking action logits in the code below.
    'flatten_masks': True,
}


env = foundation.make_env_instance(**env_config)

env.get_agent(0)


# Note: The code for sampling actions (this cell),
# and playing an episode (below) are general.
# That is, it doesn't depend on
# the Scenario and Component classes used in the environment!
def sample_random_action(agent, mask):
    """Sample random UNMASKED action(s) for agent."""

    # Return a list of actions: 1 for each action subspace
    if agent.multi_action_mode:
        split_masks = np.split(mask, agent.action_spaces.cumsum()[:-1])
        return [
            np.random.choice(np.arange(len(m_)), p=m_/m_.sum())
            for m_ in split_masks
        ]

    # Return a single action
    else:
        return np.random.choice(
            np.arange(agent.action_spaces),
            p=mask/mask.sum())


def sample_random_actions(env, obs):
    """Samples random UNMASKED actions for each agent in obs."""

    actions = {
        a_idx: sample_random_action(env.get_agent(a_idx), a_obs['action_mask'])
        for a_idx, a_obs in obs.items()
    }

    return actions


obs = env.reset()


actions = sample_random_actions(env, obs)
obs, rew, done, info = env.step(actions)




for key, val in obs['0'].items(): 
    print("{:50} {}".format(key, type(val)))



def do_plot(ID, env, ax, fig):
    """Plots world state during episode sampling."""
    plotting.plot_env_state(ID, env, ax)
    ax.set_aspect('equal')
    display.display(fig)
    display.clear_output(wait=True)


def play_random_episode(env, plot_every=1, output_directory="Simulation", do_dense_logging=False):
    """Plays an episode with randomly sampled actions.

    Demonstrates gym-style API:
        obs                  <-- env.reset(...)         # Reset
        obs, rew, done, info <-- env.step(actions, ...) # Interaction loop

    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Reset
    obs = env.reset(force_dense_logging=do_dense_logging)

    Axes = []
    # Interaction loop (w/ plotting)
    for t in range(env.episode_length):
        actions = sample_random_actions(env, obs)
        obs, rew, done, info = env.step(actions)

        if ((t+1) % plot_every) == 0:
            a = do_plot(t + 1, env, ax, fig)
            Axes.append(a)

    if ((t+1) % plot_every) != 0:
        a = do_plot(t + 1, env, ax, fig)
        Axes.append(a)


def AnimateFromImages(Dir):
    """Create movie from plot images."""

    Files = [os.path.join(Dir, p) for p in sorted(os.listdir(Dir))]

    video_name = "simulation.mp4"
    if Files:
        frame = cv2.imread(Files[0])
        height, width, layers = frame.shape

        video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'XVID'), 3, (width, height))

        for File in Files:
            video.write(cv2.imread(File))

        cv2.destroyAllWindows()
        video.release()


def Animate(Axs):
    """
    To animate using matplotlib's internal machinery
    will require some modifications on the simulation code,
    so this is TBD.
    """
    im_ani = animation.ArtistAnimation(a,
                                       ims,
                                       interval=50,
                                       repeat_delay=3000,
                                       blit=True)
    # To save this second animation with some metadata, use the following command:
    im_ani.save('im.mp4', metadata={'artist': 'G0D'})


def main():
    output_directory = "Simulation"
    play_random_episode(env, plot_every=1)
    AnimateFromImages(output_directory)


if __name__ == "__main__":
    main()
