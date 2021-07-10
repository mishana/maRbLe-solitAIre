if __name__ == '__main__':
    import os

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    from env import MarbleSolitaireEnv, MarbleAction
    import matplotlib.pyplot as plt

    env = MarbleSolitaireEnv(init_fig=True, interactive_plot=True)
    env.render()
    plt.show()

    from stable_baselines3.common.policies import ActorCriticPolicy
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3 import PPO

    model = PPO(ActorCriticPolicy, env, verbose=1)
    model.learn(total_timesteps=50000)
    pass

    obs = env.reset()
    for i in range(100):
        action, _states = model.predict(obs)
        # print(f'i:{action[0]}, j:{action[1]}, a:{MarbleAction(action[2]).name}')
        if env._is_valid_action(*action):
            print(f'i,j: {env.idx_to_i_j(action[0])}, a:{MarbleAction(action[1]).name}')
            env.render(action=action, show_action=True)
            plt.show()
        obs, rewards, dones, info = env.step(action)
        if env._is_valid_action(*action):
            env.render()
            plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
