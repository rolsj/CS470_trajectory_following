import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from agents.utils.create_env import EnvFactorySimpleFollowerAviary
from agents.utils.parse_configuration import Configuration


def run_train(config: Configuration, env_factory: EnvFactorySimpleFollowerAviary):

    # CONFIG ##################################################

    train_env = env_factory.get_train_env()
    eval_env = env_factory.get_eval_env()

    # #########################################################

    # SETUP ###################################################

    # model
    model = PPO('MlpPolicy',    
                train_env,
                tensorboard_log=config.output_path_location+'/tb/',
                verbose=1)

    # callbacks
    callback_on_best = StopTrainingOnRewardThreshold(
        reward_threshold=config.target_reward,
        verbose=1
    )
    eval_callback = EvalCallback(eval_env,
                                 callback_on_new_best=callback_on_best,
                                 verbose=1,
                                 best_model_save_path=config.output_path_location+'/',
                                 log_path=config.output_path_location+'/',
                                 eval_freq=int(1000),
                                 deterministic=True,
                                 render=False)
    
    # ##########################################################

    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)
    print('[INFO] Number of timesteps:', config.n_timesteps)

    # TRAIN ####################################################

    # fit
    model.learn(total_timesteps=config.n_timesteps,
                callback=eval_callback,
                log_interval=100)
    
    # save model
    model.save(config.output_path_location+'/final_model.zip')
    print(config.output_path_location)

    # print training progression 
    with np.load(config.output_path_location+'/evaluations.npz') as data:
        for j in range(data['timesteps'].shape[0]):
            print(str(data['timesteps'][j])+","+str(data['results'][j][0]))
    
    # ##########################################################



