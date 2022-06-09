import os
import torch
from torch.multiprocessing import set_start_method
from agents.ppo import Agent
from utils.atari_wrapper import create_env
from models.ppo_atari import PolicyValueModel

if __name__ == '__main__':    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device: ', device)
    set_start_method('spawn')
    torch.multiprocessing.set_sharing_strategy('file_system')

    name = 'ppo_support_value'

    env_name = 'BreakoutNoFrameskip-v4'
    env_params = dict(env_name = env_name, y1 = 30, y2 = 200, x1 = 5, x2 = 155,
                      denominator = 148.0, skip = 4, penalty = True,
                      steps_after_reset = 0)

    count_of_iterations = 10000
    count_of_processes = 2
    count_of_envs = 16
    count_of_steps = 128
    count_of_epochs = 4
    batch_size = 512
    lr = 2.5e-4
    value_support_size = 7

    path = 'results/breakout/' + name + '/'
    
    if os.path.isdir(path):
        print('directory has already existed')
    else:
        os.mkdir(path)
        print('new directory has been created')
    

    env = create_env(**env_params)
    dim1, dim2, dim3 = env.reset().shape
    count_of_actions = env.action_space.n

    model = PolicyValueModel(count_of_actions, dim2 * dim3, 
                             value_support_size = value_support_size)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    agent = Agent(model, optimizer, coef_value = 0.5, 
                  value_support_size = value_support_size,
                  device = device, path = path)
    agent.train(env_params, create_env, count_of_actions,
              count_of_iterations = count_of_iterations,
              count_of_processes = count_of_processes,
              count_of_envs = count_of_envs,
              count_of_steps = count_of_steps,
              count_of_epochs = count_of_epochs,
              batch_size = batch_size, input_dim = (dim1, dim2, dim3))
