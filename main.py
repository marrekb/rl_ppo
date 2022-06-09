import os
import argparse
import torch
from torch.multiprocessing import set_start_method
from utils.atari_wrapper import create_env

if __name__ == '__main__':    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device: ', device)
    set_start_method('spawn')
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-e', '--environment', default = 'pong',
                        type = str, help = 'pong, breakout, pacman, montezuma')
    parser.add_argument('-n', '--name', default = 'ppo',
                        type = str, help = 'plot name')
    parser.add_argument('-ci', '--citerations', default = 10000,
                        type = int, help = 'count of iterations')
    parser.add_argument('-cp', '--cprocesses', default = 2,
                        type = int, help = 'count of processes')
    parser.add_argument('-ce', '--cenvs', default = 16,
                        type = int, help = 'count of envs per process')
    parser.add_argument('-cs', '--csteps', default = 128,
                        type = int, help = 'count of steps per iteration')
    parser.add_argument('-cep', '--cepochs', default = 4,
                        type = int, help = 'count of epochs per iteration')
    parser.add_argument('-bs', '--batchsize', default = 512,
                        type = int, help = 'batch size')
    parser.add_argument('-lr', '--lrate', default = 2.5e-4,
                        type = float, help = 'learning rate')
    parser.add_argument('-vs', '--valuesupport', default = 1,
                        type = int, help = 'value support')
    parser.add_argument('-cv', '--coefvalue', default = 0.5,
                        type = float, help = 'coefficient of value loss')
    parser.add_argument('-cen', '--coefentropy', default = 0.5,
                        type = float, help = 'coefficient of entropy loss')
    parser.add_argument('-g', '--gamma', default = 0.997,
                        type = float, help = 'discount factor')
                  
    parser.add_argument('-rnd', '--rnd', default = False,
                        help = 'use of rnd module')                   
    parser.add_argument('-lrd', '--lraternd', default = 2.5e-4,
                        type = float, help = 'learning rate of rnd module') 
    parser.add_argument('-ivs', '--intvaluesupport', default = 1,
                        type = int, help = 'internal value support')
    parser.add_argument('-ig', '--intgamma', default = 0.99,
                        type = float, help = 'discount factor of internal rewards')                
                        
    args = parser.parse_args()
    
    if args.environment == 'pong':
        env_params = dict(env_name = 'PongNoFrameskip-v4', y1 = 33, y2 = 195, x1 = 0,
                          x2 = 160, denominator = 236.0, skip = 4, penalty = False)
    elif args.environment == 'breakout':
        env_params = dict(env_name = 'BreakoutNoFrameskip-v4', y1 = 30, y2 = 200, 
                          x1 = 5, x2 = 155, denominator = 148.0, skip = 4, 
                          penalty = True, steps_after_reset = 0)
    elif args.environment == 'pacman':
        env_params = dict(env_name = 'MsPacmanNoFrameskip-v4', y1 = 1, y2 = 171, x1 = 0, x2 = 160,
                          denominator = 214.0, skip = 4, penalty = True,
                          steps_after_reset = 65)
    elif args.environment == 'montezuma':
        env_params = dict(env_name = 'MontezumaRevengeNoFrameskip-v4', y1 = 0, y2 = 210, x1 = 0,
                          x2 = 160, denominator = 255.0, skip = 4, penalty = True)
    else:
        env_params = dict(env_name = args.environment, y1 = 0, y2 = 210, x1 = 0,
                          x2 = 160, denominator = 255.0, skip = 4, penalty = True)
        
        path = 'results/' + args.environment + '/'
        if not os.path.isdir(path):
            os.mkdir(path)
        
    path = 'results/' + args.environment + '/' + args.name + '/'
    
    if os.path.isdir(path):
        print('directory has already existed')
    else:
        os.mkdir(path)
        print('new directory has been created')
    
    env = create_env(**env_params)
    dim1, dim2, dim3 = env.reset().shape
    count_of_actions = env.action_space.n
    
    if not args.rnd:
        from models.ppo_atari import PolicyValueModel
        from agents.ppo import Agent
        
        model = PolicyValueModel(count_of_actions, dim2 * dim3, 
                                 value_support_size = args.valuesupport)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lrate)
    
        agent = Agent(model, optimizer, coef_value = args.coefvalue, 
                      gamma = args.gamma, coef_entropy = args.coefentropy,
                      value_support_size = args.valuesupport,
                      device = device, path = path)
    else:
        from models.ppo_atari_int import PolicyValueModel
        from models.rnd_atari import RND_Model
        from agents.ppo_rnd import Agent
        
        model = PolicyValueModel(count_of_actions, dim2 * dim3, 
                                 ext_value_support_size = args.valuesupport,
                                 int_value_support_size = args.intvaluesupport)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lrate)
    
        rnd_model = RND_Model(dim2 * dim3)
        rnd_model.to(device)
        rnd_optimizer =  torch.optim.Adam(rnd_model.parameters(), lr = args.lraternd)
    
        agent = Agent(model, optimizer, rnd_model, rnd_optimizer,
                      ext_gamma = args.gamma, int_gamma = args.intgamma,
                      coef_value = args.coefvalue,
                      coef_entropy = args.coefentropy,
                      device = device, path = path,
                      ext_value_support_size = args.valuesupport, 
                      int_value_support_size = args.intvaluesupport)
    
    agent.train(env_params, create_env, count_of_actions,
                count_of_iterations = args.citerations,
                count_of_processes = args.cprocesses,
                count_of_envs = args.cenvs,
                count_of_steps = args.csteps,
                count_of_epochs = args.cepochs,
                batch_size = args.batchsize, 
                input_dim = (dim1, dim2, dim3))
