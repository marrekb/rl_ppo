import numpy as np
import torch
import torch.nn.functional as F
from torch.multiprocessing import Process, Pipe
from utils.stats import MovingAverageScore, write_to_file
from utils.head_support import support_to_scalar, scalar_to_support

def worker(connection, env_params, env_func, count_of_iterations, count_of_envs,
           count_of_steps, ext_gamma, int_gamma, gae_lambda):

    envs = [env_func(**env_params) for _ in range(count_of_envs)]
    observations = torch.stack([torch.from_numpy(env.reset()) for env in envs])
    game_score = np.zeros(count_of_envs)

    mem_log_probs = torch.zeros((count_of_steps, count_of_envs, 1))
    mem_actions = torch.zeros((count_of_steps, count_of_envs, 1), dtype = torch.long)
    mem_ext_values = torch.zeros((count_of_steps + 1, count_of_envs, 1))
    mem_int_values = torch.zeros((count_of_steps + 1, count_of_envs, 1))
    mem_ext_rewards = torch.zeros((count_of_steps, count_of_envs, 1))
    mem_int_rewards = torch.zeros((count_of_steps, count_of_envs, 1))

    for iteration in range(count_of_iterations):
        mem_non_terminals = torch.ones((count_of_steps, count_of_envs, 1))
        scores = []

        for step in range(count_of_steps):
            connection.send(observations.float())
            logits, ext_values, int_values, int_rewards = connection.recv()
            probs = F.softmax(logits, dim = -1)
            actions = probs.multinomial(num_samples = 1)
            log_probs = F.log_softmax(logits, dim = -1).gather(1, actions)

            mem_log_probs[step] = log_probs
            mem_actions[step] = actions

            mem_ext_values[step] = ext_values
            mem_int_values[step] = int_values
            if step > 0:
                mem_int_rewards[step - 1] = int_rewards

            for idx in range(count_of_envs):
                observation, ext_reward, terminal, _ = envs[idx].step(actions[idx, 0].item())
                mem_ext_rewards[step, idx, 0] = ext_reward
                game_score[idx] += ext_reward
                if ext_reward < 0:
                    mem_non_terminals[step, idx, 0] = 0

                if terminal:
                    mem_non_terminals[step, idx, 0] = 0
                    scores.append(game_score[idx])
                    game_score[idx] = 0
                    observation = envs[idx].reset()
                observations[idx] = torch.from_numpy(observation)

        connection.send(observations.float())

        ext_values, int_values, int_rewards = connection.recv()
        mem_int_rewards[step] = int_rewards
        step += 1
        mem_ext_values[step] = ext_values
        mem_int_values[step] = int_values

        mem_ext_rewards = torch.clamp(mem_ext_rewards, -1.0, 1.0)
        mem_int_rewards = torch.clamp(mem_int_rewards, -1.0, 1.0)

        ext_advantages = torch.zeros((count_of_steps, count_of_envs, 1))
        int_advantages = torch.zeros((count_of_steps, count_of_envs, 1))
        ext_values = torch.zeros((count_of_steps, count_of_envs, 1))
        int_values = torch.zeros((count_of_steps, count_of_envs, 1))
        ext_gae = torch.zeros((count_of_envs, 1))
        int_gae = torch.zeros((count_of_envs, 1))

        for step in reversed(range(count_of_steps)):
            ext_delta = mem_ext_rewards[step] + ext_gamma * mem_ext_values[step + 1] * mem_non_terminals[step] \
                    - mem_ext_values[step]
            ext_gae = ext_delta + ext_gamma * gae_lambda * ext_gae * mem_non_terminals[step]
            ext_values[step] = ext_gae + mem_ext_values[step]
            ext_advantages[step] = ext_gae.clone()

            int_delta = mem_int_rewards[step] + int_gamma * mem_int_values[step + 1] * mem_non_terminals[step] \
                    - mem_int_values[step]
            int_gae = int_delta + int_gamma * gae_lambda * int_gae * mem_non_terminals[step]
            int_values[step] = int_gae + mem_int_values[step]
            int_advantages[step] = int_gae.clone()

        connection.send([mem_log_probs, mem_actions, ext_values, int_values,
                         ext_advantages, int_advantages, scores])
    connection.recv()
    connection.close()

class Agent:
    def __init__(self, model, optimizer, rnd_model, rnd_optimizer,
                 ext_gamma = 0.997, int_gamma = 0.99, epsilon = 0.1,
                 coef_value = 0.5, coef_entropy = 0.001, gae_lambda = 0.95,
                 coef_ext = 2.0, coef_int = 1.0,
                 path = 'results/', device = 'cpu',
                 ext_value_support_size = 1, int_value_support_size = 1):

        self.model = model
        self.optimizer = optimizer

        self.rnd_model = rnd_model
        self.rnd_optimizer = rnd_optimizer

        self.ext_gamma = ext_gamma
        self.int_gamma = int_gamma
        self.gae_lambda = gae_lambda

        self.coef_value = coef_value
        self.coef_entropy = coef_entropy
        self.coef_ext = coef_ext
        self.coef_int = coef_int


        self.lower_bound = 1 - epsilon
        self.upper_bound = 1 + epsilon

        self.path = path
        self.device = device
        
        self.ext_value_support_size = ext_value_support_size
        self.support_to_value = ext_value_support_size > 1
        self.ext_value_support_interval = ext_value_support_size * 2 + 1
        
        self.int_value_support_size = int_value_support_size
        self.int_value_support_interval = int_value_support_size * 2 + 1
        
        int_support = int_value_support_size > 1
        if self.support_to_value != int_support:
            raise Exception('Value heads are trained in different approaches. ' +\
                            'Please, train both value heads as either regression or categorical task.')

    def train(self, env_params, env_func, count_of_actions,
              count_of_iterations = 10000, count_of_processes = 2,
              count_of_envs = 16, count_of_steps = 128, count_of_epochs = 4,
              batch_size = 512, input_dim = (4, 96, 96)):

        print('Training is starting')

        logs_score = 'iteration,episode,avg_score,best_avg_score,best_score'
        logs_loss = 'iteration,episode,policy,ext_value,int_value,entropy,rnd'

        score = MovingAverageScore()
        buffer_size = count_of_processes * count_of_envs * count_of_steps
        batches_per_iteration = count_of_epochs * buffer_size / batch_size

        dim0, dim1, dim2 = input_dim
        obs_mean = torch.zeros((1, dim1, dim2), device = self.device)

        processes, connections = [], []
        for _ in range(count_of_processes):
            parr_connection, child_connection = Pipe()
            process = Process(target = worker, args = (
                child_connection, env_params, env_func, count_of_iterations,
                count_of_envs, count_of_steps, self.ext_gamma, self.int_gamma,
                self.gae_lambda))
            connections.append(parr_connection)
            processes.append(process)
            process.start()

        mem_dim = (count_of_processes, count_of_steps, count_of_envs)
        mem_observations = torch.zeros((mem_dim + input_dim), device = self.device)
        mem_actions = torch.zeros((*mem_dim, 1), device = self.device, dtype = torch.long)
        mem_log_probs = torch.zeros((*mem_dim , 1), device = self.device)
        
        if self.support_to_value:
            mem_ext_values = torch.zeros((*mem_dim, self.ext_value_support_interval), device = self.device)
            mem_int_values = torch.zeros((*mem_dim, self.int_value_support_interval), device = self.device)
        else:
            mem_ext_values = torch.zeros((*mem_dim, 1), device = self.device)
            mem_int_values = torch.zeros((*mem_dim, 1), device = self.device)
            
        mem_ext_advantages = torch.zeros((*mem_dim, 1), device = self.device)
        mem_int_advantages = torch.zeros((*mem_dim, 1), device = self.device)

        for iteration in range(count_of_iterations):
            for step in range(count_of_steps):
                observations = [conn.recv() for conn in connections]
                observations = torch.stack(observations).to(self.device)
                mem_observations[:, step] = observations
                observations = observations.view(-1, *input_dim)

                with torch.no_grad():
                    logits, ext_values, int_values = self.model(observations)

                    obs_norm = observations[:, 0] - obs_mean
                    rnd_pred, rnd_targ = self.rnd_model(obs_norm.view(-1, 1, dim1, dim2))
                    int_rewards = torch.sum((rnd_targ - rnd_pred)**2, dim = 1) / 2.0

                logits = logits.view(-1, count_of_envs, count_of_actions).cpu()
                
                if self.support_to_value:
                    ext_values = support_to_scalar(ext_values, self.ext_value_support_size).view(
                        -1, count_of_envs, 1).cpu()
                    int_values = support_to_scalar(int_values, self.int_value_support_size).view(
                        -1, count_of_envs, 1).cpu()
                else:
                    ext_values = ext_values.view(-1, count_of_envs, 1).cpu()
                    int_values = int_values.view(-1, count_of_envs, 1).cpu()
                        
                int_rewards = int_rewards.view(-1, count_of_envs, 1).cpu()

                for idx in range(count_of_processes):
                    connections[idx].send([logits[idx], ext_values[idx],
                                           int_values[idx], int_rewards[idx]])

            observations = [conn.recv() for conn in connections]
            observations = torch.stack(observations).to(self.device)
            observations = observations.view(-1, *input_dim)

            with torch.no_grad():
                _, ext_values, int_values = self.model(observations)
                obs_norm = observations[:, 0] - obs_mean
                rnd_pred, rnd_targ = self.rnd_model(obs_norm.view(-1, 1, dim1, dim2))
                int_rewards = torch.sum((rnd_targ - rnd_pred)**2, dim = 1) / 2.0

            if self.support_to_value:
                ext_values = support_to_scalar(ext_values, self.ext_value_support_size).view(
                    -1, count_of_envs, 1).cpu()
                int_values = support_to_scalar(int_values, self.int_value_support_size).view(
                    -1, count_of_envs, 1).cpu()
            else:
                ext_values = ext_values.view(-1, count_of_envs, 1).cpu()
                int_values = int_values.view(-1, count_of_envs, 1).cpu()
                
            int_rewards = int_rewards.view(-1, count_of_envs, 1).cpu()

            for idx in range(count_of_processes):
                connections[idx].send([ext_values[idx], int_values[idx],
                                       int_rewards[idx]])

            for idx in range(count_of_processes):
                log_probs, actions, ext_values, int_values, ext_advantages, int_advantages, scores \
                    = connections[idx].recv()
                mem_actions[idx] = actions.to(self.device)
                mem_log_probs[idx] = log_probs.to(self.device)
                
                if self.support_to_value:
                    mem_ext_values[idx] = scalar_to_support(
                        ext_values.to(self.device).view(-1, 1), 
                        self.ext_value_support_size
                    ).view(-1, count_of_envs, self.ext_value_support_interval)
                    mem_int_values[idx] = scalar_to_support(
                        int_values.to(self.device).view(-1, 1), 
                        self.int_value_support_size
                    ).view(-1, count_of_envs, self.int_value_support_interval)
                else:
                    mem_ext_values[idx] = ext_values.to(self.device)
                    mem_int_values[idx] = int_values.to(self.device)
                
                mem_ext_advantages[idx] = ext_advantages.to(self.device)
                mem_int_advantages[idx] = int_advantages.to(self.device)
                score.add(scores)

            avg_score, best_score = score.mean()
            print('iteration: ', iteration, '\taverage score: ', avg_score)
            if best_score:
                print('New best avg score has been achieved', avg_score)
                torch.save(self.model.state_dict(), self.path + 'model.pt')

            mem_observations = mem_observations.view(-1, *input_dim)
            mem_actions = mem_actions.view(-1, 1)
            mem_log_probs = mem_log_probs.view(-1, 1)
            
            if self.support_to_value:
                mem_ext_values = mem_ext_values.view(-1, self.ext_value_support_interval)
                mem_int_values = mem_int_values.view(-1, self.int_value_support_interval)
            else:
                mem_ext_values = mem_ext_values.view(-1, 1)
                mem_int_values = mem_int_values.view(-1, 1)
                
            mem_ext_advantages = mem_ext_advantages.view(-1, 1)
            mem_int_advantages = mem_int_advantages.view(-1, 1)
            mem_advantages = self.coef_ext * mem_ext_advantages \
                             + self.coef_int * mem_int_advantages
            mem_advantages = (mem_advantages - mem_advantages.mean()) / (mem_advantages.std() + 1e-5)

            obs_mean = (obs_mean * iteration + mem_observations[:, 0].mean(0)) / (iteration + 1)

            s_policy, s_ext_value, s_int_value, s_entropy = 0.0, 0.0, 0.0, 0.0
            s_rnd = 0.0

            for epoch in range(count_of_epochs):
                perm = torch.randperm(buffer_size, device = self.device).view(-1, batch_size)
                for idx in perm:
                    logits, ext_values, int_values = self.model(mem_observations[idx])
                    probs = F.softmax(logits, dim = -1)
                    log_probs = F.log_softmax(logits, dim = -1)
                    new_log_probs = log_probs.gather(1, mem_actions[idx])

                    entropy_loss = (log_probs * probs).sum(1, keepdim=True).mean()
                    
                    if self.support_to_value:
                        ext_values_log_probs = F.log_softmax(ext_values, dim = -1)
                        ext_value_loss = torch.sum(- mem_ext_values[idx] * ext_values_log_probs, dim = 1).mean()
                        
                        int_values_log_probs = F.log_softmax(int_values, dim = -1)
                        int_value_loss = torch.sum(- mem_int_values[idx] * int_values_log_probs, dim = 1).mean()
                    else:
                        ext_value_loss = F.mse_loss(ext_values, mem_ext_values[idx])
                        int_value_loss = F.mse_loss(int_values, mem_int_values[idx])
                    
                    value_loss = ext_value_loss + int_value_loss

                    ratio = torch.exp(new_log_probs - mem_log_probs[idx])
                    surr_policy = ratio * mem_advantages[idx]
                    surr_clip = torch.clamp(ratio, self.lower_bound, self.upper_bound) \
                                * mem_advantages[idx]
                    policy_loss = - torch.min(surr_policy, surr_clip).mean()

                    s_policy += policy_loss.item()
                    s_ext_value += ext_value_loss.item()
                    s_int_value += int_value_loss.item()
                    s_entropy += entropy_loss.item()

                    self.optimizer.zero_grad()
                    loss = policy_loss + self.coef_value * value_loss \
                            + self.coef_entropy * entropy_loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                    self.optimizer.step()

                    self.rnd_optimizer.zero_grad()
                    obs_norm = (mem_observations[idx, 0] - obs_mean)
                    obs_norm = obs_norm.view(-1, 1, dim1, dim2)
                    rnd_pred, rnd_targ = self.rnd_model(obs_norm)
                    rnd_loss = F.mse_loss(rnd_pred, rnd_targ)
                    rnd_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.rnd_model.parameters(), 0.5)
                    self.rnd_optimizer.step()

                    s_rnd += loss.item()

            mem_observations = mem_observations.view((mem_dim + input_dim))
            mem_actions = mem_actions.view((*mem_dim, 1))
            mem_log_probs = mem_log_probs.view((*mem_dim, 1))
            
            if self.support_to_value:
                mem_ext_values = mem_ext_values.view((*mem_dim, self.ext_value_support_interval))
                mem_int_values = mem_int_values.view((*mem_dim, self.int_value_support_interval))
            else:
                mem_ext_values = mem_ext_values.view((*mem_dim, 1))
                mem_int_values = mem_int_values.view((*mem_dim, 1))
                
            mem_ext_advantages = mem_ext_advantages.view((*mem_dim, 1))
            mem_int_advantages = mem_int_advantages.view((*mem_dim, 1))

            logs_score += '\n' + str(iteration) + ',' \
                         + str(score.get_count_of_episodes()) + ',' \
                         + str(avg_score) + ',' \
                         + str(score.get_best_avg_score()) + ',' \
                         + str(score.get_best_score())

            logs_loss += '\n' + str(iteration) + ',' \
                         + str(avg_score) + ',' \
                         + str(s_policy / batches_per_iteration) + ',' \
                         + str(s_ext_value / batches_per_iteration) + ',' \
                         + str(s_int_value / batches_per_iteration) + ',' \
                         + str(s_entropy / batches_per_iteration) + ',' \
                         + str(s_rnd / batches_per_iteration)

            if iteration % 10 == 0:
                write_to_file(logs_score, self.path + 'logs_score.csv')
                write_to_file(logs_loss, self.path + 'logs_loss.csv')
        print('Training has ended, best avg score is ', score.get_best_avg_score())

        for connection in connections:
            connection.send(1)
        for process in processes:
            process.join()
