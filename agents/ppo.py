import numpy as np
import torch
import torch.nn.functional as F
from torch.multiprocessing import Process, Pipe
from utils.stats import MovingAverageScore, write_to_file
from utils.head_support import support_to_scalar, scalar_to_support

def worker(connection, env_params, env_func, count_of_iterations, count_of_envs,
           count_of_steps, gamma, gae_lambda):

    envs = [env_func(**env_params) for _ in range(count_of_envs)]
    observations = torch.stack([torch.from_numpy(env.reset()) for env in envs])
    game_score = np.zeros(count_of_envs)

    mem_log_probs = torch.zeros((count_of_steps, count_of_envs, 1))
    mem_actions = torch.zeros((count_of_steps, count_of_envs, 1), dtype = torch.long)
    mem_values = torch.zeros((count_of_steps + 1, count_of_envs, 1))
    mem_rewards = torch.zeros((count_of_steps, count_of_envs, 1))

    for iteration in range(count_of_iterations):
        mem_non_terminals = torch.ones((count_of_steps, count_of_envs, 1))
        scores = []
        for step in range(count_of_steps):
            connection.send(observations.float())
            logits, values = connection.recv()
            probs = F.softmax(logits, dim = -1)
            actions = probs.multinomial(num_samples = 1)
            log_probs = F.log_softmax(logits, dim = -1).gather(1, actions)

            mem_log_probs[step] = log_probs
            mem_actions[step] = actions
            mem_values[step] = values


            for idx in range(count_of_envs):
                observation, reward, terminal, _ = envs[idx].step(actions[idx, 0].item())
                mem_rewards[step, idx, 0] = reward
                game_score[idx] += reward
                if reward < 0:
                    mem_non_terminals[step, idx, 0] = 0

                if terminal:
                    mem_non_terminals[step, idx, 0] = 0
                    scores.append(game_score[idx])
                    game_score[idx] = 0
                    observation = envs[idx].reset()
                observations[idx] = torch.from_numpy(observation)

        connection.send(observations.float())
        mem_values[step + 1] = connection.recv()
        mem_rewards = torch.clamp(mem_rewards, -1.0, 1.0)
        advantages = torch.zeros((count_of_steps, count_of_envs, 1))
        values = torch.zeros((count_of_steps, count_of_envs, 1))
        t_gae = torch.zeros((count_of_envs, 1))

        for step in reversed(range(count_of_steps)):
            delta = mem_rewards[step] + gamma * mem_values[step + 1] * mem_non_terminals[step] \
                    - mem_values[step]
            t_gae = delta + gamma * gae_lambda * t_gae * mem_non_terminals[step]
            values[step] = t_gae + mem_values[step]
            advantages[step] = t_gae.clone()

        connection.send([mem_log_probs, mem_actions, values, advantages, scores])
    connection.recv()
    connection.close()

class Agent:
    def __init__(self, model, optimizer, gamma = 0.997, epsilon = 0.1,
                 coef_value = 0.5, coef_entropy = 0.001, gae_lambda = 0.95,
                 path = 'results/', device = 'cpu',
                 value_support_size = 1):

        self.model = model
        self.optimizer = optimizer

        self.gamma = gamma
        self.coef_value = coef_value
        self.coef_entropy = coef_entropy
        self.gae_lambda = gae_lambda

        self.lower_bound = 1 - epsilon
        self.upper_bound = 1 + epsilon

        self.path = path
        self.device = device
        
        self.value_support_size = value_support_size
        self.support_to_value = value_support_size > 1
        self.value_support_interval = value_support_size * 2 + 1

    def train(self, env_params, env_func, count_of_actions,
              count_of_iterations = 10000, count_of_processes = 2,
              count_of_envs = 16, count_of_steps = 128, count_of_epochs = 4,
              batch_size = 512, input_dim = (4, 96, 96)):

        print('Training is starting')

        logs_score = 'iteration,episode,avg_score,best_avg_score,best_score'
        logs_loss = 'iteration,episode,policy,value,entropy'

        score = MovingAverageScore()
        buffer_size = count_of_processes * count_of_envs * count_of_steps
        batches_per_iteration = count_of_epochs * buffer_size / batch_size

        processes, connections = [], []
        for _ in range(count_of_processes):
            parr_connection, child_connection = Pipe()
            process = Process(target = worker, args = (
                child_connection, env_params, env_func, count_of_iterations,
                count_of_envs, count_of_steps, self.gamma, self.gae_lambda))
            connections.append(parr_connection)
            processes.append(process)
            process.start()

        mem_dim = (count_of_processes, count_of_steps, count_of_envs)
        mem_observations = torch.zeros((mem_dim + input_dim), device = self.device)
        mem_actions = torch.zeros((*mem_dim, 1), device = self.device, dtype = torch.long)
        mem_log_probs = torch.zeros((*mem_dim , 1), device = self.device)
        if self.support_to_value:
            mem_values = torch.zeros((*mem_dim, self.value_support_interval), device = self.device)
        else:
            mem_values = torch.zeros((*mem_dim, 1), device = self.device)
        mem_advantages = torch.zeros((*mem_dim, 1), device = self.device)

        for iteration in range(count_of_iterations):
            for step in range(count_of_steps):
                observations = [conn.recv() for conn in connections]
                observations = torch.stack(observations).to(self.device)
                mem_observations[:, step] = observations

                with torch.no_grad():
                    logits, values = self.model(observations.view(-1, *input_dim))

                # If you selected actions in the main process, your iteration
                # would last about 0.5 seconds longer (measured on 2 processes)
                logits = logits.view(-1, count_of_envs, count_of_actions).cpu()
                
                if self.support_to_value:
                    values = support_to_scalar(values, self.value_support_size).view(
                        -1, count_of_envs, 1).cpu()
                else:
                    values = values.view(-1, count_of_envs, 1).cpu()

                for idx in range(count_of_processes):
                    connections[idx].send([logits[idx], values[idx]])

            observations = [conn.recv() for conn in connections]
            observations = torch.stack(observations).to(self.device)

            with torch.no_grad():
                _, values = self.model(observations.view(-1, *input_dim))

            if self.support_to_value:
                values = support_to_scalar(values, self.value_support_size).view(
                    -1, count_of_envs, 1).cpu()
            else:
                values = values.view(-1, count_of_envs, 1).cpu()
              
            for idx in range(count_of_processes):
                connections[idx].send(values[idx])

            for idx in range(count_of_processes):
                log_probs, actions, values, advantages, scores = connections[idx].recv()
                mem_actions[idx] = actions.to(self.device)
                mem_log_probs[idx] = log_probs.to(self.device)
                if self.support_to_value:
                    mem_values[idx] = scalar_to_support(
                        values.to(self.device).view(-1, 1), 
                        self.value_support_size
                    ).view(-1, count_of_envs, self.value_support_interval)
                else:
                    mem_values[idx] = values.to(self.device)
                mem_advantages[idx] = advantages.to(self.device)
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
                mem_values = mem_values.view(-1, self.value_support_interval)
            else:
                mem_values = mem_values.view(-1, 1)
            mem_advantages = mem_advantages.view(-1, 1)
            mem_advantages = (mem_advantages - mem_advantages.mean()) / (mem_advantages.std() + 1e-5)

            s_policy, s_value, s_entropy = 0.0, 0.0, 0.0

            for epoch in range(count_of_epochs):
                perm = torch.randperm(buffer_size, device = self.device).view(-1, batch_size)
                for idx in perm:
                    logits, values = self.model(mem_observations[idx])
                    probs = F.softmax(logits, dim = -1)
                    log_probs = F.log_softmax(logits, dim = -1)
                    new_log_probs = log_probs.gather(1, mem_actions[idx])

                    entropy_loss = (log_probs * probs).sum(1, keepdim=True).mean()
                    
                    if self.support_to_value:
                        values_log_probs = F.log_softmax(values, dim = -1)
                        value_loss = torch.sum(- mem_values[idx] * values_log_probs, dim = 1).mean()
                    else:
                        value_loss = F.mse_loss(values, mem_values[idx])

                    ratio = torch.exp(new_log_probs - mem_log_probs[idx])
                    surr_policy = ratio * mem_advantages[idx]
                    surr_clip = torch.clamp(ratio, self.lower_bound, self.upper_bound) \
                                * mem_advantages[idx]
                    policy_loss = - torch.min(surr_policy, surr_clip).mean()

                    s_policy += policy_loss.item()
                    s_value += value_loss.item()
                    s_entropy += entropy_loss.item()

                    self.optimizer.zero_grad()
                    loss = policy_loss + self.coef_value * value_loss \
                            + self.coef_entropy * entropy_loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                    self.optimizer.step()

            mem_observations = mem_observations.view((mem_dim + input_dim))
            mem_actions = mem_actions.view((*mem_dim, 1))
            mem_log_probs = mem_log_probs.view((*mem_dim, 1))
            if self.support_to_value:
                mem_values = mem_values.view((*mem_dim, self.value_support_interval))
            else:
                mem_values = mem_values.view((*mem_dim, 1))
            mem_advantages = mem_advantages.view((*mem_dim, 1))

            logs_score += '\n' + str(iteration) + ',' \
                         + str(score.get_count_of_episodes()) + ',' \
                         + str(avg_score) + ',' \
                         + str(score.get_best_avg_score()) + ',' \
                         + str(score.get_best_score())

            logs_loss += '\n' + str(iteration) + ',' \
                         + str(avg_score) + ',' \
                         + str(s_policy / batches_per_iteration) + ',' \
                         + str(s_value / batches_per_iteration) + ',' \
                         + str(s_entropy / batches_per_iteration)

            if iteration % 10 == 0:
                write_to_file(logs_score, self.path + 'logs_score.csv')
                write_to_file(logs_loss, self.path + 'logs_loss.csv')
        print('Training has ended, best avg score is ', score.get_best_avg_score())

        for connection in connections:
            connection.send(1)
        for process in processes:
            process.join()
