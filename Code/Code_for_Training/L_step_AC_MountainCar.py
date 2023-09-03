#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gym, os
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import json
import matplotlib.pyplot as plt
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# config_file = input("Please enter path to configuration file: ")
config_file = "./train_config.json"


with open(config_file) as json_file:
    config = json.load(json_file)

def video_callable(episode_number):
    return episode_number%config['recording_frequency'] == 0

env = gym.make(config['env'])
if config['record']:
    env = gym.wrappers.Monitor(env, config['recording_path'], force = True, video_callable=video_callable)

if config['episode_length'] is not None:
    env._max_episode_steps = config['episode_length']

if config['numpy_seed'] is not None:
    np.random.seed(config['numpy_seed'])

if config['environment_seed'] is not None:
    env.seed(config['environment_seed'])

if config['pytorch_seed'] is not None:
    torch.manual_seed(config['pytorch_seed'])

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
actor_h_layers_sizes = config['actor']['hidden_layer_neurons']
critic_h_layers_sizes = config['critic']['hidden_layer_neurons']
gamma = config['gamma']
lr_A = config['actor']['learning_rate']
lr_C = config['critic']['learning_rate']
load_A = config['actor']['load']
load_C = config['critic']['load']
iterations = config['iterations']

class Actor(nn.Module):
    def __init__(self, input_size, h_layers_sizes, output_size):
        super(Actor, self).__init__()
        self.input_size = input_size
        self.h_layers_sizes = h_layers_sizes
        self.all_layers_sizes = [input_size] + h_layers_sizes + [output_size]
        self.output_size = output_size
        self.linears = nn.ModuleList([nn.Linear(self.all_layers_sizes[i], self.all_layers_sizes[i+1], bias=False) for i in range(len(self.all_layers_sizes)-1)])

    def forward(self, state):
        output = torch.tanh(self.linears[0](state))
        for i in range(1,len(self.linears)-1):
            output = torch.tanh(self.linears[i](output))
        output = self.linears[-1](output)
        distribution = Categorical(F.softmax(output, dim=-1))
        return distribution


class Critic(nn.Module):
    def __init__(self, input_size, h_layers_sizes, output_size):
        super(Critic, self).__init__()
        self.input_size = input_size
        self.h_layers_sizes = h_layers_sizes
        self.all_layers_sizes = [input_size] + h_layers_sizes + [output_size]
        self.output_size = output_size
        self.linears = nn.ModuleList([nn.Linear(self.all_layers_sizes[i], self.all_layers_sizes[i+1]) for i in range(len(self.all_layers_sizes)-1)])

    def forward(self, state):
        value = F.relu(self.linears[0](state))
        for i in range(1,len(self.linears)-1):
            value = F.relu(self.linears[i](value))
        value = self.linears[-1](value)
        return value

def lr_scheduler(optimizerA, optimizerC, total_reward):
    for schedule in config['learning_rate_scheduler']['schedule']:
        if total_reward >= schedule[0][0] and total_reward < schedule[0][1]:
            optimizerA.param_groups[0]['lr'] = schedule[1]['lr_A']
            optimizerC.param_groups[0]['lr'] = schedule[1]['lr_C']

def run(actor, critic, n_iters, l,k):
    print("value of k and l:",k,l)
    optimizerA = optim.Adam(actor.parameters(), lr=lr_A)
    optimizerC = optim.Adam(critic.parameters(), lr=lr_C)
    running_total_reward = 0
    max_running_total_reward = -float('inf')
    reward_list = []
    cur_reward_list = []
    file1 = open("./log_curScore_avgScore.txt","w+")
    n_iters=max(400,int(n_iters/pow(2,k)))

    for iter in range(n_iters):
        state = env.reset()
        total_reward = 0
        state = torch.FloatTensor(state).to(device)

        for i in count():
            if config['render']:
                env.render()
            
            dist = actor(state)

            action = dist.sample()
            next_state, reward, done, _ = env.step(action.cpu().numpy())
            
        
            if next_state[0] > 0.1 and next_state[0] < 0.4:
                reward += 10
            elif next_state[0] >= 0.4 and next_state[0] < 0.6:
                reward += 20
            if done == True and i < config['episode_length']-1:
                reward += 1000
                
            
            total_reward += reward

            log_prob = dist.log_prob(action).unsqueeze(0)
            
            value = critic(state)

            next_state = torch.FloatTensor(next_state).to(device)
            next_value = critic(next_state)

            if done:
                running_total_reward = total_reward if running_total_reward == 0 else running_total_reward * 0.9 + total_reward * 0.1
                print('Iteration: {}, Current Total Reward: {}, Running Total Reward: {}'.format(iter, round(total_reward,2), round(running_total_reward,2)))
                file1.write("{}\t{}\t{}\t{}\t{}\n".format(k,l,iter, round(total_reward,2), round(running_total_reward,2)))
                
                reward_list.append(running_total_reward)
                cur_reward_list.append(total_reward)

                if max_running_total_reward <= running_total_reward:
                    torch.save(actor, config['actor']['final_save_path'])
                    torch.save(critic, config['critic']['final_save_path'])
                    max_running_total_reward = running_total_reward
                
                error = reward - value
                critic_loss = error.pow(2)
                optimizerC.zero_grad()
                critic_loss.backward()
                optimizerC.step()
                optimizerC.zero_grad()

                if np.remainder(iter,l) == 0:
                    value = critic(state)
                    next_value = critic(next_state)
                    advantage = reward - value
                    actor_loss = -log_prob * advantage.detach()
                    optimizerA.zero_grad()
                    actor_loss.backward()
                    optimizerA.step()
                    optimizerA.zero_grad()

                if config['learning_rate_scheduler']['required']:
                    lr_scheduler(optimizerA, optimizerC, max_running_total_reward)

                break
            else:
                error = reward + gamma * next_value.detach() - value
                critic_loss = error.pow(2)
                optimizerC.zero_grad()
                critic_loss.backward()
                optimizerC.step()
                optimizerC.zero_grad()

                if np.remainder(iter,l) == 0:
                    value = critic(state)
                    next_value = critic(next_state)
                    advantage = reward + gamma * next_value.detach() - value
                    actor_loss = -log_prob * advantage.detach()
                    optimizerA.zero_grad()
                    actor_loss.backward()
                    optimizerA.step()
                    optimizerA.zero_grad()

                state = next_state

    env.close()
    file1.close()
    with open(config['rewards_path'], 'w') as fp:
        json.dump(reward_list, fp, indent=4)
    return np.std(cur_reward_list), running_total_reward    
        
def proj_val(l_val):
        print("value of l_val",l_val)
        j,d=divmod(l_val,1)
        d = float("{0:.2f}".format(d))
        print("Value of j and d",j,d)
        y=random.choices([j,j+1],[1-d,d])
        return int(y[0])
    
def updateStep(actor, critic, n_iters):
#         actor=actor, critic=critic, n_iters=n_iters
        l=5
        k_val=[]
        l_val=[]
        J_dl_val=[]
        ep_val=[]
        file = open("./log_k_projl_avgScore_std.txt","a+")

        for k in range(500):
            delta=1
            # l_update_lr=(0.0001/pow((k+1),(2/3)))
            l_update_lr=0.0005          
            k_val.append(k)
            l_val.append(round(l))
            l_prev=l
            if k%2==0:
                eta = np.random.normal(0.0, 1.0)
                l1=l
                l1=min(max(1,proj_val(l1+eta*delta)),200)
                J_dl, tot_ep_val=run(actor, critic, n_iters,l1,k)
                print("Value of J_dl:",J_dl)
                var1=pow(J_dl,2)
                # l=l-(l_update_lr*(pow(J_dl,2)/delta))                
                J_dl_val.append(J_dl)
                ep_val.append(tot_ep_val)
                file.write("{}\t{}\t{}\t{}\n".format(k,l_prev,round(tot_ep_val,2),J_dl))                 
          
            else:
                l2=l            
                l2=min(max(1,proj_val(l2-eta*delta)),200)                
                J_dl,tot_ep_val=run(actor, critic, n_iters,l2,k)
                print("Value of J_dl:",J_dl)
                var2=pow(J_dl,2)
                # l=l+(l_update_lr*(pow(J_dl,2)/delta))                
                J_dl_val.append(J_dl)
                ep_val.append(tot_ep_val)
                file.write("{}\t{}\t{}\t{}\n".format(k,l_prev,round(tot_ep_val,2),J_dl))

                l=l-((l_update_lr*eta*(var1-var2))/(2*delta))
            print("Updated l value",l)
                
#                 print("k_val,l_val,J_dl_val",k_val,l_val,J_dl_val)
            plot1 =plt.figure(1)
            plt.plot(k_val,ep_val,'b')
            plt.xlabel("No. of l-training iteration")
            plt.ylabel("Average Training score")
            # plt.title('No. of l-training iterations vs Cartpole training variance')
            plt.savefig("./ltraining_score_MtCar_A2C_l_5.svg")

            plot2 = plt.figure(2)
            plt.plot(k_val,J_dl_val,'r')
            plt.xlabel("No. of l-training iteration")
            plt.ylabel("Standard dev. of trainging score")
            # plt.title('No. of l-training iterations vs Acrobot training variance')
            plt.savefig("./ltraining_std_MtCar_A2C_v2C_l_5.svg")

            plot3 = plt.figure(3)    
            plt.plot(k_val,l_val,'g')
            plt.xlabel("No. of l-training iteration")
            plt.ylabel("Projected l value")
            # plt.title('No. of l-training iterations vs projected l value')
            plt.savefig("./projected_l_value_MtCar_A2C_l_5.svg")
        
            print("Value of l is:",l)
        #The following is the testing.
        file.close()
        k=500
        l_opt=min(max(1,proj_val(l)),200)
        J_dl, Avg_score=run(actor, critic, n_iters,l_opt,k)
        print("Test Standard deviation and Average Score is:",J_dl,Avg_score)
        

if __name__ == '__main__':
    if load_A:
        path_A = config['actor']['load_path']
        actor = torch.load(path_A)
        print('Actor Model loaded')
    else:
        actor = Actor(state_size, actor_h_layers_sizes, action_size).to(device)
        torch.save(actor, config['actor']['initial_save_path'])
    
    if load_C:
        path_C = config['critic']['load_path']
        critic = torch.load(path_C)
        print('Critic Model loaded')
    else:    
        critic = Critic(state_size, critic_h_layers_sizes, 1).to(device)
        torch.save(critic, config['critic']['initial_save_path'])

    updateStep(actor, critic, n_iters=iterations)
#     trainIters(actor, critic, n_iters=iterations)


# In[ ]:




