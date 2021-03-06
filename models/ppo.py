import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.distributions import Categorical


class Memory:
    # 记忆因子，里边存有动作，状态，奖励，隐藏因子等
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.hidden = []

    def clear_memory(self):
        #清除记忆因子
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.hidden[:]


class ActorCritic(nn.Module):
    def __init__(self, feature_dim, state_dim, action_dim, hidden_state_dim=1024, policy_conv=True):
        super(ActorCritic, self).__init__()
        
        # encoder with convolution layer for MobileNetV3, EfficientNet and RegNet
        if policy_conv:
            self.state_encoder = nn.Sequential(
                nn.Conv2d(feature_dim, 32, kernel_size=1, stride=1, padding=0, bias=False),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(int(state_dim * 32 / feature_dim), hidden_state_dim),
                nn.ReLU()
            )
        # encoder with linear layer for ResNet and DenseNet
        else:
            self.state_encoder = nn.Sequential(
                nn.Linear(state_dim, 2048),
                nn.ReLU(),
                nn.Linear(2048, hidden_state_dim),
                nn.ReLU()
            )
        # 1024,1024,true(=0)
        self.gru = nn.GRU(hidden_state_dim, hidden_state_dim, batch_first=False)
        # 演员
        self.actor = nn.Sequential(
            nn.Linear(hidden_state_dim, action_dim),
            nn.Softmax(dim=-1))
        # 评论家
        self.critic = nn.Sequential(
            nn.Linear(hidden_state_dim, 1))

        self.hidden_state_dim = hidden_state_dim
        self.action_dim = action_dim
        self.policy_conv = policy_conv
        self.feature_dim = feature_dim
        # 49
        self.feature_ratio = int(math.sqrt(state_dim/feature_dim))

    def forward(self):
        raise NotImplementedError

    def act(self, state_ini, memory, restart_batch=False, training=True):
        if restart_batch:
            del memory.hidden[:]
            # hidden里送入一个torch.zeros(1,32,1024)
            # memory.hidden.append(torch.zeros(1, state_ini.size(0), self.hidden_state_dim))
            memory.hidden.append(torch.zeros(1, state_ini.size(0), self.hidden_state_dim).cuda())

        if not self.policy_conv:
            state = state_ini.flatten(1)
        else:
            # state:[32,3,224,224]
            state = state_ini

        state = self.state_encoder(state) # 送入神经网络,input[32,1280,7,7],output[32,1024]

        # state.view(1,32,1024), memory.hidden[-1]):[1,32,1024];output:state[1,32,1024],hidden_output:[1,32,1024]
        state, hidden_output = self.gru(state.view(1, state.size(0), state.size(1)), memory.hidden[-1])

        memory.hidden.append(hidden_output)

        state = state[0] #[32,1024]
        action_probs = self.actor(state) #[32,49]
        dist = Categorical(action_probs)# 生成分布，根据概率分布来产生sample，产生的sample是输入tensor的index

        if training:
            action = dist.sample()# 随机采样
            action_logprob = dist.log_prob(action)
            memory.states.append(state_ini)
            memory.actions.append(action)
            memory.logprobs.append(action_logprob)
        else:
            action = action_probs.max(1)[1]# 不训练的话直接取概率最大值的意思
        # action.shape = 32

        return action

    def evaluate(self, state, action):
        seq_l = state.size(0)
        batch_size = state.size(1)

        if not self.policy_conv:
            state = state.flatten(2)
            state = state.view(seq_l * batch_size, state.size(2))
        else:
            state = state.view(seq_l * batch_size, state.size(2), state.size(3), state.size(4))

        state = self.state_encoder(state)
        state = state.view(seq_l, batch_size, -1)

        state, hidden = self.gru(state, torch.zeros(1, batch_size, state.size(2)).cuda())
        state = state.view(seq_l * batch_size, -1)

        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(torch.squeeze(action.view(seq_l * batch_size, -1))).cuda()
        dist_entropy = dist.entropy().cuda()
        state_value = self.critic(state)

        return action_logprobs.view(seq_l, batch_size), \
               state_value.view(seq_l, batch_size), \
               dist_entropy.view(seq_l, batch_size)


class PPO(nn.Module):
    # 1280，1280*7*7，49，1024，true，gamma=0.7，lr=0.0003
    def __init__(self, feature_dim, state_dim, action_dim, hidden_state_dim, policy_conv, gpu=0,
                lr=0.0003, betas=(0.9, 0.999), gamma=0.7, K_epochs=1, eps_clip=0.2):
        super(PPO, self).__init__()
        self.lr = lr # 0.0003
        self.betas = betas # （0.9，0.999）
        self.gamma = gamma # 0.7
        self.eps_clip = eps_clip # 0.2
        self.K_epochs = K_epochs # 1

        # 1280，1280*7*7，49，1024，true
        self.policy = ActorCritic(feature_dim, state_dim, action_dim, hidden_state_dim, policy_conv).cuda(gpu)
        # self.policy = ActorCritic(feature_dim, state_dim, action_dim, hidden_state_dim, policy_conv)
        # 优化器
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        # 相同的策略网络
        self.policy_old = ActorCritic(feature_dim, state_dim, action_dim, hidden_state_dim, policy_conv).cuda(gpu)
        # self.policy_old = ActorCritic(feature_dim, state_dim, action_dim, hidden_state_dim, policy_conv)
        # 加载权重
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state, memory, restart_batch=False, training=True):
        # [32,3,224,224],memory,true(=0)false(>0),true
        return self.policy_old.act(state, memory, restart_batch, training)

    def update(self, memory):
        rewards = []
        discounted_reward = 0

        for reward in reversed(memory.rewards):
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.cat(rewards, 0).cuda()

        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        old_states = torch.stack(memory.states, 0).cuda().detach()
        old_actions = torch.stack(memory.actions, 0).cuda().detach()
        old_logprobs = torch.stack(memory.logprobs, 0).cuda().detach()

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            ratios = torch.exp(logprobs - old_logprobs.detach())

            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())