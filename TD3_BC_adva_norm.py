import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l1_norm = nn.LayerNorm(256)
        self.l2 = nn.Linear(256, 256)
        self.l2_norm = nn.LayerNorm(256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l4_norm = nn.LayerNorm(256)
        self.l5 = nn.Linear(256, 256)
        self.l5_norm = nn.LayerNorm(256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = self.l1_norm(self.l1(sa))
        q1 = F.relu(q1)
        q1 = self.l2_norm(self.l2(q1))
        q1 = F.relu(q1)
        q1 = self.l3(q1)

        q2 = self.l4_norm(self.l4(sa))
        q2 = F.relu(q2)
        q2 = self.l5_norm(self.l5(q2))
        q2 = F.relu(q2)
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = self.l1_norm(self.l1(sa))
        q1 = F.relu(q1)
        q1 = self.l2_norm(self.l2(q1))
        q1 = F.relu(q1)
        q1 = self.l3(q1)
        return q1


class Vnet(nn.Module):
    def __init__(self, state_dim):
        super(Vnet, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l1_norm = nn.LayerNorm(256)
        self.l2 = nn.Linear(256, 256)
        self.l2_norm = nn.LayerNorm(256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, state):
        v = self.l1_norm(self.l1(state))
        v = F.relu(v)
        v = self.l2_norm(self.l2(v))
        v = F.relu(v)
        v = self.l3(v)

        return v


class TD3_BC(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            alpha=2.5,
            beta=0.5,
    ):

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.vnet = Vnet(state_dim).to(device)
        self.vnet_target = copy.deepcopy(self.vnet)
        self.vnet_optimizer = torch.optim.Adam(self.vnet.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha
        self.beta = beta
        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        log_dict = {}
        self.total_it += 1
        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                    self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            # 修改为BCQ中的更新形式
            target_Q = torch.min(target_Q1, target_Q2)
            # target_Q = 0.75 * torch.min(target_Q1, target_Q2) + 0.25 * torch.max(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q
            log_dict['target_Q-value'] = target_Q.mean().item()

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        log_dict['critic_loss'] = critic_loss.item()
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        '''update vnet'''
        with torch.no_grad():
            target_V = self.vnet_target(next_state)
            target_V = reward + not_done * self.discount * target_V
        value = self.vnet(state)  # (batch_size, 1)每一个状态的价值
        vnet_loss = F.mse_loss(target_V, value)
        log_dict['vnet_loss'] = vnet_loss.item()
        self.vnet_optimizer.zero_grad()
        vnet_loss.backward()
        self.vnet_optimizer.step()

        # bc_weight = 1.0 * (target_V > value.detach())  # 选出有价值的动作
        # [2025.4.14 修改] version2，信任度角度
        trust_score = target_V - value.detach()
        bc_weight = torch.sigmoid(trust_score / self.beta)

        # bc_ratio = len(np.nonzero(bc_weight)) / len(bc_weight)
        bc_ratio = torch.mean(bc_weight).detach()
        log_dict['bc_ratio'] = bc_ratio
        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss
            pi = self.actor(state)
            Q = self.critic.Q1(state, pi)
            lmbda = self.alpha / Q.abs().mean().detach()
            # '*bc_weight'： 好的动作要，不好的动作不要
            actor_loss = -lmbda * Q.mean() + (torch.square(pi - action) * bc_weight).mean()
            log_dict['actor_loss'] = actor_loss.item()
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.vnet.parameters(), self.vnet_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            return [log_dict, self.total_it]
    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
