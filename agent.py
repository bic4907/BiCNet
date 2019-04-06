import torch
import numpy as np, random

from network import Actor, Critic
from random_process import OrnsteinUhlenbeckProcess
from utils import soft_update, hard_update

class BiCNet():

    def __init__(self, s_dim, a_dim, n_agents, **kwargs):
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.config = kwargs['config']
        self.n_agents = n_agents
        self.device = 'cuda' if self.config.use_cuda else 'cpu'
        # Networks
        self.policy = Actor(s_dim, a_dim, n_agents)
        self.policy_target = Actor(s_dim, a_dim, n_agents)
        self.critic = Critic(s_dim, a_dim, n_agents)
        self.critic_target = Critic(s_dim, a_dim, n_agents)

        if self.config.use_cuda:
           self.policy.cuda()
           self.policy_target.cuda()
           self.critic.cuda()
           self.critic_target.cuda()

        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config.a_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.c_lr)

        hard_update(self.policy, self.policy_target)
        hard_update(self.critic, self.critic_target)

        self.random_process = OrnsteinUhlenbeckProcess(size=self.a_dim, theta=self.config.ou_theta, mu=self.config.ou_mu, sigma=self.config.ou_sigma)
        self.replay_buffer = list()
        self.epsilon = 1.
        self.depsilon = self.epsilon / self.config.epsilon_decay

        self.c_loss = None
        self.a_loss = None

    def choose_action(self, obs, noisy=True):
        obs = torch.Tensor([obs]).to(self.device)

        action = self.policy(obs).cpu().detach().numpy()[0]
        if noisy:
            for agent_idx in range(self.n_agents):
                action[agent_idx] += max(self.epsilon, 0.001) * self.random_process.sample()
            self.epsilon -= self.depsilon
        np.clip(action, -1., 1.)

        return action

    def reset(self):
        self.random_process.reset_states()

    def prep_train(self):
        self.policy.train()
        self.critic.train()
        self.policy_target.train()
        self.critic_target.train()

    def prep_eval(self):
        self.policy.eval()
        self.critic.eval()
        self.policy_target.eval()
        self.critic_target.eval()


    def random_action(self):
        return np.random.uniform(low=-1, high=1, size=(self.n_agents, 2))

    def memory(self, s, a, r, s_, done):
        self.replay_buffer.append((s, a, r, s_, done))

        if len(self.replay_buffer) >= self.config.memory_length:
            self.replay_buffer.pop(0)

    def get_batches(self):
        experiences = random.sample(self.replay_buffer, self.config.batch_size)

        state_batches = np.array([_[0] for _ in experiences])
        action_batches = np.array([_[1] for _ in experiences])
        reward_batches = np.array([_[2] for _ in experiences])
        next_state_batches = np.array([_[3] for _ in experiences])
        done_batches = np.array([_[4] for _ in experiences])

        return state_batches, action_batches, reward_batches, next_state_batches, done_batches

    def train(self):
        state_batches, action_batches, reward_batches, next_state_batches, done_batches = self.get_batches()

        state_batches = torch.Tensor(state_batches).to(self.device)
        action_batches = torch.Tensor(action_batches).to(self.device)
        reward_batches = torch.Tensor(reward_batches).view(-1, self.n_agents, 1).to(self.device)
        next_state_batches = torch.Tensor(next_state_batches).to(self.device)
        done_batches = torch.Tensor((done_batches == False) * 1).view(-1, self.n_agents, 1).to(self.device)

        target_next_actions = self.policy_target.forward(next_state_batches).detach()
        target_next_q = self.critic_target.forward(next_state_batches, target_next_actions).detach()

        main_q = self.critic(state_batches, action_batches)

        # Critic Loss
        self.critic.zero_grad()
        baselines = reward_batches + done_batches * self.config.gamma * target_next_q
        loss_critic = torch.nn.MSELoss()(main_q, baselines.cuda())
        loss_critic.backward()
        self.critic_optimizer.step()

        # TODO Edit Actor Loss
        # Actor Loss
        self.policy.zero_grad()
        clear_action_batches = self.policy.forward(state_batches)
        loss_actor = (-self.critic.forward(state_batches, clear_action_batches)).mean()
        loss_actor.backward()
        self.policy_optimizer.step()

        # This is for logging
        self.c_loss = loss_critic.item()
        self.a_loss = loss_actor.item()

        soft_update(self.policy, self.policy_target, self.config.tau)
        soft_update(self.critic, self.critic_target, self.config.tau)

    def get_loss(self):
        return self.c_loss, self.a_loss