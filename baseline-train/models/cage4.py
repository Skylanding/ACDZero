import torch
from torch import nn
from torch.optim import Adam
from torch.distributions.categorical import Categorical
from torch_geometric.nn import GCNConv

from models.memory_buffer import MultiPPOMemory
from models.utils import combine_marl_states
from models.mcts import MuZeroPlanner

MAX_SERVERS = 6
MAX_USERS = 10
MAX_EDGES = 8


def pad_sequence(seq, lens, padding):
    padded = torch.zeros(lens.size(0), padding, seq.size(-1))
    mask = torch.ones(padded.size(0), padded.size(1))

    offset = 0
    for i,len in enumerate(lens):
        st = offset
        en = offset+len

        padded[i][:len] = seq[st:en]
        mask[i][len:] = 0
        offset += len

    return padded, mask.unsqueeze(-1)

def extract_hosts(x, servers, n_servers, users, n_users):
    srv = x[servers]
    srv,s_mask = pad_sequence(srv, n_servers, MAX_SERVERS)  # B x MAX_s x d

    usr = x[users]
    usr,u_mask = pad_sequence(usr, n_users, MAX_USERS)      # B x MAX_u x d

    hosts = torch.cat([srv,usr], dim=1)
    mask = torch.cat([s_mask, u_mask], dim=1)
    return hosts, mask

class SimpleSelfAttention(nn.Module):
    '''
    Implimenting global-node self-attention from
        https://arxiv.org/pdf/2009.12462.pdf
    '''
    def __init__(self, in_dim, h_dim, g_dim):
        super().__init__()

        self.att = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.Softmax(dim=-1)
        )
        self.feat = nn.Linear(in_dim, h_dim)
        self.glb = nn.Sequential(
            nn.Linear(h_dim+g_dim, g_dim),
            nn.Tanh()
        )

        self.g_dim = g_dim
        self.h_dim = h_dim

    def forward(self, v, mask, g=None):
        '''
        Inputs:
            v:      B x N x d tensor
            mask:   B x N x 1 tensor of 1s or 0s
            g:      B x d tensor
        '''
        if g is None:
            g = torch.zeros((v.size(0), self.g_dim))

        att = self.att(v)                   # B x N x h
        feat = self.feat(v)                 # B x N x h
        out = (att*feat*mask).sum(dim=1)    # B x h

        g_ = self.glb(torch.cat([out,g], dim=-1))  # B x g
        return g + g_                              # Short-circuit


class InductiveActorNetwork(nn.Module):
    def __init__(self, in_dim, global_state_space=3,
                 node_action_space=4, edge_action_space=2, global_action_space=1,
                 hidden1=256, hidden2=64, gdim=64, lr=0.0003, concat_edges=False):
        super().__init__()

        self.conv1 = GCNConv(in_dim, hidden1)
        self.conv2 = GCNConv(hidden1, hidden2)

        self.g0_attn = SimpleSelfAttention(in_dim, hidden1, gdim)
        self.g1_attn = SimpleSelfAttention(hidden1, hidden1, gdim)
        self.g2_attn = SimpleSelfAttention(hidden2, hidden2, gdim)

        # Just learn a good parameter to encode each phase as
        self.global_net = nn.Linear(
            global_state_space, gdim
        )

        self.node_actions = nn.Sequential(
            nn.Linear(hidden2+gdim, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, hidden2 // 2),
            nn.ReLU(),
            nn.Linear(hidden2 // 2, node_action_space)
        )

        # If edges should be processed as
        # f(src * dst) or f(src || dst)
        self.concat_edges = concat_edges
        self.edge_actions = nn.Sequential(
            nn.Linear(hidden2 if not concat_edges else hidden2*2, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, hidden2 // 2),
            nn.ReLU(),
        )
        self.edge_out = nn.Linear(hidden2 // 2 + gdim, edge_action_space)

        self.global_out = nn.Sequential(
            nn.Linear(gdim, gdim//2),
            nn.ReLU(),
            nn.Linear(gdim//2, global_action_space)
        )

        self.sm = nn.Softmax(dim=1)
        self.opt = Adam(self.parameters(), lr)

        self.node_action_space = node_action_space
        self.edge_action_space = edge_action_space
        self.global_action_space = global_action_space

    def forward(self, x, ei, global_vec, servers, n_servers, users, n_users, action_edges, multi_subnet):
        # Always come in groups of 9
        rtrs = action_edges.unique(sorted=True).squeeze(-1)
        bs = rtrs.size(0) // 9
        rtrs = rtrs.reshape(bs, 9)
        if multi_subnet:
            rtrs = rtrs.repeat_interleave(3,0)

        rtr_mask = torch.ones(rtrs.size(0), 9, 1)

        # Global init features
        g0 = self.global_net(global_vec)

        v,mask = extract_hosts(x, servers, n_servers, users, n_users)
        rtr = x[rtrs]
        v = torch.cat([v, rtr], dim=1)
        mask = torch.cat([mask, rtr_mask], dim=1)
        g = self.g0_attn(v,mask, g=g0)

        # Layer 1
        x = torch.relu(self.conv1(x, ei))
        v,mask = extract_hosts(x, servers, n_servers, users, n_users)
        rtr = x[rtrs]
        v = torch.cat([v, rtr], dim=1)
        mask = torch.cat([mask, rtr_mask], dim=1)
        g = self.g1_attn(v,mask, g=g)

        # Layer 2
        x = torch.relu(self.conv2(x, ei))
        v,mask = extract_hosts(x, servers, n_servers, users, n_users)
        rtr = x[rtrs]
        v = torch.cat([v, rtr], dim=1)
        mask = torch.cat([mask, rtr_mask], dim=1)
        g = self.g2_attn(v,mask, g=g) # B x d_g

        # B x 16 x d
        z,mask = extract_hosts(x, servers, n_servers, users, n_users)

        # Attach global vec to all nodes in each batch
        z = torch.cat(
            [z, g.unsqueeze(1).repeat(1,z.size(1),1)],
            dim=-1
        )
        node_a = self.node_actions(z) * mask

        # B x 16 x a_n
        nbatches = node_a.size(0)

        # Make rows actions, and columns nodes
        node_a = node_a.transpose(1,2)  # B x a_n x 16
        node_a = node_a.reshape(        # B x 16*a_n
            nbatches,
            (MAX_SERVERS+MAX_USERS)*self.node_action_space
        )

        # Calculate edge-level action probs
        src,dst = action_edges
        src = x[src]; dst = x[dst]

        if self.concat_edges:
            edge_a = self.edge_actions(torch.cat([src,dst], dim=-1))
        else:
            edge_a = self.edge_actions(src) * self.edge_actions(dst)

        # Add in global vector
        edge_a = torch.cat([
            edge_a,
            g.repeat_interleave(MAX_EDGES,0)
        ], dim=1)
        edge_a = self.edge_out(edge_a)

        # Assume edge actions are always in groups of 8 (as they are in CAGE4)
        edge_a = edge_a.reshape(        # B x 8 x a_e
            node_a.size(0),
            MAX_EDGES,
            edge_a.size(-1)
        )
        edge_a = edge_a.transpose(1,2)  # B x a_e x 8 (columns are nodes, rows are actions)
        edge_a = edge_a.reshape(        # B x 8*a_e
            nbatches, edge_a.size(1)*MAX_EDGES
        )

        # Finally, compute prob of taking a global action
        # (Not an action upon a node or an edge. E.g. sleep)
        glb_a = self.global_out(g) # B x d

        out = torch.cat([node_a, edge_a, glb_a], dim=-1)

        # blue_agent_4 sends in groups of 3 subnets.
        # Really makes batching tricky
        if multi_subnet:
            out = out.reshape(out.size(0)//3, out.size(1)*3)

        out[out == 0] = -float('inf')   # So softmax prob is 0
        out = self.sm(out)

        return Categorical(out)

    @property
    def action_dim(self):
        return (MAX_SERVERS+MAX_USERS)*self.node_action_space + MAX_EDGES*self.edge_action_space + self.global_action_space


class InductiveCriticNetwork(nn.Module):
    def __init__(self, in_dim, global_state_space=3,
                 hidden1=256, hidden2=64, gdim=64, lr=0.001):
        super().__init__()

        self.conv1 = GCNConv(in_dim, hidden1)
        self.conv2 = GCNConv(hidden1, hidden2)
        self.out = nn.Sequential(
            nn.Linear(hidden2, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, 1)
        )

        self.gs = nn.Linear(
            global_state_space, gdim
        )
        self.g0_attn = SimpleSelfAttention(in_dim, hidden1, gdim)
        self.g1_attn = SimpleSelfAttention(hidden1, hidden1, gdim)
        self.g2_attn = SimpleSelfAttention(hidden2, hidden2, gdim)

        self.out = nn.Sequential(
            nn.Linear(gdim, gdim//2),
            nn.ReLU(),
            nn.Linear(gdim//2, 1)
        )
        self.opt = Adam(self.parameters(), lr)

    def forward(self, x, ei, global_vec, servers, n_servers, users, n_users, action_edges, multi_subnet):
        g0 = self.gs(global_vec)

        v,mask = extract_hosts(x, servers, n_servers, users, n_users)
        g = self.g0_attn(v, mask, g=g0)

        x = torch.relu(self.conv1(x, ei))
        v,mask = extract_hosts(x, servers, n_servers, users, n_users)
        g = self.g1_attn(v, mask, g=g)

        x = torch.relu(self.conv2(x, ei))
        v,mask = extract_hosts(x, servers, n_servers, users, n_users)
        g = self.g2_attn(v, mask, g=g)

        # I guess just average the three global vectors together?
        if multi_subnet:
            g = g.reshape(g.size(0) // 3, 3, g.size(-1))
            g = g.mean(dim=1)

        return self.out(g)


class InductiveGraphPPOAgent():
    '''
    Class to manage agents' memories and learning (when training)
    When training is complete, uses the InductiveActorNetwork to decide
    which action to take
    '''
    def __init__(self, in_dim, gamma=0.99, lmbda=0.95, clip=0.1, bs=5, epochs=6,
                 a_kwargs=dict(), c_kwargs=dict(), training=True, concat_edges=False,
                 use_mcts=False, num_simulations=32, c_puct=1.5, temperature=1.0,
                 lambda_pi=0.5, latent_dim=128, mcts_head_loss_w=0.0,
                 temperature_train=1.0, temperature_eval=0.1,
                 dirichlet_epsilon=0.25, dirichlet_alpha=0.3,
                 use_dynamic_c_puct=True, c_base=19652, c_init=1.25):

        self.actor = InductiveActorNetwork(in_dim, concat_edges=concat_edges, **a_kwargs)
        self.critic = InductiveCriticNetwork(in_dim, **c_kwargs)
        self.memory = MultiPPOMemory(bs, agents=5)
        self.use_mcts = use_mcts
        self.lambda_pi = lambda_pi

        self.args = (in_dim,)
        self.kwargs = dict(
            gamma=gamma, lmbda=lmbda, clip=clip, bs=bs, epochs=epochs,
            a_kwargs=a_kwargs, c_kwargs=c_kwargs, training=training, concat_edges=concat_edges,
            use_mcts=use_mcts, num_simulations=num_simulations, c_puct=c_puct,
            temperature=temperature, lambda_pi=lambda_pi, latent_dim=latent_dim,
            temperature_train=temperature_train, temperature_eval=temperature_eval,
            dirichlet_epsilon=dirichlet_epsilon, dirichlet_alpha=dirichlet_alpha,
            use_dynamic_c_puct=use_dynamic_c_puct, c_base=c_base, c_init=c_init,
            mcts_head_loss_w=mcts_head_loss_w
        )

        # PPO Hyperparams
        self.gamma = gamma
        self.lmbda = lmbda
        self.clip = clip
        self.bs = bs
        self.epochs = epochs

        self.training = training
        self.deterministic = False
        self.mse = nn.MSELoss()
        self.mcts_head_loss_w = mcts_head_loss_w

        # Simple MuZero-style latent model (policy/value from actor/critic, learned dynamics)
        # action_dim may change for multi_subnet agents (blue_agent_4 outputs 3x actions)
        self.action_dim = None
        self.latent_dim = latent_dim
        self.repr_mlp = None
        self.action_emb = None
        self.gru = nn.GRUCell(latent_dim, latent_dim)
        self.reward_head = None
        self.policy_head = None
        self.value_head = nn.Linear(latent_dim, 1)
        self.planner = None
        if self.use_mcts:
            self.planner = MuZeroPlanner(
                self,
                num_simulations=num_simulations,
                c_puct=c_puct,
                gamma=gamma,
                temperature_train=temperature_train,
                temperature_eval=temperature_eval,
                dirichlet_epsilon=dirichlet_epsilon,
                dirichlet_alpha=dirichlet_alpha,
                use_dynamic_c_puct=use_dynamic_c_puct,
                c_base=c_base,
                c_init=c_init
            )

    # Required by CAGE but not utilized
    def end_episode(self):
        pass

    # Required by CAGE but not utilized
    def set_initial_values(self, action_space, observation):
        pass

    def train(self):
        '''
        Set modules to training mode
        '''
        self.actor.train()
        self.critic.train()
        if self.planner:
            self.planner.set_training(True)

    def eval(self):
        '''
        Set modules to eval mode 
        '''
        self.training = False
        self.actor.eval()
        self.critic.eval()
        if self.planner:
            self.planner.set_training(False)

    def _zero_grad(self):
        '''
        Reset opt
        '''
        self.actor.opt.zero_grad()
        self.critic.opt.zero_grad()

    def _step(self):
        '''
        Call opt autograd
        '''
        self.actor.opt.step()
        self.critic.opt.step()


    def set_deterministic(self, val):
        self.deterministic = val

    def set_mems(self, mems):
        self.memory.mems = mems

    def save(self, outf='saved_models/ppo.pt'):
        me = (self.args, self.kwargs)
        mcts_state = None
        if self.repr_mlp is not None and self.action_emb is not None:
            mcts_state = {
                'action_dim': self.action_dim,
                'repr_mlp': self.repr_mlp.state_dict(),
                'action_emb': self.action_emb.state_dict(),
                'gru': self.gru.state_dict(),
                'reward_head': self.reward_head.state_dict(),
                'value_head': self.value_head.state_dict(),
                'policy_head': self.policy_head.state_dict(),
                'planner_cfg': {
                    'num_simulations': self.planner.num_simulations if self.planner else None,
                    'c_puct': self.planner.c_puct if self.planner else None,
                    'temperature_train': self.planner.temperature_train if self.planner else None,
                    'temperature_eval': self.planner.temperature_eval if self.planner else None,
                    'dirichlet_epsilon': self.planner.dirichlet_epsilon if self.planner else None,
                    'dirichlet_alpha': self.planner.dirichlet_alpha if self.planner else None,
                    'use_dynamic_c_puct': self.planner.use_dynamic_c_puct if self.planner else None,
                    'c_base': self.planner.c_base if self.planner else None,
                    'c_init': self.planner.c_init if self.planner else None,
                }
            }

        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'agent': me,
            'mcts': mcts_state
        }, outf)

    @torch.no_grad()
    def get_action(self, obs, *args):
        '''
        Sample an action from the actor's distribution
        given the current state. 

        If eval(), only returns the action 
        If train() returns action, value, and log prob 
        '''
        state,is_blocked = obs
        if is_blocked:
            return None if not self.training else (None, 0.0, 0.0, None)

        pi_mcts = None

        if self.use_mcts and self.planner is not None:
            action, pi_mcts = self.planner.run(state)
            action = torch.tensor(action)
            distro = self.actor(*state)
        else:
            distro = self.actor(*state)
            if self.deterministic:
                action = distro.probs.argmax()
            else:
                action = distro.sample()

        if not self.training:
            return action.item()

        value = self.critic(*state)
        prob = distro.log_prob(action)
        pi_list = pi_mcts if pi_mcts is not None else distro.probs.detach().cpu().tolist()
        return action.item(), value.item(), prob.item(), pi_list

    # --- MuZero-style model interface for planner ---
    def _ensure_model_dims(self, prob_dim: int):
        """
        Lazily (re)initializes latent-model heads to match current action_dim.
        Needed because agent 4 can output 3x action space (multi_subnet).
        """
        if self.action_dim == prob_dim and self.repr_mlp is not None:
            return
        self.action_dim = prob_dim
        # Recreate heads/embeddings to match current dim
        self.repr_mlp = nn.Sequential(
            nn.Linear(prob_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        self.action_emb = nn.Embedding(prob_dim, self.latent_dim)
        self.reward_head = nn.Linear(self.latent_dim, 1)
        self.policy_head = nn.Linear(self.latent_dim, prob_dim)

    def _build_mcts_modules(self, prob_dim: int):
        """Explicitly build latent modules to a given action_dim (for loading)."""
        self.action_dim = prob_dim
        self.repr_mlp = nn.Sequential(
            nn.Linear(prob_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        self.action_emb = nn.Embedding(prob_dim, self.latent_dim)
        self.reward_head = nn.Linear(self.latent_dim, 1)
        self.policy_head = nn.Linear(self.latent_dim, prob_dim)

    def initial_inference(self, state):
        """
        Returns (policy_logits, value, latent) for root node.
        policy_logits/value are 1D tensors; latent is B x latent_dim
        """
        dist = self.actor(*state)
        prob_dim = dist.probs.shape[-1]
        self._ensure_model_dims(prob_dim)
        policy_logits = torch.log(dist.probs + 1e-8)
        value = self.critic(*state).squeeze(-1)
        latent = self.repr_mlp(dist.probs)
        # ensure shape (batch, latent_dim)
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)
        return policy_logits.squeeze(0), value, latent

    def recurrent_inference(self, latent, action):
        """
        Given latent and action (int), predict next (policy_logits, value, reward, latent_next)
        """
        if not torch.is_tensor(action):
            action = torch.tensor(action, device=latent.device)
        if action.dim() == 0:
            action = action.unsqueeze(0)
        # action embedding
        a_emb = self.action_emb(action)
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)
        next_latent = self.gru(a_emb, latent)
        reward = self.reward_head(next_latent).squeeze(-1)
        policy_logits = self.policy_head(next_latent)
        value = self.value_head(next_latent).squeeze(-1)
        return policy_logits.squeeze(0), value.squeeze(0), reward.squeeze(0), next_latent.detach()

    def remember(self, idx, s, a, v, p, pi, r, t):
        '''
        Save an observation to the agent's memory buffer
        '''
        self.memory.remember(idx, s,a,v,p,pi,r,t)

    def learn(self, verbose=False):
        '''        
        This runs the PPO update algorithm on memories stored in self.memory 
        Assumes that an external process is adding memories to the buffer
        '''
        for e in range(self.epochs):
            s,a,v,p,pi,r,t, batches = self.memory.get_batches()

            # Calculate discounted reward
            rewards = []
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(r), reversed(t)):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + self.gamma * discounted_reward
                rewards.insert(0, discounted_reward)

            # Normalize 
            r = torch.tensor(rewards, dtype=torch.float)
            r = (r - r.mean()) / (r.std() + 1e-5) # Normalize rewards

            # Calculate advantage 
            advantages = r - torch.tensor(v)
            closs,aloss,eloss = 0,0,0

            # Optimize for clipped advantage for each minibatch 
            for b_idx,b in enumerate(batches):
                b = b.tolist()
                new_probs = []

                # Combine graphs from minibatches so GNN is called once
                s_ = [s[idx] for idx in b]
                a_ = [a[idx] for idx in b]
                batched_states = combine_marl_states(s_)

                self._zero_grad()

                # Forward pass 
                dist = self.actor(*batched_states)
                critic_vals = self.critic(*batched_states)

                new_probs = dist.log_prob(torch.tensor(a_))
                old_probs = torch.tensor([p[i] for i in b])
                entropy = dist.entropy()

                a_t = advantages[b]

                # Equiv to exp(new) / exp(old) b.c. recall: these are log probs
                r_theta = (new_probs - old_probs).exp()
                clipped_r_theta = torch.clip(
                    r_theta, min=1-self.clip, max=1+self.clip
                )

                # Use whichever one is minimal
                actor_loss = torch.min(r_theta*a_t, clipped_r_theta*a_t)
                actor_loss = -actor_loss.mean()

                # Critic uses MSE loss between expected value of state and observed
                # reward with discount factor
                critic_loss = self.mse(r[b].unsqueeze(-1), critic_vals)

                # Not totally necessary but maybe will help?
                entropy_loss = entropy.mean()

                # Calculate gradient and backprop
                # Optional policy distillation toward improved pi (e.g., MCTS)
                pi_targets = [pi[i] for i in b]
                has_pi = any([x is not None for x in pi_targets])
                if has_pi:
                    # Replace None with current probs to keep shapes valid
                    replaced = []
                    probs_detached = dist.probs.detach().cpu()
                    for j, tgt in enumerate(pi_targets):
                        if tgt is None:
                            replaced.append(probs_detached[j].tolist())
                        else:
                            replaced.append(tgt)
                    pi_tensor = torch.tensor(replaced, dtype=dist.probs.dtype, device=dist.probs.device)
                    # Cross-entropy: target * log pred (pred already prob)
                    pi_loss = -(pi_tensor * torch.log(dist.probs + 1e-8)).sum(dim=1).mean()
                else:
                    pi_loss = 0.0

                total_loss = actor_loss + 0.5*critic_loss - 0.01*entropy_loss
                if has_pi:
                    total_loss = total_loss + self.lambda_pi * pi_loss
                total_loss.backward()
                self._step()

                # Print loss for each minibatch if verbose 
                # (aggregate loss is printed regardless)
                if verbose:
                    print(f'[{e}] C-Loss: {0.5*critic_loss.item():0.4f}  A-Loss: {actor_loss.item():0.4f} E-loss: {-entropy_loss.item()*0.01:0.4f}')

                closs += critic_loss.item()
                aloss += actor_loss.item()
                eloss += entropy_loss.item()

            # Print avg loss across minibatches
            closs /= len(batches)
            aloss /= len(batches)
            eloss /= len(batches)
            print(f'[{e}] C-Loss: {0.5*closs:0.4f}  A-Loss: {aloss:0.4f} E-loss: {-eloss*0.01:0.4f}')

        # After we have sampled our minibatches e times, clear the memory buffer
        self.memory.clear()
        return total_loss.item()


def load(in_f):
    '''
    Loads model checkpoint file 
    '''
    data = torch.load(in_f)
    args,kwargs = data['agent']

    agent = InductiveGraphPPOAgent(*args, **kwargs)
    agent.actor.load_state_dict(data['actor'])
    agent.critic.load_state_dict(data['critic'])

    # Load MCTS latent modules if present
    if 'mcts' in data and data['mcts'] is not None:
        m = data['mcts']
        if m.get('action_dim') is not None:
            agent._build_mcts_modules(m['action_dim'])
            agent.repr_mlp.load_state_dict(m['repr_mlp'])
            agent.action_emb.load_state_dict(m['action_emb'])
            agent.gru.load_state_dict(m['gru'])
            agent.reward_head.load_state_dict(m['reward_head'])
            agent.value_head.load_state_dict(m['value_head'])
            agent.policy_head.load_state_dict(m['policy_head'])
            # Restore planner config if planner exists
            if agent.planner and m.get('planner_cfg'):
                cfg = m['planner_cfg']
                if cfg.get('num_simulations') is not None:
                    agent.planner.num_simulations = cfg['num_simulations']
                if cfg.get('c_puct') is not None:
                    agent.planner.c_puct = cfg['c_puct']
                if cfg.get('temperature_train') is not None:
                    agent.planner.temperature_train = cfg['temperature_train']
                if cfg.get('temperature_eval') is not None:
                    agent.planner.temperature_eval = cfg['temperature_eval']
                if cfg.get('dirichlet_epsilon') is not None:
                    agent.planner.dirichlet_epsilon = cfg['dirichlet_epsilon']
                if cfg.get('dirichlet_alpha') is not None:
                    agent.planner.dirichlet_alpha = cfg['dirichlet_alpha']
                if cfg.get('use_dynamic_c_puct') is not None:
                    agent.planner.use_dynamic_c_puct = cfg['use_dynamic_c_puct']
                if cfg.get('c_base') is not None:
                    agent.planner.c_base = cfg['c_base']
                if cfg.get('c_init') is not None:
                    agent.planner.c_init = cfg['c_init']

    agent.eval()
    return agent

