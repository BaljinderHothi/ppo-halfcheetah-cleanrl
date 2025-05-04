""" this is the copied over ppo_continuous_action.py
my task is to

[X]add some code to save the model checkpoint as it trains

[X]train a neural network on HalfCheetah-v4 and screenshot the chart

[X]write some code (you can include it in the rl file) to load a checkpoint,
run it on an environment for 100 episodes, and dump all of the observations 
and actions in .npz form (or some other data format, I just dont want to see .pkl).



additional note:
after you train the model and dump the data, dump a bunch more data from the checkpoint 
and try to train a model using it. send the loss curves (minimize the negative log prob 
from the model output in the cleanrl file) and reward curves over time 

i like to report something like P(a* | s) which would be exp(-negative log prob.mean()) 
along with the reward, and your loss would be -logprobs.mean()
"""
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    capture_video: bool = False
    save_model: bool = True
    upload_model: bool = False
    hf_entity: str = ""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    total_timesteps: int = 5000000 #increased for more exploring
    learning_rate: float = 3e-4  
    num_envs: int = 16  #gonna increase this to 32 bc i finished setting up cuda on my new pc, it runs on gpu now >:)
    num_steps: int = 2048  
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 10
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01  #added entropy to encourage exploration
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.015  #added target KL to prevent policy collapse

    # to be filled in runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env
    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


#enhanced Agent architecture with larger networks and ReLU activation
class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_shape = np.array(envs.single_observation_space.shape).prod()
        action_shape = np.prod(envs.single_action_space.shape)
        
        #larger critic network with ReLU
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        
        #larger actor network with ReLU
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, action_shape), std=0.01),
        )
        
        
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_shape) - 0.5)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv([
        make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)
    ])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    
    def get_lr_multiplier(iteration, warmup_iterations=0.05):
        max_iterations = args.num_iterations
        warmup_steps = int(max_iterations * warmup_iterations)
        
        if iteration < warmup_steps:
            
            return iteration / warmup_steps
        else:
            
            progress = (iteration - warmup_steps) / (max_iterations - warmup_steps)
            return 0.5 * (1.0 + np.cos(np.pi * progress))

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    
    #tracking highest return for saving best model
    best_episodic_return = float('-inf')

    for iteration in range(1, args.num_iterations + 1):
        
        if args.anneal_lr:
            lrnow = args.learning_rate * get_lr_multiplier(iteration)
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        episodic_return = info["episode"]["r"]
                        writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                        
                        
                        if episodic_return > best_episodic_return:
                            best_episodic_return = float(episodic_return)
                            torch.save(agent.state_dict(), f"best_model.pth")
                            print(f"New best model with return: {best_episodic_return}")

        
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        
        clipfracs = []
        for epoch in range(args.update_epochs):
            
            b_inds = np.random.permutation(args.batch_size)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                
                print(f"Early stopping at epoch {epoch} due to reaching max kl: {approx_kl:.2f}")
                break

        #save the model periodically
        if iteration % 20 == 0:
            torch.save(agent.state_dict(), f"checkpoint_{global_step}.pth")
            print(f"Saved checkpoint at {global_step} steps")

        #log metrics
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        
        # for later on: P(a* | s) metric
        with torch.no_grad():
            prob_metric = torch.exp(-b_logprobs.mean())
            writer.add_scalar("charts/action_probability", prob_metric.item(), global_step)
        
        print(f"Iteration: {iteration}/{args.num_iterations}, SPS: {int(global_step / (time.time() - start_time))}, Current best return: {best_episodic_return:.2f}")

    
    final_path = f"checkpoint_final.pth"
    torch.save(agent.state_dict(), final_path)
    print(f"Saved final model: {final_path}")

    #dump the data
    print("Starting evaluation for 100 episodes...")
    eval_env = gym.make(args.env_id)
    agent.load_state_dict(torch.load("best_model.pth", map_location=device))  
    agent.eval()
    observations, actions_list, rewards_list = [], [], []
    total_rewards = []
    
    for ep in range(100):
        obs_e, _ = eval_env.reset()
        done = False
        total_reward = 0
        while not done:
            obs_tensor = torch.tensor(obs_e, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                act, logprob, _, _ = agent.get_action_and_value(obs_tensor)
            act_np = act.cpu().numpy()[0]
            observations.append(obs_e)
            actions_list.append(act_np)
            obs_e, reward, term, trunc, info = eval_env.step(act_np)
            rewards_list.append(reward)
            total_reward += reward
            done = term or trunc
        total_rewards.append(total_reward)
        print(f"Episode {ep+1}/100, Total Reward: {total_reward:.2f}")
    
    print(f"Average Evaluation Reward: {np.mean(total_rewards):.2f}")
    np.savez("evaluation_data.npz", observations=np.array(observations), actions=np.array(actions_list), rewards=np.array(rewards_list))
    print("Saved evaluation_data.npz")

    #gonna try to finish the IL portion here

    num_imitations = 45 #this is just a place holder, gonna check to see how the run does in tensorboard and adjust it

    data = np.load("evaluation_data.npz")
    obs_np = data["observations"]
    acts_np = data["actions"]
    #this is to just load the dumped data

    N = obs_np.shape[0]
    indices = np.arange(N)
    
    #here is the imitation loop
    #loops over epochs, each epoch we shuffle data, slice it into little batches, compute the loss and keep updating I THINK 
    for epoch in range(1, num_imitations +1):
        np.random.shuffle(indices)

        for start in range(0, N, args.minibatch_size):
            mb_idx = indices[start : start + args.minibatch_size]
            s_batch = torch.tensor(obs_np[mb_idx], dtype=torch.float32).to(device)
            a_batch = torch.tensor(acts_np[mb_idx], dtype=torch.float32).to(device)

            #now to compute the loss for the minibatches
            _, logprob, _, _ = agent.get_action_and_value(s_batch, action=a_batch)
            loss = -logprob.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        #now to do evaluations of the new imiated policy
        #gonna run a few episodes to measure average reward under supervised policy

        agent.eval()
        total_reward = []
        for _ in range(5):
            obs_e, _ = envs.envs[0].reset()
            done, ep_ret = False, 0.0
            while not done:
                with torch.no_grad():
                    act, _, _, _ = agent.get_action_and_value(
                        torch.tensor(obs_e, dtype=torch.float32).unsqueeze(0).to(device)
                    )
                obs_e, r, term, trunc, _ = envs.envs[0].step(act.cpu().numpy()[0])
                ep_ret += r
                done = term or trunc
            total_rewards.append(ep_ret)
        mean_ret = float(np.mean(total_rewards))
        agent.train()


        #now if this all worked properly, this should return properly

        writer.add_scalar("imit/loss",   loss.item(),   epoch)
        writer.add_scalar("imit/reward", mean_ret,      epoch)
        print(f"[Imitation] epoch {epoch:2d}  loss {loss.item():.3f}  reward {mean_ret:.1f}")
