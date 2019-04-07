from env.make_env import make_env
import argparse, datetime
from tensorboardX import SummaryWriter
import numpy as np
import torch
import gym

from agent import BiCNet
from normalized_env import ActionNormalizedEnv, ObsEnv, reward_from_state

def main(args):

    env = make_env('simple_spread')
    # env = make_env('simple')
    # env = gym.make('Pendulum-v0')
    env = ActionNormalizedEnv(env)
    env = ObsEnv(env)

    kwargs = dict()
    kwargs['config'] = args
    torch.manual_seed(args.seed)

    if args.tensorboard:
        writer = SummaryWriter(log_dir='runs/'+args.log_dir)
    model = BiCNet(14, 2, 3, **kwargs)
    # model = BiCNet(4, 2, 1, **kwargs)

    episode = 0
    total_step = 0

    while episode < args.max_episodes:

        state = env.reset()

        episode += 1
        step = 0
        accum_reward = 0
        prev_reward = np.zeros((3), dtype=np.float32)

        while True:

            # action = agent.random_action()
            if episode > args.warmup:
                action = model.choose_action(state, noisy=True)
            else:
                action = model.random_action()

            next_state, reward, done, info = env.step(action)
            step += 1
            total_step += 1


            reward = np.array(reward)

            '''KeyboardInterrupt
            Reward Shaping
                - Distance to landmarks
            '''

            rew1 = reward_from_state(next_state)
            #if step % 5 == 0:
            #     rew1 -= 0.1
            reward = rew1 + (np.array(reward, dtype=np.float32) / 100.)
            accum_reward += sum(reward)

            if args.render and episode % 100 == 0:
                env.render(mode='rgb_array')
            model.memory(state, action, reward, next_state, done)

            state = next_state

            if len(model.replay_buffer) >= args.batch_size and total_step % args.steps_per_update == 0:
                model.prep_train()
                model.train()
                model.prep_eval()

            if args.episode_length < step or (True in done):
                c_loss, a_loss = model.get_loss()
                action_std = model.get_action_std()
                print("[Episode %05d] reward %6.4f eps %.4f" % (episode, accum_reward, model.epsilon), end='')
                if args.tensorboard:
                    writer.add_scalar(tag='agent/reward', global_step=episode, scalar_value=accum_reward.item())
                    writer.add_scalar(tag='agent/epsilon', global_step=episode, scalar_value=model.epsilon)
                    if c_loss and a_loss:
                        writer.add_scalars('agent/loss', global_step=episode, tag_scalar_dict={'actor':a_loss, 'critic':c_loss})
                    if action_std:
                        writer.add_scalar(tag='agent/action_std', global_step=episode, scalar_value=action_std)
                if c_loss and a_loss:
                    print(" a_loss %3.2f c_loss %3.2f" % (a_loss, c_loss), end='')
                if action_std:
                    print(" action_std %3.2f" % (action_std), end='')


                print()
                env.reset()
                model.reset()
                break





    if args.tensorboard:
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_episodes', default=1000000, type=int)
    parser.add_argument('--episode_length', default=50, type=int)
    parser.add_argument('--memory_length', default=int(1e5), type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument('--tau', default=0.001, type=float)
    parser.add_argument('--gamma', default=0.95, type=float)
    parser.add_argument('--warmup', default=100, type=int)
    parser.add_argument('--use_cuda', default=True, type=bool)
    parser.add_argument('--seed', default=777, type=int)
    parser.add_argument('--a_lr', default=0.001, type=float)
    parser.add_argument('--c_lr', default=0.001, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--render', action="store_true")
    parser.add_argument('--ou_theta', default=0.15, type=float)
    parser.add_argument('--ou_mu', default=0.0, type=float)
    parser.add_argument('--ou_sigma', default=0.2, type=float)
    parser.add_argument('--epsilon_decay', default=1000000, type=int)
    parser.add_argument('--reward_coef', default=1, type=float)
    parser.add_argument('--tensorboard', action="store_true")
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument('--log_dir', default=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    args = parser.parse_args()
    main(args)
