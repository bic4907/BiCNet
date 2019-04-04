from env.make_env import make_env
import argparse, datetime
from tensorboardX import SummaryWriter
import numpy as np

def main(args):

    env = make_env('simple_spread')

    kwargs = dict()
    kwargs['config'] = args


    if args.tensorboard:
        writer = SummaryWriter(log_dir='runs/'+args.log_dir)

    while True:

        env.reset()
        while True:


            output, (hidden, cell) = self(input, (hidden, cell))


            env.step(np.random.rand(3, 5))
            env.render(mode='rgb_array')






    if args.tensorboard:
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_episodes', default=25000, type=int)
    parser.add_argument('--episode_length', default=25, type=int)
    parser.add_argument('--memory_length', default=int(1e6), type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument('--tau', default=0.01, type=float)
    parser.add_argument('--gamma', default=0.95, type=float)
    parser.add_argument('--use_cuda', default=True, type=bool)
    parser.add_argument('--a_lr', default=0.01, type=float)
    parser.add_argument('--c_lr', default=0.01, type=float)
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--render', default=False, type=bool)
    parser.add_argument('--ou_theta', default=0.15, type=float)
    parser.add_argument('--ou_mu', default=0.0, type=float)
    parser.add_argument('--ou_sigma', default=0.2, type=float)
    parser.add_argument('--epsilon_decay', default=600000, type=int)
    parser.add_argument('--reward_coef', default=1, type=float)
    parser.add_argument('--tensorboard', default=False, type=bool)
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument('--log_dir', default=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    args = parser.parse_args()
    main(args)
