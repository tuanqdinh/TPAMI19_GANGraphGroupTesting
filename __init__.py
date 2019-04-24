import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='wgan-gp')
parser.add_argument('--gan', type=str, default='wgan-conv')
parser.add_argument('--result_path', type=str, default='result', help='output path')
parser.add_argument('--data_path', type=str, default='data/', help='path for data')

parser.add_argument('--log_step', type=int, default=5, help='step size for prining log info')
parser.add_argument('--save_step', type=int, default=10, help='step size for saving trained models')

parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--critic_iters', type=int, default=2)
parser.add_argument('--learning_rate', type=float, default=0.005)
parser.add_argument('--LAMBDA', type=float, default=10)
# Model parameters
parser.add_argument('--signal_size', type=int, default=4225)
parser.add_argument('--img_size', type=int, default=65)
parser.add_argument('--embed_size', type=int, default=100)
parser.add_argument('--hidden_size', type=int, default=1024)

parser.add_argument('--num_workers', type=int, default=4)

### input
parser.add_argument('--off_data', type=int, default=1)
parser.add_argument('--off_model', type=int, default=2)
parser.add_argument('--off_ctrl', type=int, default=1)
parser.add_argument('--off_gan', type=int, default=1)
parser.add_argument('--alpha', type=float, default=0.03)

args = parser.parse_args()
print(args)
