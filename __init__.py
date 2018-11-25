import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='wgan-gp')
parser.add_argument('--dataset', type=str, default='mesh')
parser.add_argument('--output_path', type=str, default='results', help='path for the script')
parser.add_argument('--data_path', type=str, default='data/adni_data_4k.mat', help='path for the data')

parser.add_argument('--log_step', type=int, default=5, help='step size for prining log info')
parser.add_argument('--save_step', type=int, default=10, help='step size for saving trained models')

parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--critic_iters', type=int, default=2)
parser.add_argument('--learning_rate', type=float, default=0.005)
parser.add_argument('--LAMBDA', type=float, default=10)

parser.add_argument('--laplacian', type=bool, default=True)
parser.add_argument('--alpha', type=float, default=1)
# Model parameters
parser.add_argument('--signal_size', type=int, default=4225)
parser.add_argument('--img_size', type=int, default=65)
parser.add_argument('--embed_size', type=int, default=100)
parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--num_layers', type=int, default=1, help='number of layers in lstm')

parser.add_argument('--num_workers', type=int, default=4)

args = parser.parse_args()
print(args)
