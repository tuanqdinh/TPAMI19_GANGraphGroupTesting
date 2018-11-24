import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--output_path', type=str, default='result/demoCaptionCoCo/', help='path for the script')
parser.add_argument('--data_path', type=str, default='dataset/', help='path for the data')
parser.add_argument('--crop_size', type=int, default=224, help='size for randomly cropping images')
parser.add_argument('--caption_path', type=str, default='dataset/coco/annotations/captions_train2014.json', help='path for train annotation json file')
parser.add_argument('--image_path', type=str, default='dataset/coco/train2014', help='path for train images')
parser.add_argument('--resized_image_folder', type=str, default='resized2014_sm', help='folder name for resized images')
parser.add_argument('--coco_image_check', type=bool, default=False, help='check if coco image folder is complete')
parser.add_argument('--batch_size', type=int, default=64)

parser.add_argument('--log_step', type=int, default=10, help='step size for prining log info')
parser.add_argument('--save_step', type=int, default=1000, help='step size for saving trained models')

# Model parameters
parser.add_argument('--embed_size', type=int, default=256, help='dimension of word embedding vectors')
parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
parser.add_argument('--num_layers', type=int, default=1, help='number of layers in lstm')

parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--learning_rate', type=float, default=0.005)

args = parser.parse_args()
print(args)
