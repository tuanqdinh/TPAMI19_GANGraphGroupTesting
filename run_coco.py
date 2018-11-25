import os, platform
import pickle
import torch.nn as nn
from time import gmtime, strftime
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from dataset.coco import *
from models.dmip_coco import *
from __init__ import args


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# data preparation
if not os.path.exists(args.data_path):
    os.makedirs(args.data_path)

# Create model directory
model_path = os.path.join(args.output_path, 'snapshot')
if not os.path.exists(model_path):
    os.makedirs(model_path)

data_loader = get_loader(img_dir, args.caption_path, vocab,
                         transform, args.batch_size,
                         shuffle=True, num_workers=args.num_workers,
                         image_check=args.coco_image_check)

# Build the models
model = CocoDMIP(hidden_size=args.hidden_size,
                vocab_size=vocab.idx,
                embed_size=args.embed_size,
                num_layers=args.num_layers).to(device)
# from IPython import embed; embed()

# Loss and optimizer
captions_criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

logf = open(os.path.join(args.output_path, "log%s.txt" % strftime("%Y-%m-%d-%H:%M:%S", gmtime())), 'w')




# Train the models
total_step = len(data_loader)
# from IPython import embed; embed()

for epoch in range(args.num_epochs):
    for i, (images, captions, lengths) in enumerate(data_loader):
        # Set mini-batch dataset
        images = images.to(device)
        captions = captions.to(device)

        # Forward, backward and optimize
        mine_score_map = model(images, captions, lengths)
        mine_score = mine_score_map.mean()**2

        optimizer.zero_grad()
        (-mine_score).backward()
        optimizer.step()

        # Print log info
        if i % args.log_step == 0:
            log('Epoch [{}/{}], Step [{}/{}], MineScore: {:.4f}'
                  .format(epoch, args.num_epochs, i, total_step, mine_score.item()))
            # Save the model checkpoints
        if (i + 1) % args.save_step == 0:
            torch.save(model.state_dict(), os.path.join(model_path, 'coco-{}-{}.pth'.format(epoch + 1, i + 1)))
