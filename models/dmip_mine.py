import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.models as models
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


normalizer = lambda x: x / (x.norm(p=2, dim=1, keepdim=True).expand_as(x) + 1e-8)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BasicBlock(nn.Module):

    def __init__(self, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out

class Bottleneck(nn.Module):

    def __init__(self, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv3 = nn.Conv2d(1, 1, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        return out

class BBoxResNet(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        super(BBoxResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=7, stride=2, padding=3, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, layers[0])
        self.layer2 = self._make_layer(block, layers[1], stride=2)
        self.layer3 = self._make_layer(block, layers[2], stride=2)
        self.layer4 = self._make_layer(block, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.fill_(1/(m.kernel_size[0] * m.kernel_size[1]))

    def _make_layer(self, block, blocks, stride=1):
        layers = []
        layers.append(block(stride))
        for i in range(1, blocks):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, boxes):
        x = torch.zeros((len(boxes), 1, 224, 224))
        for bi, box in enumerate(boxes):
            r_x, r_y, r_w, r_h = box
            x[bi, 0, r_x:r_x+r_w, r_y:r_y+r_h] = 1.
        with torch.no_grad():
            x = self.conv1(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            xmin = torch.Tensor(np.min(x.data.numpy(), axis=(1, 2, 3))[:, None, None, None])
            xmax = torch.Tensor(np.max(x.data.numpy(), axis=(1, 2, 3))[:, None, None, None])
            x = (x - xmin) / (xmax - xmin + 1e-10)
        return x[:, 0, :, :]

def BBoxResnet18():
    model = BBoxResNet(BasicBlock, [2, 2, 2, 2])
    return model

def BBoxResnet50():
    model = BBoxResNet(Bottleneck, [3, 4, 6, 3])
    return model

def BBoxResnet152():
    model = BBoxResNet(Bottleneck, [3, 8, 36, 3])
    return model

class DMIPVGenome(nn.Module):
    def __init__(self, hidden_size, vocab_size, embed_size, num_layers, archCNN='resnet18', archTXT='LSTM'):
        super(DMIPVGenome, self).__init__()
        self.hidden_size = hidden_size
        if archTXT == 'LSTM':
            self.text_encoder = EncoderRNN(
                vocab_size=vocab_size,
                embed_size=embed_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
            )
        elif archTXT == 'Embedding':
            self.text_encoder = EncoderEmbedding(vocab_size, hidden_size)

        self.image_encoder = EncoderCNN(arch=archCNN, hidden_size=hidden_size)

    def forward(self, images, captions, lengths, BBoxMap):
        threshold = np.percentile(BBoxMap.data.cpu().numpy(), 80)
        BBMask = BBoxMap > threshold
        images_fmap = self.image_encoder.forward(images)
        captions_code, txtOutput = self.text_encoder.forward(captions, lengths)
        captions_code_shuffled = captions_code[torch.randperm(len(captions_code)), :]

        pxy = (images_fmap * captions_code[:, :, None, None]).sum(1)
        pxy = pxy/norm(pxy)
        pxpy = (images_fmap * captions_code_shuffled[:, :, None, None]).sum(1)
<<<<<<< HEAD:models/dmip_mine.py
        pxpy = pxpy /norm(pxpy)
        mine_score_map = pxy - torch.log(torch.mean(torch.exp(pxpy), 0)[None,:,:])
=======
>>>>>>> 50fea19b11ee5533d4986bf933945af46270b29e:models/dmip.py

        pos_part = torch.mean(pxy * BBMask.float().to(pxy.device))
        neg_part = torch.log(torch.mean(torch.exp(pxpy * BBMask.float().to(pxy.device))))
        return pos_part, neg_part, txtOutput

    def txt2heatmap(self, images, captions, lengths):
        images_fmap = self.image_encoder.forward(images)
        captions_code, _ = self.text_encoder.forward(captions, lengths)
        fmap = (images_fmap * captions_code[:, :, None, None]).sum(1)
        # fmap = (fmap + 1) / 2
        return fmap

    def word2heatmap(self, images, captions, lengths):
        images_fmap = self.image_encoder.forward(images)
        words_code, _ = self.text_encoder.word_feat(captions, lengths)
        fmap = (images_fmap * words_code[0, 1:-1, :, None, None]).sum(1)
        # fmap = (fmap + 1) / 2
        return fmap



class DMIPCoCo(nn.Module):
    def __init__(self, hidden_size, vocab_size, embed_size, num_layers, archCNN='resnet18', archTXT='LSTM'):
        super(DMIPCoCo, self).__init__()
        self.hidden_size = hidden_size
        if archTXT == 'LSTM':
            self.text_encoder = EncoderRNN(
                vocab_size=vocab_size,
                embed_size=embed_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
            )
        elif archTXT == 'Embedding':
            self.text_encoder = EncoderEmbedding(vocab_size, hidden_size)

<<<<<<< HEAD:models/dmip_mine.py
        self.image_encoder = EncoderCNN(arch=archCNN)
=======
        self.image_encoder = EncoderCNN(arch=archCNN, hidden_size=hidden_size)
>>>>>>> 50fea19b11ee5533d4986bf933945af46270b29e:models/dmip.py

    def forward(self, images, captions, lengths):
        images_fmap = self.image_encoder.forward(images)
        captions_code, txtOutput = self.text_encoder.forward(captions, lengths)
        captions_code_shuffled = captions_code[torch.randperm(len(captions_code)), :]

        # pred_xy = mine_net(images_fmap, captions_code)
        # pred_xcy = mine_net(images_fmap, captions_code_shuffled)
        # ret = torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_xcy)))
        # mine_loss = -ret
        # pxy = (images_fmap * captions_code[:, :, None, None]).sum(1)
        # pxpy = (images_fmap * captions_code_shuffled[:, :, None, None]).sum(1)
        # torch.mean(pxy), torch.log(torch.mean(torch.exp(pxpy))), txtOutput
        return images_fmap, captions_code, captions_code_shuffled

    def txt2heatmap(self, images, captions, lengths):
        images_fmap = self.image_encoder.forward(images)
        captions_code, _ = self.text_encoder.forward(captions, lengths)
        # fmap = (images_fmap * captions_code[:, :, None, None]).sum(1)
        # fmap = (fmap + 1) / 2
        images_view = images_fmap.permute(2, 3, 0, 1)
        images_avg = images_view.view(-1, images_view.size(2), images_view.size(3))
        return images_avg, captions_code

    def word2heatmap(self, images, captions, lengths):
        images_fmap = self.image_encoder.forward(images)
        words_code, _ = self.text_encoder.word_feat(captions, lengths)
        fmap = (images_fmap * words_code[0, 1:-1, :, None, None]).sum(1)
        # fmap = (fmap + 1) / 2
        return fmap

    def img2txt(self, images, mode='gloabel'):
        images_fmap = self.image_encoder.forward(images)
        images_fmap = images_fmap.mean(2).mean(2)
        start_txt_feat = self.text_encoder.tanh(self.text_encoder.embed(images.new_zeros(20, dtype=torch.long)))
        states = (images_fmap.new_zeros(1, 1, images_fmap.size()[1], requires_grad=False), images_fmap.unsqueeze(0))
        sample_cap_ids = self.text_encoder.sample(start_txt_feat, states=states)
        return sample_cap_ids

class MineNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MineNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(input_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x, y):
        # from IPython import embed; embed()
        h1 = self.relu(self.fc1(x) + self.fc2(y))
        h2 = self.fc3(h1)
        return h2

class EncoderCNN(nn.Module):
    def __init__(self, hidden_size, arch='resnet18'):
        """Load the pretrained ResNet-18 and replace top fc and avgpool layer."""
        super(EncoderCNN, self).__init__()
        if arch == 'resnet18':
            # import util.resnet
            resnet = models.resnet18(pretrained=True)
            # resnet = util.resnet.__dict__['resnet18'](pretrained=True)
        elif arch == 'wideresnet18':
            from wideresnet import resnet18
            resnet = resnet18(pretrained=False)

        modules = list(resnet.children())[:-2]
        self.embedder = nn.Conv2d(512, hidden_size, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.resnet = nn.Sequential(*modules)

    def forward(self, images):
        """Extract feature vectors from input images."""
        features = self.relu(self.embedder(self.resnet(images)))
        features = normalizer(features)
        return features

class EncoderEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(EncoderEmbedding, self).__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.relu = nn.ReLU()
        init.xavier_uniform_(self.embed.weight)

    def forward(self, captions, lengths):
        features = self.embed(captions)
        features = features * ((captions != 0)[:, :, None]).float()
        features = features.sum(1)
        features = self.relu(features)
        features = normalizer(features)
        return features, None

    def word_feat(self, captions, lengths):
        features = self.embed(captions)
        features = self.relu(features)
        features = features / (features.norm(p=2, dim=2, keepdim=True).expand_as(features) + 1e-8)
        return features, None


class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, drop=0.0):
        super(EncoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.drop = nn.Dropout(drop)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers)
        self.features = hidden_size
        self.fc_cap = nn.Linear(hidden_size, vocab_size)

        self._init_lstm(self.lstm.weight_ih_l0)
        self._init_lstm(self.lstm.weight_hh_l0)
        self.lstm.bias_ih_l0.data.zero_()
        self.lstm.bias_hh_l0.data.zero_()

        init.xavier_uniform_(self.embed.weight)

    def _init_lstm(self, weight):
        for w in weight.chunk(4, 0):
            init.xavier_uniform_(w)

    def forward(self, captions, lengths):

        embedded = self.embed(captions)
        tanhed = self.tanh(self.drop(embedded))
        packed = pack_padded_sequence(tanhed, lengths, batch_first=True)
        outputs, (_, c) = self.lstm(packed)
        features = normalizer(c.squeeze(0))
        outputs = self.fc_cap(outputs[0])
        return features, outputs


    def sample(self, features, states=None, max_seg_length=20):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(max_seg_length):
            hiddens, states = self.lstm(inputs, states)  # hiddens: (batch_size, 1, hidden_size)
            outputs = self.fc_cap(hiddens.squeeze(1))  # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)  # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)  # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)  # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)  # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids
