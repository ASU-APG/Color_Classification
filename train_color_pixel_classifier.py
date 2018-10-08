'This script is for pre-train a resnet-50 for attribute grounding'
from __future__ import print_function
from tensorboardX import SummaryWriter
from lib.dataset.Ebay_dataset import EbayColor
from models.resnet import resnet50
from torchvision import transforms
import matplotlib.pyplot as plt
from lib.net_util import *
from parser import *

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"


# Pixel-wise cross-entropy loss for Color Learning
def pixel_loss(conv_feat, mask, label, one_hot, model):
    batch_total = 0
    correct = 0
    batch_loss = 0
    # Iterating through batches
    for b in range(conv_feat.shape[0]):
        # Iterating through pixels embeddings
        for i in range(conv_feat.shape[-2]):
            for j in range(conv_feat.shape[-1]):
                # Penalize the pixels of interests
                if mask[b, 0, i, j] != 0:
                    pixel_feat = conv_feat[b, :, i, j]
                    pixel_feat = pixel_feat.contiguous().view(1, -1)
                    prediction = model.fc(pixel_feat)
                    batch_loss += opts.criterion[1](prediction, one_hot[b])
                    # Computing the precision
                    _, predicted = torch.max(prediction.data, 1)
                    correct += predicted.eq(label[b].data).cpu().sum()
                    batch_total += 1
    batch_loss /= batch_total
    return batch_loss, batch_total, correct


def train_net(net, opts):

    print('training at epoch {}'.format(opts.epoch+1))

    if opts.use_gpu:
        net.cuda()

    net.train(True)
    train_loss = 0
    total_time = 0
    batch_idx = 0
    optimizer = opts.current_optimizer
    end_time = time.time()
    fig = plt.figure()

    for batch_idx, (images, mask, color_onehot, color_label) in enumerate(data_loader):

        images = Variable(images).cuda()
        color_label = Variable(color_label).cuda()
        color_onehot = Variable(color_onehot).cuda().float()
        # Feed Forward to Backbone Net
        conv_feat4, conv_feat2 = net(images)

        # Reshape the Binary Mask
        mask = torch.nn.functional.adaptive_avg_pool2d(mask, (conv_feat2.shape[-1], conv_feat2.shape[-2]))

        # Pixel Penalizing
        loss, batch_total, batch_correct = pixel_loss(conv_feat2, mask, color_label, color_onehot, net)

        opts.correct += batch_correct
        opts.total += batch_total

        train_loss += loss.data[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_idx += 1

        print('MSE: %.8f' % (loss.data))
        print('(random) Precision: %.8f' % (opts.correct/opts.total))
        if batch_idx % 10 == 0:
            writer.add_scalar('MSE Loss', train_loss / (batch_idx + 1), opts.iter_n)
            writer.add_scalar('Precision', opts.correct/opts.total, opts.iter_n)
            opts.iter_n += 10
    train_loss /= (batch_idx + 1)

    opts.train_epoch_logger.log({
        'epoch': (opts.epoch+1),
        'loss': train_loss,
        'time': total_time,
    })

    opts.train_losses.append(train_loss)

    # Save checkpoint.
    net_states = {
        'state_dict': net.state_dict(),
        'epoch': opts.epoch + 1,
        'loss': opts.train_losses,
        'optimizer': opts.current_optimizer.state_dict()
    }

    if opts.epoch % opts.checkpoint_epoch == 0:
        save_file_path = os.path.join(opts.checkpoint_path, 'Color_pretrain_regression_{}.pth'.format(opts.epoch))
        torch.save(net_states, save_file_path)

    print('Batch Loss: %.8f, elapsed time: %3.f seconds.' % (train_loss, total_time))


if __name__ == '__main__':
    opts = parse_opts()
    writer = SummaryWriter()

    if opts.gpu_id >= 0:
        torch.cuda.set_device(opts.gpu_id)
        opts.multi_gpu = False

    torch.manual_seed(opts.seed)
    if opts.use_gpu:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.manual_seed(opts.seed)

    # Loading Data
    print("Preparing EbayColor data set...")
    opts.correct = 0
    opts.total = 0
    size = (512, 512)
    feat_size = (32, 32)
    transform = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])
    data_set = EbayColor('./data/', feat_size, transform)
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=opts.batch_size, shuffle=True)

    if not os.path.exists(opts.result_path):
        os.mkdir(opts.result_path)

    opts.train_epoch_logger = Logger(os.path.join(opts.result_path, 'train.log'),
                                     ['epoch', 'time', 'loss'])
    opts.train_batch_logger = Logger(os.path.join(opts.result_path, 'train_batch.log'),
                                     ['epoch', 'batch', 'loss'])
    opts.test_epoch_logger = Logger(os.path.join(opts.result_path, 'test.log'),
                                    ['epoch', 'time', 'loss'])

    # Model
    print('==> Building model...')

    model = resnet50(False)
    model.fc = torch.nn.Linear(512, 11)

    # Evaluation mode for batch normalization freeze
    # model.eval()
    # for p in model.parameters():
    #     p.requires_grad = True

    # Load Back bone Module
    if opts.resume:
        state_dict = torch.load(opts.resume)['state_dict']
        new_params = model.state_dict()
        new_params.update(state_dict)
        model.load_state_dict(new_params)

    start_epoch = 0
    print('==> model built.')
    opts.criterion = [torch.nn.CrossEntropyLoss(), torch.nn.MSELoss()]

    # Training
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in parameters])
    print(params, 'trainable parameters in the network.')
    set_parameters(opts)
    opts.iter_n = 0

    for epoch in range(start_epoch, start_epoch+opts.n_epoch):
        opts.epoch = epoch
        if epoch is 0:
            params = filter(lambda p: p.requires_grad, model.parameters())
            opts.current_optimizer = opts.optimizer(params, lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)

        elif (epoch % opts.lr_adjust_epoch) == 0 and epoch is not 0:
            opts.lr /= 10
            params = filter(lambda p: p.requires_grad, model.parameters())
            opts.current_optimizer = opts.optimizer(params, lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)

        train_net(model, opts)

    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()