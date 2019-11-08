import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# -- Model --
class classifier(torch.nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        self.params = [layer_sizes]

        assert len(layer_sizes) >= 2

        temp_layer_sizes = layer_sizes[:]
        layers = []

        in_layer = temp_layer_sizes.pop(0)
        for out_layer in temp_layer_sizes:
            layers.append(nn.Linear(in_layer, out_layer))
            layers.append(nn.ReLU())
            in_layer = out_layer
        layers.pop()

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        input_size = self.params[0][0] # 28 ** 2
        output_size = self.params[0][-1] # 28 ** 2
        return self.net(x.view(-1,input_size)).view(-1,output_size)

    def classify(self, x):
        return self.forward(x).argmax(dim=1)

    def save_model(self, des=''):
        from os import path
        return torch.save((self.state_dict(), self.params), path.join(path.dirname(path.abspath(__file__)), 'cls'+des+'.th'))

def load_model():
    from os import path
    std, params = torch.load(path.join(path.dirname(path.abspath(__file__)), 'cls.th'), map_location='cpu')
    r = classifier(*params)
    r.load_state_dict(std)
    return r


# -- Training --
def train_classifier(args):
    # Find device and set model to device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # Continue Training
    if args.continue_training: model = load_model().to(device)
    else: model = classifier(args.layers).to(device)
    # Data Loader
    if args.valid_ds == '':
        train_data, valid_data = load_data(args.train_ds, batch_size=args.batch_size, valid=True)
    else:
        train_data = load_data(args.train_ds, batch_size=args.batch_size)
        valid_data = load_data(args.valid_ds, batch_size=args.batch_size)
    # Initialize Model Name
    if args.tensorboard or args.wandb:
        get_model_id(args.description, args.log_dir)
    # TensorBoard logger setup
    train_logger, valid_logger = get_tb_summary_writer(args.log_dir if args.tensorboard else None, valid=False)
    # Wandb setup
    if args.wandb:
        if args.log_dir is not None:
            wandb.init(name=get_model_id(), config=args, dir=args.log_dir, project="simple_nn_classifier")
        else:
            wandb.init(name=get_model_id(), config=args, project="simple_nn_classifier")
        wandb.watch(model)
    num_steps_per_epoch = len(train_data)
    # Criterion
    loss = torch.nn.CrossEntropyLoss()
    # Optimizer
    if args.adam: optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    else:         optimizer = torch.optim.SGD( model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-3)
    # Step Scheduler
    if args.step_schedule: scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=4)

    global_step = 0
    # Epoch Loop
    for i in range(args.epochs):
        model.train()
        # Training Loop
        for img, label in train_data:
            # Set X and Y to Device
            img, label = img.to(device), label.to(device)

            # Compute logit and loss of peak
            loss_val = loss(model(img), label)

            # Plot loss
            if train_logger is not None: train_logger.add_scalar('loss', loss_val, global_step)
            if args.wandb: wandb.log({"epoch":global_step/num_steps_per_epoch, "loss": loss_val}, step=global_step)

            # Grad Step
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1

        # Log training accuracy 
        # Log validation accuracy
        train_accuracy = get_mnist_accuracy(model, train_data)
        if valid_data is not None: valid_accuracy = get_mnist_accuracy(model, valid_data)
        if train_logger is not None:
            train_logger.add_scalar('accuracy', train_accuracy, global_step)
            valid_logger.add_scalar('accuracy', valid_accuracy, global_step)
        if args.wandb:
            wandb.log({"training accuracy":train_accuracy}, step=global_step)
            wandb.log({"validation accuracy":valid_accuracy}, step=global_step)

        # Print if no logger
        if not args.tensorboard and not args.wandb:
            print("Epoch: {:4d} -- Training Loss: {:10.6f}".format(i, loss_val.item()))

        # Step Learning Rate
        if args.step_schedule:
            scheduler.step(total_loss)
            if train_logger is not None:
                train_logger.add_scalar('learning rate', optimizer.param_groups[0]['lr'], global_step)

    # Training Complete
    print("Saved: ", not args.no_save)
    if not args.no_save: model.save_model()
    return model

def load_data(dataset_path, batch_size=32, train=True, valid=False, transform=torchvision.transforms.ToTensor()):
    dataset = torchvision.datasets.MNIST(root=dataset_path, train=train, transform=transform, download=True)
    if batch_size is None: batch_size = len(dataset)
    if valid:
        ds0, ds1 = torch.utils.data.random_split(dataset, [4*len(dataset)//5, len(dataset)-4*len(dataset)//5])
        return DataLoader(ds0, batch_size=batch_size, shuffle=True), DataLoader(ds1, batch_size=batch_size, shuffle=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def get_tb_summary_writer(log_dir='./logs', description=None, valid=True):
    if get_tb_summary_writer.ret is not None: return get_tb_summary_writer.ret
    from os import path
    train_logger, valid_logger = None, None
    if log_dir is not None:
        # Set up SummaryWriters
        if description is None: description = '/' + get_model_id()
        train_logger = tb.SummaryWriter(path.join((log_dir + description), 'train'), flush_secs=1)
        valid_logger = None if not valid else tb.SummaryWriter(path.join((log_dir + description), 'valid'), flush_secs=1)
    get_tb_summary_writer.ret = train_logger, valid_logger
    return get_tb_summary_writer.ret
get_tb_summary_writer.ret = None

def get_model_num(dir='./logs'):
    if get_model_num.ret is not None: return get_model_num.ret

    from os import walk, path, mkdir
    if not path.isdir(dir):
        mkdir(dir)

    def safe_int(i):
        try:
            return int(i)
        except (ValueError, TypeError):
            return -1

    if dir is None: return 0
    model_nums = list(map(safe_int, list(map(lambda x: x.split('-')[0], list(walk(dir))[0][1]))))
    model_nums.sort()
    model_nums.insert(0, -1)
    get_model_num.ret = model_nums[-1] + 1
    return get_model_num.ret
get_model_num.ret = None

def get_model_id(des='', dir='./logs'):
    if get_model_id.ret is not None: return get_model_id.ret
    model_num = get_model_num(dir)
    description = str(model_num)
    if des != '': description += '-' + des
    get_model_id.ret = description
    import os
    os.mkdir(dir+'/'+get_model_id.ret)
    return get_model_id.ret
get_model_id.ret = None


# -- Evaluation --
def eval_mnist(model):
    model.eval()
    test_ds = load_data('../datasets/', batch_size=None, train=False)
    x, y = next(iter(test_ds))
    correct_preds = (model.classify(x) == y).float()
    total_correct = correct_preds.sum()
    total_accuracy = total_correct/len(correct_preds)

    # Class by class accuracy
    print()
    for i in range(10):
        class_i = (y == i).float()
        class_correct = (correct_preds * class_i).sum()
        class_total = class_i.sum()
        class_accuracy = class_correct/class_total
        print("Class: {} | {: 5.1f}%".format(i, class_accuracy*100))
    print("Total Accuracy: {: 5.1f}%".format(total_accuracy*100))

def get_mnist_accuracy(model, dataset):
    model.eval()
    total_correct, total_num = 0, 0
    for x, y in dataset:
        correct_preds = (model.classify(x) == y).float()
        total_correct += correct_preds.sum()
        total_num += len(correct_preds)
    return total_correct/total_num


# -- Commandline --
def print_args(args, parser, defaults=False):
    d_default = parser.get_default('d')
    print("\n------------------------------------------------------------------------------")
    print("|  Option Descriptions   |       Option Strings       |   ",("Values Passed In" if not defaults else " Default Values"))
    print("------------------------------------------------------------------------------")
    print("|Training Dataset Dir:   | '--train_ds'               |  ", (args.train_ds          if not defaults else parser.get_default('train_ds')))
    print("|Validation Dataset Dir: | '--valid_ds'               |  ", (args.valid_ds          if not defaults else parser.get_default('valid_ds')))
    print("|Log Directory:          | '--log_dir'                |  ", (args.log_dir           if not defaults else parser.get_default('log_dir')))
    print("|TensorBoard:            | '--tensorboard'            |  ", (args.tensorboard       if not defaults else parser.get_default('tensorboard')))
    print("|Weights & Bias:         | '--wandb'                  |  ", (args.wandb             if not defaults else parser.get_default('wandb')))
    print("|Description:            | '--description'            |  ", (args.description       if not defaults else parser.get_default('description')))
    print("|Epochs:                 | '-e', '--epochs'           |  ", (args.epochs            if not defaults else parser.get_default('epochs')))
    print("|Batch Size:             | '-b', '--batch_size'       |  ", (args.batch_size        if not defaults else parser.get_default('batch_size')))
    print("|Learning Rate:          | '-l', '--learning_rate'    |  ", (args.learning_rate     if not defaults else parser.get_default('learning_rate')))
    print("|Adam Optimizer:         | '-a', '--adam'             |  ", (args.adam              if not defaults else parser.get_default('adam')))
    print("|Step Schedule:          | '-s', '--step_schedule'    |  ", (args.step_schedule     if not defaults else parser.get_default('step_schedule')))
    print("|No Save:                | '-n', '--no_save'          |  ", (args.no_save           if not defaults else parser.get_default('no_save')))
    print("|Checkpoints:            | '-c', '--checkpoints'      |  ", (args.checkpoints       if not defaults else parser.get_default('checkpoints')))
    print("|Continue Training:      | '--continue_training'      |  ", (args.continue_training if not defaults else parser.get_default('continue_training')))
    print("|Hidden Layer:           | '-i', '--layers'           |  ", (args.layers            if not defaults else parser.get_default('layers')))
    print("------------------------------------------------------------------------------\n")

def parse_layers_string(layers):
    return list(map(int, layers.split(' ')))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--train_ds', type=str, default='../datasets/')
    parser.add_argument('--valid_ds', type=str, default='')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('-T','--tensorboard', action='store_true', default=False)
    parser.add_argument('-W','--wandb', action='store_true', default=False)
    parser.add_argument('--description', type=str, default='')
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-l', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-a', '--adam', action='store_true', default=False)
    parser.add_argument('-s', '--step_schedule', action='store_true', default=False)
    parser.add_argument('-n', '--no_save', action='store_true', default=False)
    parser.add_argument('-c', '--checkpoints', action='store_true', default=False)
    parser.add_argument('--continue_training', action='store_true', default=False)
    parser.add_argument('-i', '--layers', type=str, default="256 128")
    parser.add_argument('-h', '--help', action='store_true', default=False)
    args = parser.parse_args()

    if args.help:
        print_args(args, parser, defaults=True)
        exit() 
    if args.tensorboard:
        import torch.utils.tensorboard as tb
        from os import walk 
    if args.wandb:
        import wandb 

    args.layers = parse_layers_string(args.layers)
    args.layers.insert(0, 28**2)
    args.layers.append(10)
    
    print_args(args, parser)
    model = train_classifier(args)
    eval_mnist(model)
