import logging
import torch

logger = logging.getLogger(__name__)

def print_key_pairs(v, title="Parameters", print_function=None):
    """
    Dump key/value pairs to print_function
    :param v:
    :param title:
    :param print_function:
    :return:
    """
    items = v.items() if type(v) is dict else v
    print_function("=" * 40)
    print_function(title)
    print_function("=" * 40)
    for key,value in items:
        print_function("{:<15}: {:<10}".format(key, value if value is not None else "None"))
    print_function("-" * 40)

# Modified from ACRED: https://github.com/yuhaozhang/tacred-relation/blob/master/utils/torch_utils.py
### model IO
def save(model, optimizer, epoch, filename):
    params = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch
    }
    try:
        torch.save(params, filename)
    except BaseException:
        logger.warning("[ Warning: model saving failed. ]")

def load(model, optimizer, filename):
    try:
        dump = torch.load(filename)
    except BaseException:
        print("[ Fail: model loading failed. ]")
    if model is not None:
        model.load_state_dict(dump['model'])
    if optimizer is not None:
        optimizer.load_state_dict(dump['optimizer'])
    return model, optimizer

def load_config(filename):
    try:
        dump = torch.load(filename)
    except BaseException:
        raise ValueError("[ Fail: model loading failed. ]")
    return dump['config']
