from importsfolder import *

class config:
    TRAINING_FILE = "../input/petfinder-pawpularity-score/train.csv"
    TRAINING_IMAGE_PATH = "../input/petfinder-pawpularity-score/train/"
    DEVICE = torch.device("cuda")
    TRAIN_BATCH_SIZE = 1
    VALID_BATCH_SIZE = 1
    EPOCHS = 5
    
class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count