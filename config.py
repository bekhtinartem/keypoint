import torch
# constant paths
ROOT_PATH = ''#../input'
OUTPUT_PATH = ''#../outputs'
# learning parameters
BATCH_SIZE = 4
LR = 0.001
EPOCHS = 30
DEVICE = 'cpu'#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# train/test split
TEST_SPLIT = 0.1
# show dataset keypoint plot
SHOW_DATASET_PLOT = False