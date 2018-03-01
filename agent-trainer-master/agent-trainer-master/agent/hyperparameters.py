from collections import namedtuple


class GenericHyperparameters(object):
    #训练次数？？？？
    NUM_EPISODES_TO_TRAIN = 75000
    #运行一次保存一次
    EPISODE_SAVE_STEP = 1
    #
    EPISODE_TRAINED_PLAY_METRICS_GATHER_STEP = 1
    #RAM（random access memory）即随机存储内存 的最大记忆池容量
    MAX_REPLAY_MEMORIES_IN_RAM = 129000
    #每个文件最大记忆容量
    MAX_REPLAY_MEMORIES_PER_FILE = 5000

    AGENT_HISTORY_LENGTH = 4

    #反应时间（毫秒级）
    REACTION_TIME_MILISECONDS = 450
    #每秒游戏画面数
    GAME_FRAMES_PER_SECOND = 30
    #Round函数返回一个数值，该数值是按照指定的小数位数进行四舍五入运算的结果。可是当保留位跟着的是5，有可能进位，也有可能舍去，机会各50%。
    #此计算相当于 每毫秒游戏画面帧数 向下取整 作为 每两个连续动作之间 间隔的画面帧数
    FRAMES_SKIPPED_UNTIL_NEXT_ACTION = int(round((REACTION_TIME_MILISECONDS * GAME_FRAMES_PER_SECOND) / 1000))

    REPLAY_MEMORIES_MINIMUM_SIZE_FOR_LEARNING = int(round(50000 / AGENT_HISTORY_LENGTH))
    REPLAY_MEMORIES_RECENT_SAMPLE_SPAN = int(round(1000000 / AGENT_HISTORY_LENGTH))
    REPLAY_MEMORIES_TRAIN_SAMPLE_SIZE = 32

    EXPLORATION_INITIAL_EPSILON = 1.0
    EXPLORATION_FINAL_EPSILON = 0.1
    EXPLORATION_EPSILON_FULL_DEGRADATION_AT_STEP = int(round(1000000 / AGENT_HISTORY_LENGTH))

    Q_UPDATE_DISCOUNT_FACTOR = 0.99

    MAXIMUM_NO_ACTIONS_BEGGINING_EPISODE = 30

class QNetworkHyperparameters(object):
    METRICS_SAVE_STEP = 1000
    SGD_BATCH_SIZE = 32
    LEARNING_RATE_INITIAL = 0.00025
    LEARNING_RATE_FINAL = 0.00001
    LEARNING_RATE_DECAY_STEP = 50100
    LEARNING_RATE_FINAL_AT_STEP = 701000
    RMS_DECAY = 0.9
    RMS_MOMENTUM = 0.95
    RMS_EPSILON = 0.01
    NUM_STEPS_ASSIGN_TRAIN_TO_FORWARD_PROP_GRAPH = 10000

ImageDescription = namedtuple("ImageDescription", ["num_channels"])
class ImageType(object):
    RGB = ImageDescription(num_channels=3)  # Red + Green + Blue channels (color image)
    Y = ImageDescription(num_channels=1)    # Luminance channel (greyscale image)
    RGBY = ImageDescription(num_channels=4) # RGB + Luminance

class PreprocessorHyperparameters(object):
    OUTPUT_WIDTH = 80
    OUTPUT_HEIGHT = 80
    #处理成灰度图？？？？？
    OUTPUT_TYPE = ImageType.Y
    OUTPUT_NUM_CHANNELS = OUTPUT_TYPE.num_channels
