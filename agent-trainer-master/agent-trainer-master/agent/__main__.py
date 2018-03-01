#OS模块简单的来说它是一个Python的系统编程的操作模块，可以处理文件和目录这些我们日常手动需要做的操作。
import os
#sys模块用来处理Python运行时配置以及资源，从而可以与前当程序之外的系统环境交互，如：Python解释器。
import sys
#argparse模块的作用是用于解析命令行参数
import argparse
#Logging provides a set of convenience functions for simple logging usage.
import logging.config
#yaml是一种配置文件类型
import yaml

import agent.config as config
from agent.trainer.session import SessionRunner, SessionMetricsPresenter


def main(args=None):
    check_python_version()
    setup_logging()
    logger = logging.getLogger(__name__)
    try:
        parse_command_line_arguments_and_launch()
    except:
        logger.error('Unexpected error:', exc_info=True)
        raise

def parse_command_line_arguments_and_launch():
    #创建一个解析对象
    parser = argparse.ArgumentParser()
    #向该对象中添加要关注的命令行参数和选项，每一个add_argument方法对应一个要关注的参数或者选项
    parser.add_argument("action", choices=['train-new', 'train-resume', 'play', 'visualize-tsne', 'metrics-show', 'metrics-export'])
    parser.add_argument("-s", type=str, help="session id")
    parser.add_argument("--resultspath", help="root for training result sessions")
    parser.add_argument("--ec2spot", help="use this options if the trainer is executed in a AWS EC2 Spot Instance",
                        action="store_true")
    #调用parse_args（）方法进行解析
    parsed_arguments = parser.parse_args()

    validate_arguments(parsed_arguments)

    if parsed_arguments.resultspath:
        config.train_results_root_folder = parsed_arguments.resultspath

    if parsed_arguments.ec2spot:
        config.trained_using_aws_spot_instance = True

    #开始新的训练
    if parsed_arguments.action == 'train-new':
        SessionRunner(config).train_new()
    elif parsed_arguments.action == 'train-resume':
        SessionRunner(config).train_resume(session_id=parsed_arguments.s)
    elif parsed_arguments.action == 'play':
        SessionRunner(config).play(session_id=parsed_arguments.s)
    elif parsed_arguments.action == 'visualize-tsne':
        SessionRunner(config).play_and_visualize_q_network_tsne(session_id=parsed_arguments.s)
    elif parsed_arguments.action == 'metrics-show':
        SessionMetricsPresenter(config).show(session_id=parsed_arguments.s)
    elif parsed_arguments.action == 'metrics-export':
        SessionMetricsPresenter(config).save_to_image(session_id=parsed_arguments.s)


def validate_arguments(parsed_arguments):
    if (parsed_arguments.action == (
                'train-resume' or 'play' or 'visualize-tsne' or 'metrics-show' or 'metrics-export')) and not parsed_arguments.s:
        raise SystemExit("-s argument (session id) is required for action {0}".format(parsed_arguments.action))


def check_python_version():
    if sys.version_info.major != 2 or sys.version_info.minor < 7:
        raise SystemExit("Python version 2.7 required")


def setup_logging(path='config_logging.yaml', default_level=logging.INFO):
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
        print("Logging configuration \"{0}\" was not found. Using default basic configuration instead".format(path))
        logging.basicConfig(level=default_level)


if __name__ == "__main__":
    main()