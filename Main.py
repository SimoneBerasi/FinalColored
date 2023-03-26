import argparse
import configparser
import os

import Training
import Prediction
from DataLoader import *
from Evaluation import Evaluation


def check_action(value):
    if value != "training" and value != "evaluation" and value != 'prediction':
        raise argparse.ArgumentTypeError("Invalid action argument")
    return value


def check_config_file(value):
    if os.path.exists(value):
        if value.endswith('.ini'):
            return value
    raise argparse.ArgumentTypeError("Invalid file path")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', action="store", help='Action to perform: training, prediction or evaluation',
                        dest="action", type=check_action, required=True)
    parser.add_argument('-f', action="store", help='Configuration file path', dest="file", type=check_config_file,
                        required=True)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()

    config = configparser.ConfigParser()
    config.read(args.file)

    if (args.action == 'training'):

        Training.loss_type = config['LOSSFUNCTION']['LossType'];
        Training.window_size = int(config['LOSSFUNCTION']['WindowSize'])
        Training.scales = int(config['LOSSFUNCTION']['Scales']);
        Training.orients = int(config['LOSSFUNCTION']['Orientations'])

        Training.steps_per_epoch = int(config['TRAINING']['Steps_per_epoch'])
        Training.generator_batch_size = int(config['TRAINING']['Generator_batch_size'])
        Training.image_size = int(config['TRAINING']['Image_size'])
        Training.n_patches = int(int(config['TRAINING']['NPatches']));
        Training.patch_size = int(config['TRAINING']['PatchSize'])
        Training.batch_size = int(config['TRAINING']['BatchSize']);
        Training.epoch = int(config['TRAINING']['Epochs'])
        Training.lr = float(config['TRAINING']['LearningRate']);
        Training.decay_step = int(config['TRAINING']['DecayStep'])
        Training.decay_fac = float(config['TRAINING']['DecayFactor']);
        Training.save_period = int(config['TRAINING']['SavePeriod'])
        Training.training_dataset_path = config['TRAINING']['Training_dataset_path']

        Training.train()

    elif (args.action == 'prediction'):
        Prediction.weights_file = config['PREDICTION']['WeightsFile'];
        Prediction.anomaly_metrics = config['PREDICTION']['AnomalyMetrics']
        Prediction.ae_patch_size = int(config['PREDICTION']['PatchSize']);
        Prediction.test_dir = config['PREDICTION']['Test_dir']
        Training.ae_stride = int(config['PREDICTION']['Stride'])
        Prediction.ae_batch_splits = int(config['PREDICTION']['BatchSplits']);
        Training.invert_reconstruction = bool(config['PREDICTION']['InvertReconstruction'])
        Training.fpr_value = float(config['PREDICTION']['ThresholdFPR']);


        Prediction.predict()

    else:
        Evaluation.anomaly_maps_dir = config['EVALUATION']['anomaly_maps_dir']
        Evaluation.pro_integration_limit = float(config['EVALUATION']['pro_integration_limit'])
        Evaluation.evaluated_objects = config['EVALUATION']['evaluated_objects']
        Evaluation.output_dir = config['EVALUATION']['output_dir']
        Evaluation.dataset_base_dir = config['EVALUATION']['dataset_base_dir']

        Evaluation.evaluation()