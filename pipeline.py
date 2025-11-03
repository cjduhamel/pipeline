import argparse
import os
from utils.lib_pipe import start_pipeline

# Import all necessary modules

from dataset_retrievers.example_retrievers import load_example_wine
from preprocessors.example_processors import example_wine_preprocessor
from architectures.example_architectures import ExampleArchitecture
from trainers.example_trainers import ExampleTrainer

#MAPS from string names to classes
#maps from data retrieval/generation name to function
DATA_MAP = {
     "load_example_wine": load_example_wine
}
# Maps from preprocessor name to preprocessor function
PREPROCESSOR_MAP = {
    "example_wine_preprocessor": example_wine_preprocessor
}
# Maps from model name to model class
MODEL_MAP = {
    "ExampleArchitecture": ExampleArchitecture
}
# maps from trainer name to trainer class
TRAINER_MAP = {
    "ExampleTrainer": ExampleTrainer
}


#use argparse to get command line arguments
parser = argparse.ArgumentParser(description="Run grid search training pipeline")

# Optional positional arguments with defaults
parser.add_argument("config_file", nargs="?", default="noise_comp_config.json", help="Path to config JSON")
parser.add_argument("-m", "--models", action="store_true", help="Keep trained models and their relevant information") # not implemented
parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")


args = parser.parse_args()

start_pipeline(args.config_file, DATA_MAP, PREPROCESSOR_MAP, MODEL_MAP, TRAINER_MAP, save_model = args.models)