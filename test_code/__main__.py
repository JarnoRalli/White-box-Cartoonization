# This file acts as the entry point, so that the user can easily call cartoonize from the package
from . import cartoonize
import sys
import argparse

if sys.version_info < (3, 9):
    # importlib.resources either doesn't exist or lacks the files() function, so use the PyPI version:
    import importlib_resources
else:
    # importlib.resources has files(), so use that:
    import importlib.resources as importlib_resources

def main():
    model_path = importlib_resources.files("whiteboxcartoon") / "saved_models"

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-m", "--model_path", help="model path", type=cartoonize.path_is_directory, default=str(model_path))
    argParser.add_argument("-i", "--input", help="path to a directory or a single image to process", type=cartoonize.path_exists)
    argParser.add_argument("-o", "--output_folder", help="path to output folder", type=cartoonize.path_is_directory)
    argParser.add_argument("-r", "--rho", help="image sharpness", type=float, default=1.0)
    args = argParser.parse_args()

    try:
        cartoonize.cartoonize(args.input, args.output_folder, args.model_path, args.rho)
    except Exception as error:
        print(error)
        sys.exit(1)
    else:
        sys.exit(0)
