import argparse


def get_arg_parser():
    parser = argparse.ArgumentParser('MAE abnormal', add_help=False)
    parser.add_argument("--ubnormal_path", type=str, default="/media/alin/New Volume/UBNormal")
    parser.add_argument("--input_dataset", type=str, default="/home/alin/datasets/Avenue_Dataset/Avenue Dataset")
    parser.add_argument("--output_dataset", type=str, default="/media/alin/New Volume/Avenue_aug_abnormal_masks")
    parser.add_argument("--run_type", type=str, default="abnormal_objects")
    parser.add_argument('--target_size', default=(640, 320), type=int, help='images input size')
    args = parser.parse_args()
    return args
