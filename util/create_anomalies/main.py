from arguments import get_arg_parser
from create_abnormal_sequences import AbnormalSequences
from integrate_abnormal_objects import DatasetAbnormalAug

if __name__ == '__main__':
    args = get_arg_parser()
    if args.run_type == "create_sequences":
        abnormal_sequences = AbnormalSequences(args)
        abnormal_sequences.create_abnormal_sequences()
    elif args.run_type == "abnormal_objects":
        aug = DatasetAbnormalAug(args)
        aug.do_aug()
