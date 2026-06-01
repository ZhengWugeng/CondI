import argparse
import importlib

def read_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', help='name of the benchmark;', type=str, default='mnist_classification')
    parser.add_argument('--dist', help='type of distribution;', type=int, default=0)
    parser.add_argument('--skew', help='the degree of niid;', type=float, default=0)
    parser.add_argument('--num_clients', help='the number of clients;', type=int, default=100)
    parser.add_argument('--seed', help='random seed;', type=int, default=2026)
    parser.add_argument('--missing', help='missing-modality clients;', action='store_true', default=False)
    parser.add_argument('--subset', help='subset of dataset only', action='store_true', default=False)
    parser.add_argument('--imputation', help='use pretrained model to impute modalities', action='store_true', default=False)
    parser.add_argument('--sample_missing_ratio', help='missing ratio inside each sample', type=float, default=0.5)
    parser.add_argument('--client_visible_modalities', help='number of visible modalities per client', type=int, default=8)
    
    try: option = vars(parser.parse_args())
    except IOError as msg: parser.error(str(msg))
    return option

if __name__ == '__main__':
    option = read_option()
    print(option)
    TaskGen = getattr(importlib.import_module('.'.join(['benchmark', option['benchmark'], 'core'])), 'TaskGen')
    generator = TaskGen(
        dist_id = option['dist'],
        skewness = option['skew'],
        num_clients=option['num_clients'],
        seed = option['seed'],
        missing=option['missing'],
        imputation=option['imputation'],
        sample_missing_ratio=option.get('sample_missing_ratio', 0.5),
        client_visible_modalities=option.get('client_visible_modalities', 8)
        # subset=option['subset']
    )
    generator.run()
