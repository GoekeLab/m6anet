from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from m6anet.scripts import dataprep, inference, train, \
        compute_norm_factors, convert


modules = ['dataprep', 'inference', 'train', 'compute_norm_factors', 'convert']

__version__ = "2.1.0"


def main():
    parser = ArgumentParser(prog='m6anet',
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('-v', '--version', action='version',
                        version='%(prog)s {version}'.format(version=__version__))

    subparsers = parser.add_subparsers(
        title='subcommands', description='valid commands',
        help='additional help', dest='command'
    )
    subparsers.required = True

    for module in modules:
        mod = globals()[module]
        p = subparsers.add_parser(module, parents=[mod.argparser()])
        p.set_defaults(func=mod.main)

    args = parser.parse_args()
    args.func(args)
