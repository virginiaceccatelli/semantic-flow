import argparse


def main():
    parser = argparse.ArgumentParser(description='Real BugsInPy Boolean-state probing and intervention')
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare', help='CPU/network: reproduce real tests and collect paired cases')
    prepare.add_argument('--data', default='data/bugstate')
    prepare.add_argument('--projects', default='', help='Comma-separated project names; default all')
    prepare.add_argument('--max-bugs-per-project', type=int, default=0, help='0 means all')
    prepare.add_argument('--test-timeout', type=int, default=180)
    run = commands.add_parser('run', help='GPU: activations, probes, interventions, plots and report')
    run.add_argument('--data', default='data/bugstate')
    run.add_argument('--output', default='results/bugstate/qwen3-4b')
    run.add_argument('--model', default='Qwen/Qwen3-4B')
    run.add_argument('--revision', default='main')
    run.add_argument('--seed', type=int, default=42)
    run.add_argument('--max-input-tokens', type=int, default=8192)
    run.add_argument('--max-new-tokens', type=int, default=4096)
    run.add_argument('--random-repeats', type=int, default=3)
    report = commands.add_parser('report', help='CPU: regenerate plots and report from saved results')
    report.add_argument('--output', default='results/bugstate/qwen3-4b')
    args = parser.parse_args()
    if args.command == 'prepare':
        from .data import prepare as action
    elif args.command == 'run':
        if args.random_repeats < 1:
            parser.error('--random-repeats must be at least 1')
        from .experiment import experiment as action
    else:
        from .report import report as action
    action(args)


if __name__ == '__main__':
    main()
