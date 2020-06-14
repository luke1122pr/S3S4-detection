#!/usr/bin/env python3
import argparse
import subprocess
import time
import wandb
from generate_sweeps import get_all_sweeps
import pprint
import sys

import libtmux

import better_exceptions; better_exceptions.hook()

ENTITY = 'toosyou'

TMUX_SERVER = libtmux.Server()

def create_sweep(wandb_project, sweep_config):
    wandb_project = wandb_project
    sweep_id = wandb.sweep(sweep_config, entity=ENTITY, 
                        project=wandb_project)
    return '{}/{}/{}'.format(ENTITY, wandb_project, sweep_id)

def run_cmds_in_tmux(session_name, cmds):
    # create session
    session = TMUX_SERVER.new_session(session_name, kill_session=True)
    window = session.select_window('0')

    for cmd in cmds:
        window.split_window(shell=cmd)
    
    window.select_layout('even-vertical')
    return session

def run_sweep_in_tmux(session_name, sweep_id, n_agents, n_runs):
    cmds = list()
    for i in range(n_agents):
        cmd = 'CUDA_VISIBLE_DEVICES="{}" pipenv run wandb agent --count {} {}'.format(
                i % 2, n_runs // n_agents, sweep_id)
        cmds.append(cmd)
    return run_cmds_in_tmux(session_name, cmds)

def run_evaluation(task, n_model, sweep_id):
    subprocess.run(['pipenv', 'run', 
                        'python3', './{}/evaluation.py'.format(task), 
                        '-n', '{}'.format(n_model), '-m', 'best_val_loss', sweep_id], check=True)

def wait_for_finish(session):
    window = session.select_window('0')
    while True:
        try:
            time.sleep(10)
            if len(window.list_panes()) <= 1:
                break

        except KeyboardInterrupt:
            ans = input('Exit? [y/N]:')
            if ans in ['y', 'Y']:
                session.kill_session()
                sys.exit(0)

def run_experiment(wandb_project, sweep_config, session_name, n_agents, n_runs):
    sweep_id = create_sweep(wandb_project, sweep_config)

    session = run_sweep_in_tmux(session_name, sweep_id, n_agents, n_runs)
    wait_for_finish(session)
    session.kill_session()

    task = sweep_config['name'].split('/')[0]
    for n_model in [1, 3, 5]:
        run_evaluation(task, n_model, sweep_id)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run both abnormal detection and hazard prediction experiments with all possible setting.')
    parser.add_argument('-r', '--runs', type=int, nargs='?', default=200,
                        help='Total number of runs for all agents to run.')
    parser.add_argument('-a', '--agents', type=int, nargs='?', default=4,
                        help='Number of agents to run experiments')
    parser.add_argument('-f', '--filters', type=str, nargs='*',
                        help='Only run sweeps with certain strings in the sweep name.')
    parser.add_argument('-e', '--excepts', type=str, nargs='*',
                        help='Only run sweeps without certain strings in the sweep name.')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='Print every setting in sweeps when dryrun.')
    parser.add_argument('--nodryrun', action='store_true', default=False,
                        help='Really run experiments.')
    
    args = parser.parse_args()
    sweeps = get_all_sweeps()

    # filter sweeps to run
    if args.filters:
        print('With only:', args.filters)
        for f in args.filters:
            sweeps = [sweep for sweep in sweeps if f in sweep['name']]

    if args.excepts:
        print('Except:', args.excepts)
        for e in args.excepts:
            sweeps = [sweep for sweep in sweeps if e not in sweep['name']]

    if args.nodryrun:
        for sweep in sweeps:
            pprint.pprint(sweep)
            wandb_project = 'ekg-' + sweep['name'].split('/')[0]
            run_experiment(wandb_project, sweep, wandb_project, args.agents, args.runs)
    else: # dryrun
        if args.verbose:
            pprint.pprint(sweeps)
        else:
            print('Only print names of sweeps, use --verbose to print detail infomation.')
            pprint.pprint(list(map(lambda s: s['name'], sweeps)))
        print('Dryrun! Use --nodryrun to really run the experiments.')