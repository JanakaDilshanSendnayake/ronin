"""
Arrange the RoNIN dataset downloaded from FRDR (https://doi.org/10.20383/102.0543) into the flat layout
expected by the training / testing scripts.

The FRDR download splits sequences across four folders:
    Data/train_dataset_1/<seq>, Data/train_dataset_2/<seq>,
    Data/seen_subjects_test_set/<seq>, Data/unseen_subjects_test_set/<seq>
and these folders do not match lists/*.txt (e.g. a018_1, a034_2, a043_3 are in list_train.txt but stored in
seen_subjects_test_set). The scripts join --root_dir / --data_dir directly with the sequence name, so all
sequences must sit in one directory:
    <out_dir>/<seq>/data.hdf5, <out_dir>/<seq>/info.json

Modes:
    link - (default) create a link per sequence, no extra disk space. On Windows a directory junction is used,
           which does not need admin rights or developer mode. Elsewhere a symlink is used.
    move - move the sequence folders (fast on the same drive, but changes the original download).
    copy - copy the sequence folders (needs ~19GB extra space).

The split lists in lists/*.txt are checked against the result, and copies containing only the available
sequences are written to <out_dir>/lists (the original lists are left unchanged).

Example:
    python arrange_frdr_dataset.py --src <path-to-FRDR-download> --out_dir <path-to-output-folder>
    python ronin_resnet.py --mode train --train_list <path-to-output-folder>/lists/list_train.txt
        --val_list <path-to-output-folder>/lists/list_val.txt --root_dir <path-to-output-folder>
        --out_dir <path-to-model-output>
"""

import argparse
import os
import shutil
import subprocess
import sys
from os import path as osp

_REPO_LISTS = osp.join(osp.dirname(osp.abspath(__file__)), '..', '..', 'lists')


def find_sequences(src, max_depth=3):
    """Return {sequence_name: path} for every folder under src containing data.hdf5 and info.json."""
    sequences, duplicates = {}, []
    src_depth = osp.abspath(src).rstrip(os.sep).count(os.sep)
    for root, dirs, files in os.walk(src):
        if 'data.hdf5' in files and 'info.json' in files:
            name = osp.basename(root)
            if name in sequences:
                duplicates.append((name, sequences[name], root))
            else:
                sequences[name] = root
            dirs[:] = []
        elif osp.abspath(root).count(os.sep) - src_depth >= max_depth:
            dirs[:] = []
    return sequences, duplicates


def make_link(src, dst):
    if os.name == 'nt':
        try:
            import _winapi
            _winapi.CreateJunction(osp.abspath(src), osp.abspath(dst))
        except (ImportError, AttributeError, OSError):
            subprocess.run(['cmd', '/c', 'mklink', '/J', osp.abspath(dst), osp.abspath(src)],
                           check=True, stdout=subprocess.DEVNULL)
    else:
        os.symlink(osp.abspath(src), dst, target_is_directory=True)


def read_list(list_path):
    with open(list_path) as f:
        return [s.strip().split(',')[0] for s in f.readlines() if len(s.strip()) > 0 and s[0] != '#']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, required=True,
                        help='Path to the FRDR download (or its Data folder)')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Output folder that will contain one sub-folder per sequence')
    parser.add_argument('--mode', type=str, default='link', choices=['link', 'move', 'copy'])
    parser.add_argument('--lists_dir', type=str, default=_REPO_LISTS,
                        help='Folder with list_*.txt used to verify the result')
    parser.add_argument('--dry_run', action='store_true', help='Only print what would be done')
    args = parser.parse_args()

    sequences, duplicates = find_sequences(args.src)
    if duplicates:
        for name, first, second in duplicates:
            print('Duplicate sequence {}: {} and {}'.format(name, first, second))
        sys.exit('Found duplicate sequence names. Aborting.')
    if len(sequences) == 0:
        sys.exit('No sequences (folders with data.hdf5 and info.json) found under ' + args.src)
    print('Found {} sequences under {}'.format(len(sequences), args.src))

    out_dir = osp.abspath(args.out_dir)
    if any(osp.commonpath([out_dir, osp.abspath(p)]) == out_dir for p in sequences.values()):
        sys.exit('--out_dir must not contain the source sequences.')
    if not args.dry_run:
        os.makedirs(out_dir, exist_ok=True)

    action = {'link': make_link, 'move': shutil.move, 'copy': shutil.copytree}[args.mode]
    done, skipped = 0, 0
    for name in sorted(sequences):
        dst = osp.join(out_dir, name)
        if osp.lexists(dst):
            skipped += 1
            continue
        if args.dry_run:
            print('{} {} -> {}'.format(args.mode, sequences[name], dst))
        else:
            action(sequences[name], dst)
        done += 1
    print('{}: {} sequences, {} already present (skipped)'.format(args.mode, done, skipped))

    # Verify the split lists against the output folder. Some listed sequences belong to the unpublished part of
    # the dataset (e.g. a007_3 in list_train.txt), so filtered copies of the lists are written to <out_dir>/lists.
    if osp.isdir(args.lists_dir):
        lists_out = osp.join(out_dir, 'lists')
        if not args.dry_run:
            os.makedirs(lists_out, exist_ok=True)
        for list_name in sorted(os.listdir(args.lists_dir)):
            if not list_name.endswith('.txt'):
                continue
            names = read_list(osp.join(args.lists_dir, list_name))
            if args.dry_run:
                available = [n for n in names if n in sequences]
            else:
                available = [n for n in names if osp.isfile(osp.join(out_dir, n, 'data.hdf5'))]
            missing = [n for n in names if n not in available]
            print('{:24s} {:3d} listed, {:3d} available, {} missing{}'.format(
                list_name, len(names), len(available), len(missing), (': ' + ', '.join(missing)) if missing else ''))
            if not args.dry_run:
                with open(osp.join(lists_out, list_name), 'w') as f:
                    f.write('\n'.join(available) + '\n')
        if not args.dry_run:
            print('Filtered lists written to ' + lists_out)
