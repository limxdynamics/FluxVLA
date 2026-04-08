"""
Script to convert Aloha hdf5 data to the LeRobot dataset v2.1 format.
"""

import argparse
from pathlib import Path

from hdf_to_lerobot_pipeline import port_hdf5
from lerobot.constants import HF_LEROBOT_HOME


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Convert Aloha HDF5 data to LeRobot dataset v2.1 format.',
    )
    parser.add_argument(
        'raw_dir',
        type=Path,
        help='Path to the directory containing HDF5 files')
    parser.add_argument(
        '--repo-id',
        type=str,
        default='0402',
        help='Dataset repository ID (default: 0402)',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Custom output directory (default: HF_LEROBOT_HOME)',
    )
    parser.add_argument(
        '--init-task',
        type=str,
        default=None,
        help='Default task description for frames without annotation',
    )
    parser.add_argument(
        '--robot-type',
        type=str,
        default='aloha',
        choices=['aloha', 'aloha_sim'],
        help='Robot type (default: aloha)',
    )
    parser.add_argument(
        '--depth',
        action='store_true',
        help='Include depth images in the conversion')

    args = parser.parse_args()

    print(f'HDF5 files directory: {args.raw_dir.absolute()}')
    print(f'Output dataset ID: {args.repo_id}')
    print(f'Robot type: {args.robot_type}')
    if args.depth:
        print('Depth conversion: enabled')
    if args.output_dir:
        print(f'Custom output directory: {args.output_dir.absolute()}')
    else:
        print(f'Using default output directory: {HF_LEROBOT_HOME.absolute()}')

    port_hdf5(
        raw_dir=args.raw_dir,
        repo_id=args.repo_id,
        robot_type=args.robot_type,
        mode='video',
        debug_mode=False,
        output_dir=args.output_dir,
        init_task=args.init_task,
        convert_depth=args.depth,
    )


if __name__ == '__main__':
    main()
