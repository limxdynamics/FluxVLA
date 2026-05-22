# Origin: Modified from
# Upstream-Repo: Physical-Intelligence/openpi
# Upstream-Path: src/openpi/shared/download.py
# Upstream-Ref: main
# SPDX-License-Identifier: Apache-2.0
# Notes: Attribution normalized; no functional change.
import concurrent.futures
import datetime
import getpass
import logging
import os
import pathlib
import re
import shutil
import stat
import time
import urllib.parse

import boto3
import boto3.s3.transfer as s3_transfer
import botocore
import filelock
import fsspec
import tqdm_loggable.auto as tqdm
from types_boto3_s3.service_resource import ObjectSummary

_OPENPI_DATA_HOME = 'OPENPI_DATA_HOME'
logger = logging.getLogger(__name__)


def get_cache_dir() -> pathlib.Path:
    default_dir = '~/.cache/openpi'
    if os.path.exists('/mnt/weka'):
        default_dir = f'/mnt/weka/{getpass.getuser()}/.cache/openpi'
    cache_dir = pathlib.Path(os.getenv(_OPENPI_DATA_HOME,
                                       default_dir)).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    _set_folder_permission(cache_dir)
    return cache_dir


def maybe_download(url: str,
                   *,
                   force_download: bool = False,
                   **kwargs) -> pathlib.Path:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme == '':
        path = pathlib.Path(url)
        if not path.exists():
            raise FileNotFoundError(f'File not found at {url}')
        return path.resolve()

    cache_dir = get_cache_dir()
    local_path = cache_dir / parsed.netloc / parsed.path.strip('/')
    local_path = local_path.resolve()

    invalidate_cache = False
    if local_path.exists():
        if force_download or _should_invalidate_cache(cache_dir, local_path):
            invalidate_cache = True
        else:
            return local_path

    try:
        lock_path = local_path.with_suffix('.lock')
        with filelock.FileLock(lock_path):
            _ensure_permissions(lock_path)
            if invalidate_cache:
                logger.info(f'Removing expired cached entry: {local_path}')
                if local_path.is_dir():
                    shutil.rmtree(local_path)
                else:
                    local_path.unlink()

            logger.info(f'Downloading {url} to {local_path}')
            scratch_path = local_path.with_suffix('.partial')

            if _is_openpi_url(url):
                _download_boto3(
                    url,
                    scratch_path,
                    boto_session=boto3.Session(region_name='us-west-1'),
                    botocore_config=botocore.config.Config(
                        signature_version=botocore.UNSIGNED),
                )
            elif url.startswith('s3://'):
                _download_boto3(url, scratch_path)
            else:
                _download_fsspec(url, scratch_path, **kwargs)

            shutil.move(scratch_path, local_path)
            _ensure_permissions(local_path)

    except PermissionError as e:
        msg = f'Permission error downloading {url}. Try: `rm -rf {local_path}*`'  # noqa: E501
        raise PermissionError(msg) from e

    return local_path


def _download_fsspec(url: str, local_path: pathlib.Path, **kwargs) -> None:
    fs, _ = fsspec.core.url_to_fs(url, **kwargs)
    info = fs.info(url)
    is_dir = info['type'] == 'directory'
    total_size = fs.du(url) if is_dir else info['size']
    with tqdm.tqdm(
            total=total_size, unit='iB', unit_scale=True,
            unit_divisor=1024) as pbar:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(fs.get, url, local_path, recursive=is_dir)
        while not future.done():
            current_size = sum(f.stat().st_size
                               for f in [*local_path.rglob('*'), local_path]
                               if f.is_file())
            pbar.update(current_size - pbar.n)
            time.sleep(1)
        pbar.update(total_size - pbar.n)


def _download_boto3(url: str,
                    local_path: pathlib.Path,
                    *,
                    boto_session=None,
                    botocore_config=None,
                    workers=16):

    def validate_and_parse_url(s3_url: str):
        parsed = urllib.parse.urlparse(s3_url)
        if parsed.scheme != 's3':
            raise ValueError(f'URL must be s3://, got {s3_url}')
        return parsed.netloc, parsed.path.strip('/')

    bucket_name, prefix = validate_and_parse_url(url)
    session = boto_session or boto3.Session()
    s3api = session.resource('s3', config=botocore_config)
    bucket = s3api.Bucket(bucket_name)

    try:
        bucket.Object(prefix).load()
    except botocore.exceptions.ClientError:
        if not prefix.endswith('/'):
            prefix += '/'

    objects = [
        x for x in bucket.objects.filter(Prefix=prefix)
        if not x.key.endswith('/')
    ]
    if not objects:
        raise FileNotFoundError(f'No objects found at {url}')

    total_size = sum(obj.size for obj in objects)
    s3t = _get_s3_transfer_manager(session, workers, botocore_config)

    def transfer(s3obj: ObjectSummary, dest: pathlib.Path, update):
        if dest.exists() and dest.stat().st_size == s3obj.size:
            update(s3obj.size)
            return None
        dest.parent.mkdir(parents=True, exist_ok=True)
        return s3t.download(
            bucket_name,
            s3obj.key,
            str(dest),
            subscribers=[s3_transfer.ProgressCallbackInvoker(update)])

    with tqdm.tqdm(
            total=total_size, unit='iB', unit_scale=True,
            unit_divisor=1024) as pbar:

        def update(size: int):
            pbar.update(size)

        futures = [
            transfer(obj,
                     local_path / pathlib.Path(obj.key).relative_to(prefix),
                     update) for obj in objects
        ]
        for f in futures:
            if f:
                f.result()

    s3t.shutdown()


def _get_s3_transfer_manager(session, workers, botocore_config=None):
    config = botocore.config.Config(max_pool_connections=workers + 2)
    if botocore_config:
        config = config.merge(botocore_config)
    client = session.client('s3', config=config)
    tconf = s3_transfer.TransferConfig(
        use_threads=True, max_concurrency=workers)
    return s3_transfer.create_transfer_manager(client, tconf)


def _set_permission(path: pathlib.Path, target: int):
    if path.stat().st_mode & target == target:
        return
    path.chmod(target)


def _set_folder_permission(path: pathlib.Path):
    _set_permission(path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)


def _ensure_permissions(path: pathlib.Path):

    def setup_path(p: pathlib.Path):
        cache_dir = get_cache_dir()
        rel_path = p.relative_to(cache_dir)
        move_path = cache_dir
        for part in rel_path.parts:
            _set_folder_permission(move_path / part)
            move_path = move_path / part

    def set_file_perm(p: pathlib.Path):
        perm = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH  # noqa: E501
        if p.stat().st_mode & 0o100:
            _set_permission(p,
                            perm | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        else:
            _set_permission(p, perm)

    setup_path(path)
    for root, dirs, files in os.walk(str(path)):
        root_path = pathlib.Path(root)
        for f in files:
            set_file_perm(root_path / f)
        for d in dirs:
            _set_folder_permission(root_path / d)


def _is_openpi_url(url: str) -> bool:
    return url.startswith('s3://openpi-assets/')


def _get_mtime(year: int, month: int, day: int) -> float:
    dt = datetime.datetime(year, month, day, tzinfo=datetime.timezone.utc)
    return time.mktime(dt.timetuple())


_INVALIDATE_CACHE_DIRS: dict[re.Pattern, float] = {
    re.compile('openpi-assets/checkpoints/'): _get_mtime(2025, 2, 3),
}


def _should_invalidate_cache(cache_dir: pathlib.Path,
                             local_path: pathlib.Path) -> bool:
    assert local_path.exists()
    rel = str(local_path.relative_to(cache_dir))
    for pattern, expiry in _INVALIDATE_CACHE_DIRS.items():
        if pattern.match(rel):
            return local_path.stat().st_mtime <= expiry
    return False
