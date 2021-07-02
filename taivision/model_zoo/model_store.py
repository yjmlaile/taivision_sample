# https://github.com/dmlc/gluon-cv/blob/master/gluoncv/model_zoo/model_store.py
# pylint: disable=wildcard-import, unused-wildcard-import, line-too-long
"""Model store which provides pretrained models."""
from __future__ import print_function

__all__ = ['get_model_file', 'purge']
import os
import zipfile
import logging
import portalocker

# from ..utils import download, check_sha1

_model_sha1 = {name: checksum for checksum, name in [
    ('cc729d95031ca98cf2ff362eb57dee4d9994e4b2', 'resnet50_v1'),
]}

apache_repo_url = 'https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/'
_url_format = '{repo_url}gluon/models/{file_name}.zip'


def short_hash(name):
    if name not in _model_sha1:
        raise ValueError('Pretrained model for {name} is not available.'.format(name=name))
    return _model_sha1[name][:8]


def get_model_file(name, tag=None, root=os.path.join('~', '.mxnet', 'models')):
    r"""Return location for the pretrained on local file system.

    This function will download from online model zoo when model cannot be found or has mismatch.
    The root directory will be created if it doesn't exist.

    Parameters
    ----------
    name : str
        Name of the model.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Returns
    -------
    file_path
        Path to the requested pretrained model file.
    """
    if 'MXNET_HOME' in os.environ:
        root = os.path.join(os.environ['MXNET_HOME'], 'models')

    use_tag = isinstance(tag, str)
    if use_tag:
        file_name = '{name}-{short_hash}'.format(name=name,
                                                 short_hash=tag)
    else:
        file_name = '{name}-{short_hash}'.format(name=name,
                                                 short_hash=short_hash(name))
    root = os.path.expanduser(root)
    params_path = os.path.join(root, file_name + '.params')
    lockfile = os.path.join(root, file_name + '.lock')
    if use_tag:
        sha1_hash = tag
    else:
        sha1_hash = _model_sha1[name]

    if not os.path.exists(root):
        os.makedirs(root, exist_ok=True)

    with portalocker.Lock(lockfile, timeout=int(os.environ.get('GLUON_MODEL_LOCK_TIMEOUT', 300))):
        if os.path.exists(params_path):
            if check_sha1(params_path, sha1_hash):
                return params_path
            else:
                logging.warning("Hash mismatch in the content of model file '%s' detected. "
                                "Downloading again.", params_path)
        else:
            logging.info('Model file not found. Downloading.')

        zip_file_path = os.path.join(root, file_name + '.zip')
        repo_url = os.environ.get('MXNET_GLUON_REPO', apache_repo_url)
        if repo_url[-1] != '/':
            repo_url = repo_url + '/'
        download(_url_format.format(repo_url=repo_url, file_name=file_name),
                 path=zip_file_path,
                 overwrite=True)
        with zipfile.ZipFile(zip_file_path) as zf:
            zf.extractall(root)
        os.remove(zip_file_path)
        # Make sure we write the model file on networked filesystems
        try:
            os.sync()
        except AttributeError:
            pass
        if check_sha1(params_path, sha1_hash):
            return params_path
        else:
            raise ValueError('Downloaded file has different hash. Please try again.')


def purge(root=os.path.join('~', '.mxnet', 'models')):
    r"""Purge all pretrained model files in local file store.

    Parameters
    ----------
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    root = os.path.expanduser(root)
    files = os.listdir(root)
    for f in files:
        if f.endswith(".params"):
            os.remove(os.path.join(root, f))


def pretrained_model_list():
    """Get list of model which has pretrained weights available."""
    _renames = {
        'resnet18_v1b_2.6x': 'resnet18_v1b_0.89',
        'resnet50_v1d_1.8x': 'resnet50_v1d_0.86',
        'resnet50_v1d_3.6x': 'resnet50_v1d_0.48',
        'resnet50_v1d_5.9x': 'resnet50_v1d_0.37',
        'resnet50_v1d_8.8x': 'resnet50_v1d_0.11',
        'resnet101_v1d_1.9x': 'resnet101_v1d_0.76',
        'resnet101_v1d_2.2x': 'resnet101_v1d_0.73',
    }
    return [_renames[x] if x in _renames else x for x in _model_sha1.keys()]
