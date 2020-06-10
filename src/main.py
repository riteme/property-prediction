from typing import (
    Text, List, Sequence,
    Any, Tuple, Optional,
    Iterable, TextIO, Dict,
    DefaultDict, Hashable, Type,
    NamedTuple
)
from typing import Counter as TCounter

from itertools import repeat
from pathlib import Path
from collections import Counter, defaultdict

import click
import torch
import multiprocessing as mp

import log
import util
import models
from util import Item
from train import train_fold, evaluate_model
from interface import ModelInterface

class GlobalInitArgs:
    verbose: bool = False
    num_threads: int = mp.cpu_count()
_GLOBAL_INITARGS = GlobalInitArgs()

@click.group()
@click.option('-v', '--verbose', is_flag=True, help='Show debug messages.')
def cli(verbose: bool) -> None:
    global GLOBAL_INITARGS
    _GLOBAL_INITARGS.verbose = verbose
    global_initialize(_GLOBAL_INITARGS)

# since multiprocessing has to choose the "spawn" method,
# every global state must be re-configured.
def global_initialize(args: GlobalInitArgs):
    if args.verbose:
        log.LOG_LEVEL = 0
    torch.set_num_threads(args.num_threads)

@cli.command(short_help='Train model.')
@click.option('-d', '--directory', type=str, default='data', show_default=True,
    help='Data directory.')
@click.option('-m', '--model-name', type=str, required=True,
    help='The name of the model to be trained.')
@click.option('-b', '--batch-size', type=int, default=64, show_default=True,
    help='Batch size.')
@click.option('-l', '--learning-rate', type=float, default=0.03, show_default=True,
    help='Learning rate for optimizer.')
@click.option('-e', '--epsilon', type=float, default=1e-3, show_default=True,
    help='Maximum difference assumed to be converged.')
@click.option('--beta', type=float, default=1.0, show_default=True,
    help='Parameter for F_β score.')
@click.option('-s', '--score-expression', type=str, default='(prc_auc,roc_auc)', show_default=True,
    help='The expression of score for maximal counter. Available metrics: "prc_auc", "roc_auc", "f_score", "loss".')
@click.option('--maximal-count', type=int, default=15, show_default=True,
    help='Number of maximals assumed to be converged.')
@click.option('--train-validate', is_flag=True,
    help='Train with validate set (data[1]).')
@click.option('--min-iteration', type=int, default=6, show_default=True,
    help='Minimum number of iterations.')
@click.option('--max-iteration', type=int, default=50, show_default=True,
    help='Maximum number of iterations.')
@click.option('-p', '--positive-percentage', type=float, default=0.5, show_default=True,
    help='Percentage of positive samples in one batch.')
@click.option('--ndrop', type=float,
    help='Probability to drop negative items during training.')
@click.option('--no-reset', is_flag=True,
    help='Do not reset model instance during cross validation.')
@click.option('-t', '--num-threads', type=int, default=mp.cpu_count(), show_default=True,
    help='Number of PyTorch threads for OpenMP.')
@click.option('-j', '--num-workers', type=int, default=1, show_default=True,
    help='Number of worker processes. "--no-reset" is not available in multiprocessing mode.')
@click.option('--cuda', is_flag=True,
    help='Prefer CUDA for PyTorch.')

# if you want to pass some parameters directly to models,
# store them in kwargs. e.g. "embedding_dim" below
@click.option('--embedding-dim', type=int)
@click.option('--no-shortcut', is_flag=True)
def train(
    directory: Text,
    model_name: Text,
    cuda: bool,
    num_threads: int,
    num_workers: int,
    **kwargs
) -> None:
    # configure PyTorch's OpenMP
    log.info(f'Number of threads: {num_threads}')
    _GLOBAL_INITARGS.num_threads = num_threads
    torch.set_num_threads(num_threads)

    data_folder = Path(directory)
    assert data_folder.is_dir(), 'Invalid data folder'

    device = require_device(cuda)
    model_type = models.select(model_name)  # see models/__init__.py

    # filter out options that are not set in command line
    # default to ignore both None and False (for flag options)
    model_kwargs = util.dict_filter(kwargs, lambda k, v: bool(v))
    model_info = (model_type, device, model_kwargs)
    model = get_model(model_info, no_initialize=True)

    # preload data into memcache
    log.info('Preparing data...')
    full_data = util.load_csv(data_folder/'train.csv')
    for smiles in full_data.keys():
        _ = model.process(smiles)

    # process folds
    folds = []
    roc_record = []
    prc_record = []
    for fold in sorted(data_folder.iterdir()):
        if not fold.is_dir():
            log.debug(f'Ignored "{fold}".')
            continue
        folds.append(fold)

    if num_workers == 1:
        model = get_model(model_info)
        for fold in folds:
            roc_auc, prc_auc = process_fold(model, fold, **kwargs)
            roc_record.append(roc_auc)
            prc_record.append(prc_auc)
    else:
        log.info(f'Number of workers: {num_workers}')

        # PyTorch seems to be not compatible with the "fork" method
        ctx = mp.get_context('spawn')
        with ctx.Pool(num_workers, global_initialize, (_GLOBAL_INITARGS, )) as workers:
            args_iterator = zip(folds, repeat(model_info), repeat(kwargs))
            results = workers.imap_unordered(task_wrapper, args_iterator)

            for roc_auc, prc_auc in results:
                roc_record.append(roc_auc)
                prc_record.append(prc_auc)

    # basic statistics
    roc = torch.tensor(roc_record)
    prc = torch.tensor(prc_record)
    float_fmt = lambda x: float('%.4f' % x)
    log.info(f'roc = {list(map(float_fmt, roc_record))}')
    log.info(f'prc = {list(map(float_fmt, prc_record))}')
    log.info('All folds: ROC-AUC = %.3f±%.3f, PRC-AUC = %.3f±%.3f' % (
        roc.mean(), roc.std(), prc.mean(), prc.std()
    ))

TModelInfo = Tuple[Type[models.BaseModel], torch.device, Dict[Text, Any]]

def get_model(model_info: TModelInfo, no_initialize: bool = False) -> ModelInterface:
    model_type, device, kwargs = model_info
    return ModelInterface(model_type, device, no_initialize=no_initialize, **kwargs)

def task_wrapper(arg_pack: Tuple[Path, TModelInfo, Dict[Text, Any]]) -> Tuple[float, float]:
    '''For multiprocessing.
    '''
    fold, model_info, kwargs = arg_pack
    log.PROC_NAME = fold.name

    # each worker has to instantiate its own model.
    # molecule memory cache is shared among all `ModelInterface`.
    model = get_model(model_info)
    return process_fold(model, fold, **kwargs)

def process_fold(
    model: ModelInterface,
    fold: Path,
    *,
    batch_size: int,
    train_validate: bool,
    positive_percentage: float,
    ndrop: Optional[float] = None,
    **kwargs
) -> Tuple[float, float]:
    log.info(f'Processing "{fold}"...')

    # load the fold and let the model parse these molecules
    # data[0]: training set
    # data[1]: validate set
    # data[2]: test set
    data = load_data(model, fold, ['train.csv', 'dev.csv', 'test.csv'])

    # prepare data
    val_batch, val_label = util.separate_items(data[1])
    test_batch, test_label = util.separate_items(data[2])
    train_data = data[0] + data[1] if train_validate else data[0]

    sampler: util.Sampler
    if ndrop is not None:
        log.debug('with --ndrop')
        # set up to randomly drop negative samples
        # see util.RandomIterator for details
        drop_fn = lambda x: ndrop if x.activity == 0 else 0
        sampler = util.RandomIterator(train_data,
            batch_size=batch_size,
            drop_fn=drop_fn
        )
    else:
        num_positive = round(batch_size * positive_percentage)
        num_negative = batch_size - num_positive
        type_fn = lambda x: x.activity
        sampler = util.SeparateSampling(train_data, {0: num_negative, 1: num_positive}, type_fn)

    # training phase
    train_fold(
        model, sampler, len(train_data), val_batch, val_label,
        batch_size=batch_size, **kwargs
    )

    # model evaluation on `dev.csv`
    roc_auc, prc_auc, pred_label = evaluate_model(
        model, test_batch, test_label, show_stats=True
    )
    log.info(f'test: {util.stat_string(test_label, pred_label)}')
    log.info(f'ROC-AUC: {roc_auc}')
    log.info(f'PRC-AUC: {prc_auc}')

    return roc_auc, prc_auc

def require_device(prefer_cuda: bool) -> torch.device:
    if prefer_cuda and not torch.cuda.is_available():
        log.warn('CUDA not available.')
        prefer_cuda = False
    return torch.device('cuda' if prefer_cuda else 'cpu')

def load_data(
    model: ModelInterface,
    folder: Path,
    files: Iterable[Text]
) -> List[List[Item]]:
    raw = [
        util.load_csv(folder/name)
        for name in files
    ]

    data = []
    for csv in raw:
        buf = []
        for smiles, activity in csv.items():
            obj = model.process(smiles)
            buf.append(Item(obj, activity))
        data.append(buf)

    return data

@cli.command(short_help='Show statistics of data.')
@click.argument('data', type=click.File('r'))
def stats(data: TextIO):
    fn_list = [
        'GetAtomicNum', 'GetExplicitValence', 'GetImplicitValence',
        'GetHybridization', 'GetMass', 'GetTotalNumHs',
        'GetNumRadicalElectrons', 'IsInRing', 'GetChiralTag',
        'GetDegree', 'GetFormalCharge', 'GetIsAromatic'
    ]

    TStats = DefaultDict[Text, TCounter[Hashable]]
    stats: TStats = defaultdict(lambda: Counter())
    csv = util.load_csv(data)

    for smiles in csv.keys():
        mol = util.parse_smiles(smiles)
        for fn_name in fn_list:
            stats[fn_name].update(
                getattr(atom, fn_name)()
                for atom in mol.GetAtoms()
            )

    for fn_name, cnt in stats.items():
        log.info(f'{fn_name}: ({len(cnt)} items)\n\t{dict(cnt)}')

if __name__ == '__main__':
    cli()