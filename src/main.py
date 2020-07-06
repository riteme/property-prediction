from typing import *
from typing import BinaryIO, TextIO, IO
from typing import Counter as TCounter

import json
import pickle
from itertools import repeat
from pathlib import Path
from collections import Counter, defaultdict

import click
import torch
import multiprocessing as mp

import log
import util
import cache
import models
from util import Item
from train import train_fold, evaluate_model
from interface import ModelInterface


TModelInfo = Tuple[Type[models.BaseModel], torch.device, Dict[Text, Any]]


class GlobalInitArgs:
    verbose: bool = False
    num_threads: int = mp.cpu_count()
    cache_file: Optional[Text] = None
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

    if args.cache_file is not None:
        cache.register_provider(ModelInterface.process, args.cache_file)


@cli.command('train', short_help='Train model.')
@click.option('-d', '--directory', type=str, default='data', show_default=True,
    help='Data directory.')
@click.option('-o', '--output', type=click.File('wb'),
    help='Save model parameters to local file.')
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
@click.option('--train-test', is_flag=True,
    help='Train with test set (data[2]).')
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
@click.option('-sp', '--spawn-method', type=click.Choice(['fork', 'spawn', 'forkserver']), default='spawn', show_default=True,
    help='Method for multiprocessing module to spawn new worker processes.')
@click.option('--cuda', is_flag=True,
    help='Prefer CUDA for PyTorch.')
@click.option('-c', '--cache-file', type=click.Path(exists=True),
    help='Path for cache file.')

# if you want to pass some parameters directly to models,
# store them in kwargs. e.g. "embedding_dim" below
@click.option('--embedding-dim', type=int)
@click.option('--no-shortcut', is_flag=True)
def _train(
    directory: Text,
    model_name: Text,
    cuda: bool,
    num_threads: int,
    num_workers: int,
    spawn_method: Text,
    output: Optional[BinaryIO],
    cache_file: Optional[Text],
    **kwargs
) -> None:
    # configure PyTorch's OpenMP
    log.info(f'Number of threads: {num_threads}')
    _GLOBAL_INITARGS.num_threads = num_threads
    _GLOBAL_INITARGS.cache_file = cache_file
    global_initialize(_GLOBAL_INITARGS)

    data_folder = Path(directory)
    assert data_folder.is_dir(), 'Invalid data folder'

    if cuda and cache_file is not None:
        log.warn('Disk cache file may not be compatible with CUDA tensors.')

    model, model_info = get_model(model_name, cuda, kwargs)

    # preload data into memcache
    full_data_path = data_folder/'train.csv'
    if full_data_path.exists():
        full_data = util.load_csv(full_data_path)
        log.info('Preparing data...')
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

    if output is not None and num_workers > 1:
        log.fatal('Attempt to save multiple instances of model.')

    # multiprocessing configuration
    if num_workers == 1:
        model = build_model(model_info)
        for fold in folds:
            roc_auc, prc_auc = process_fold(model, fold, **kwargs)
            roc_record.append(roc_auc)
            prc_record.append(prc_auc)

        # saving model parameter
        if output is not None:
            log.info('Saving model...')
            model.save_model(output)
    else:
        log.info(f'Number of workers: {num_workers}')

        # PyTorch's OpenMP seems to be not compatible with the "fork" method
        if spawn_method == 'fork' and num_threads > 1:
            log.warn('The "fork" method may result in deadlock with multithreaded training.')
        if spawn_method == 'spawn':
            log.warn('The "spawn" method will invalidate memory molecule caches.')

        ctx = mp.get_context(spawn_method)
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

def build_model(model_info: TModelInfo, no_initialize: bool = False) -> ModelInterface:
    model_type, device, kwargs = model_info
    return ModelInterface(model_type, device, no_initialize=no_initialize, **kwargs)

def get_model(
    model_name: Text,
    prefer_cuda: bool,
    kwargs: Dict
):
    device = require_device(prefer_cuda)
    model_type = models.select(model_name)  # see models/__init__.py

    # filter out options that are not set in command line
    # default to ignore both None and False (for flag options)
    model_kwargs = util.dict_filter(kwargs, lambda k, v: bool(v))
    model_info = (model_type, device, model_kwargs)
    model = build_model(model_info, no_initialize=True)
    return model, model_info

def report_scores(
    model: ModelInterface,
    batch: List[object],
    label: List[int]
) -> Tuple[float, float]:
    roc_auc, prc_auc, pred_label = evaluate_model(
        model, batch, label, show_stats=True
    )
    log.info(f'test: {util.stat_string(label, pred_label)}')
    log.info(f'ROC-AUC: {roc_auc}')
    log.info(f'PRC-AUC: {prc_auc}')

    return roc_auc, prc_auc

def require_device(prefer_cuda: bool) -> torch.device:
    if prefer_cuda and not torch.cuda.is_available():
        log.warn('CUDA not available.')
        prefer_cuda = False
    return torch.device('cuda' if prefer_cuda else 'cpu')

def process_csv(
    model: ModelInterface,
    csv: Dict[Text, int]
) -> List[Item]:
    return [
        Item(model.process(smiles), activity)
        for smiles, activity in csv.items()
    ]

def load_folds(
    model: ModelInterface,
    folder: Path,
    files: Iterable[Text]
) -> List[List[Item]]:
    return [
        process_csv(
            model,
            util.load_csv(folder/name)
        )
        for name in files
    ]

def task_wrapper(arg_pack: Tuple[Path, TModelInfo, Dict[Text, Any]]) -> Tuple[float, float]:
    '''For multiprocessing.
    '''
    fold, model_info, kwargs = arg_pack
    log.PROC_NAME = fold.name

    # each worker has to instantiate its own model.
    # molecule memory cache is shared among all `ModelInterface`.
    model = build_model(model_info)
    return process_fold(model, fold, **kwargs)

def process_fold(
    model: ModelInterface,
    fold: Path,
    *,
    batch_size: int,
    train_validate: bool,
    train_test: bool,
    positive_percentage: float,
    ndrop: Optional[float] = None,
    **kwargs
) -> Tuple[float, float]:
    log.info(f'Processing "{fold}"...')

    # load the fold and let the model parse these molecules
    data = load_folds(model, fold, ['train.csv', 'dev.csv', 'test.csv'])
    train_data, validate_set, test_set = data

    # prepare data
    val_batch, val_label = util.separate_items(validate_set)
    test_batch, test_label = util.separate_items(test_set)
    if train_validate:
        train_data += validate_set
    if train_test:
        train_data += test_set

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
    scores = report_scores(model, test_batch, test_label)
    return scores


@cli.command('stats', short_help='Show statistics of data.')
@click.argument('data', type=click.File('r'))
def _stats(data: TextIO):
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


@cli.command('cache', short_help='Build cache.')
@click.argument('data', type=click.File('r'))
@click.option('-m', '--model-name', type=str, required=True,
    help='The name of the model to be cached.')
@click.option('-o', '--output', type=click.File('wb'), required=True,
    help='Path to place the cache file.')
def _cache(data: TextIO, model_name: Text, output: BinaryIO):
    cpu = require_device(prefer_cuda=False)
    model_type = models.select(model_name)
    model = ModelInterface(model_type, cpu, False)

    csv = util.load_csv(data)
    cache = {}
    for smiles in csv.keys():
        cache_key = (smiles, )  # memcached is indexed on argument list
        data = model.process(smiles)
        cache[cache_key] = model.encode_data(data)

    pickle.dump(cache, output)


@cli.command('evaluate', short_help='Evaluate model.')
@click.argument('state_fp', type=click.File('rb'))
@click.option('-m', '--model-name', type=str, required=True,
    help='The name of the model to be evaluated.')
@click.option('-d', '--data', type=click.File('r'), required=True,
    help='Target CSV data file.')
@click.option('-c', '--cache-file', type=click.Path(exists=True),
    help='Path for cache file.')
@click.option('-r', '--args', type=str, default='{}', show_default='"{}"',
    help='Model parameters in JSON format.')
@click.option('--cuda', is_flag=True,
    help='Prefer CUDA for PyTorch.')
def _evaluate(
    state_fp: IO,
    model_name: Text,
    data: IO,
    cuda: bool,
    args: Text,
    cache_file: Optional[Text]
):
    log.debug('Parsing JSON...')
    kwargs = json.loads(args)

    log.info('Initialize model...')
    model, _ = get_model(model_name, cuda, kwargs)
    model.initialize_model()
    model.load_model(state_fp)

    log.info('Loading data...')
    if cache_file:
        cache.register_provider(ModelInterface.process, cache_file)
    raw = process_csv(model, util.load_csv(data))
    mols, labels = util.separate_items(raw)

    log.info('Evaluating...')
    report_scores(model, mols, labels)


if __name__ == '__main__':
    cli()