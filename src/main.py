from typing import (
    Text, List, Sequence,
    Any, Tuple, Optional,
    Iterable
)

import time
from pathlib import Path
from math import ceil

import click
import torch

import log
import util
import models
from util import Item
from interface import ModelInterface

@click.group()
@click.option('-v', '--verbose', is_flag=True, help='Show debug messages.')
def cli(verbose: bool) -> None:
    if verbose:
        log.LOG_LEVEL = 0

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
@click.option('-s', '--score-expression', type=str, default='(prc_auc, roc_auc)', show_default=True,
    help='The expression of score for maximal counter. Available metrics: "prc_auc", "roc_auc", "f_score".')
@click.option('--maximal-count', type=int, default=10, show_default=True,
    help='Number of maximals assumed to be converged.')
@click.option('--train-validate', is_flag=True,
    help='Train with validate set (data[1]).')
@click.option('--min-iteration', type=int, default=6, show_default=True,
    help='Minimum number of iterations.')
@click.option('--max-iteration', type=int, default=50, show_default=True,
    help='Maximum number of iterations.')
@click.option('--ndrop', type=float,
    help='Probability to drop negative items during training.')
@click.option('--cuda', is_flag=True,
    help='Prefer CUDA for PyTorch.')

# if you want to pass some parameters directly to models,
# store them in kwargs. e.g. "embedding_dim" below
@click.option('--embedding-dim', type=int)
def train(
    directory: Text,
    model_name: Text,
    batch_size: int, learning_rate: float,
    epsilon: float, beta: float,
    train_validate: bool,
    score_expression: Text,
    maximal_count: int,
    min_iteration: int,
    max_iteration: int,
    cuda: bool,
    ndrop: Optional[float] = None,
    **kwargs
) -> None:
    # filter out options that are not set in command line
    kwargs = util.dict_filter(kwargs, lambda k, v: v is not None)

    data_folder = Path(directory)
    assert data_folder.is_dir(), 'Invalid data folder'

    dev = require_device(cuda)
    for fold in sorted(data_folder.iterdir()):
        log.info(f'Processing "{fold}"...')

        # model & optimizer
        model_type = models.select(model_name)  # see models/__init__.py
        model = ModelInterface(model_type, dev, **kwargs)
        optimizer = torch.optim.Adam(params=model.inst.parameters(), lr=learning_rate)

        # load the fold and let the model parse these molecules
        # data[0]: training set
        # data[1]: validate set
        # data[2]: test set
        data = load_data(model, fold, ['train.csv', 'dev.csv', 'test.csv'])
        log.debug(f'atom_map: {model.atom_map}')

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
            half = batch_size // 2
            sampler = util.SeparateSampling(train_data, {0: half, 1: half}, lambda x: x.activity)

        # training phase
        min_loss = 1e99  # track history minimal loss
        batch_per_epoch = ceil(len(train_data) / batch_size)
        log.debug(f'batch_per_epoch={batch_per_epoch}')

        watcher = util.MaximalCounter()
        for i in range(max_iteration):  # epochs
            sum_loss = 0.0
            epoch_start = time.time()

            for _ in range(batch_per_epoch):  # batches
                # generate batch
                batch, _label = util.separate_items(sampler.get_batch())
                label = torch.tensor(_label)

                # train a mini-batch
                batch_loss = train_step(model, optimizer, batch, label)
                sum_loss += batch_loss

            loss = sum_loss / batch_per_epoch
            roc_auc, prc_auc, pred = evaluate_model(model, val_batch, val_label)
            f_score = util.metrics.fbeta_score(val_label, pred, beta=beta)
            watcher.record(eval(score_expression, None, {
                'prc_auc': prc_auc,
                'roc_auc': roc_auc,
                'f_score': f_score
            }))
            time_used = time.time() - epoch_start

            log.debug(f'[{i}] train:    loss={loss},min={min_loss}')
            log.debug(f'[{i}] validate: {util.stat_string(val_label, pred)}. roc={roc_auc},prc={prc_auc},fβ={f_score}')
            log.debug(f'[{i}] watcher: {watcher}')
            log.debug(f'[{i}] epoch time={"%.3f" % time_used}s')

            # if i >= min_iteration and abs(min_loss - loss) < epsilon:
            #     break

            # save state
            if watcher.is_updated():
                model.save_checkpoint()
            if i >= min_iteration and watcher.count >= maximal_count:
                break

            min_loss = min(min_loss, loss)
            sum_loss = 0.0

        # load best model
        model.load_checkpoint()

        # model evaluation on `dev.csv`
        roc_auc, prc_auc, pred = evaluate_model(model, test_batch, test_label)
        log.info(f'test: {util.stat_string(test_label, pred)}')
        log.info(f'ROC-AUC: {roc_auc}')
        log.info(f'PRC-AUC: {prc_auc}')

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
    for i in range(len(raw)):
        buf = []
        for smiles, activity in raw[i].items():
            obj = model.process(smiles)
            buf.append(Item(obj, activity))
        data.append(buf)

    return data

def train_step(
    model: ModelInterface,
    # `torch.optim.optimizer.Optimizer` is ghost.
    # WHY DOES MYPY NOT RECOGNIZE `torch.optim.Optimizer`?
    optimizer: 'torch.optim.optimizer.Optimizer',
    batch: Sequence[Any],
    label: torch.Tensor
) -> float:
    loss = evaluate_loss(model, batch, label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate_loss(
    model: ModelInterface,
    batch: Sequence[Any],
    label: torch.Tensor
) -> torch.Tensor:
    # criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 60.0]))
    criterion = torch.nn.CrossEntropyLoss()
    pred = model.forward(batch)
    loss = criterion(pred, label)
    return loss

def evaluate_model(
    model: ModelInterface,
    batch: List[object],
    label: List[int]
) -> Tuple[float, float, torch.Tensor]:
    pred = model.predict(batch)
    return *util.evaluate_auc(label, pred), pred

if __name__ == '__main__':
    cli()