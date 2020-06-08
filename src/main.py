from typing import Text, List, Sequence, Any, Tuple, Optional

from pathlib import Path

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
@click.option('-d', '--directory', type=str, default='data',
    help='Data directory.')
@click.option('-m', '--model-name', type=str, required=True,
    help='The name of the model to be trained.')
@click.option('-b', '--batch-size', type=int, default=64,
    help='Batch size.')
@click.option('-l', '--learning-rate', type=float, default=0.03,
    help='Learning rate for optimizer.')
@click.option('-e', '--epsilon', type=float, default=1e-3,
    help='Maximum difference assumed to be converged.')
@click.option('--train-with-test', is_flag=True,
    help='Train with test data (data[1]).')
@click.option('--min-iteration', type=int, default=10,
    help='Minimum number of iterations.')
@click.option('--max-iteration', type=int, default=100,
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
    epsilon: float, cuda: bool,
    train_with_test: bool,
    min_iteration: int,
    max_iteration: int,
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

        # load the fold
        raw = [
            util.load_csv(fold/name)
            for name in ['train.csv', 'test.csv', 'dev.csv']
        ]

        # let the model parse these molecules
        data = []
        for i in range(len(raw)):
            buf = []
            for smiles, activity in raw[i].items():
                obj = model.process(smiles)
                buf.append(Item(obj, activity))
            data.append(buf)
        log.debug(f'atom_map: {model.atom_map}')

        test_batch, _test_label = util.separate_items(data[1])
        test_label = torch.tensor(_test_label)

        # training phase
        train_data = data[0] + data[1] if train_with_test else data[0]

        # set up to randomly drop negative samples
        # see util.RandomIterator for details
        drop_prob = ndrop if ndrop is not None else 0
        drop_fn = lambda x: drop_prob if x.activity == 0 else 0
        data_ptr = util.RandomIterator(train_data,
            drop_fn=drop_fn if ndrop is not None else None
        )

        countdown = min_iteration
        min_loss = 1e99  # track history minimal loss
        sum_loss, batch_cnt = 0.0, 0
        for _ in range(max_iteration):
            # generate batch
            batch, _label = util.separate_items(data_ptr.iterate(batch_size))
            label = torch.tensor(_label)

            # train a mini-batch
            batch_loss = train_step(model, optimizer, batch, label)
            sum_loss += batch_loss
            batch_cnt += 1
            # log.debug(f'{batch_loss}, {sum_loss}')

            # convergence test
            if data_ptr.is_cycled():
                loss = sum_loss / batch_cnt
                pred = model.predict(test_batch)
                log.debug(f'{util.stat_string(_test_label, pred)}. loss={loss},min={min_loss}')

                if countdown <= 0 and abs(min_loss - loss) < epsilon:
                    log.debug('Converged.')
                    break

                countdown -= 1
                min_loss = min(min_loss, loss)
                sum_loss, batch_cnt = 0.0, 0

        # model evaluation on `dev.csv`
        roc_auc, prc_auc = evaluate_model(model, data[2])
        log.info(f'ROC-AUC: {roc_auc}')
        log.info(f'PRC-AUC: {prc_auc}')
        # log.debug(f'parameters: {list(model.inst.parameters())}')

def require_device(prefer_cuda: bool) -> torch.device:
    if prefer_cuda and not torch.cuda.is_available():
        log.warn('CUDA not available.')
        prefer_cuda = False
    return torch.device('cuda' if prefer_cuda else 'cpu')

def train_step(
    model: ModelInterface,
    # `torch.optim.optimizer.Optimizer` is ghost.
    # WHY DOES MYPY NOT RECOGNIZE `torch.optim.Optimizer`?
    optimizer: 'torch.optim.optimizer.Optimizer',
    batch: Sequence[Any],
    label: torch.Tensor
) -> float:
    # criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 60.0]))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer.zero_grad()

    pred = model.forward(batch)
    loss = criterion(pred, label)
    loss.backward()
    optimizer.step()

    return loss.item()

def evaluate_model(
    model: ModelInterface,
    data: List[Item]
) -> Tuple[float, float]:
    batch, label = util.separate_items(data)
    pred = model.predict(batch)
    log.debug(f'final: {util.stat_string(label, pred)}')
    return util.evaluate_auc(label, pred)

if __name__ == '__main__':
    cli()