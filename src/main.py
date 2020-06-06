from typing import Text, List, Sequence, Any, Tuple

from pathlib import Path

import click
import torch

import log
import util
from util import Item
from interface import ModelInterface

from model import (
    BaseModel,
    StupidModel
)

@click.group()
@click.option('-v', '--verbose', is_flag=True, help='Show debug messages.')
def cli(verbose: bool) -> None:
    if verbose:
        log.LOG_LEVEL = 0

@cli.command(short_help='Train model.')
@click.option('-d', '--directory', type=str, default='data',
    help='Data directory.')
@click.option('-b', '--batch-size', type=int, default=32,
    help='Batch size.')
@click.option('-l', '--learning-rate', type=float, default=0.03,
    help='Learning rate for optimizer.')
@click.option('-e', '--epsilon', type=float, default=1e-3,
    help='Maximum difference assumed to be converged.')
@click.option('--cuda', is_flag=True,
    help='Prefer CUDA for PyTorch.')
def train(directory: Text,
    batch_size: int, learning_rate: float,
    epsilon: float, cuda: bool
) -> None:
    data_folder = Path(directory)
    assert data_folder.is_dir(), 'Invalid data folder'

    dev = require_device(cuda)
    model = ModelInterface(StupidModel, dev)
    optimizer = torch.optim.Adam(params=model.inst.parameters(), lr=learning_rate)

    for fold in sorted(data_folder.iterdir()):
        log.info(f'Processing "{fold}"...')

        raw = [
            util.load_csv(fold/name)
            for name in ['train.csv', 'test.csv', 'dev.csv']
        ]

        data = []
        for i in range(len(raw)):
            buf = []
            for smiles, activity in raw[i].items():
                buf.append(Item(model.process(smiles), activity))
            data.append(buf)

        test_batch, _test_label = util.separate_items(data[1])
        test_label = torch.tensor(_test_label)

        data_ptr = util.RandomIterator(data[0])
        last_loss = None
        while True:
            batch, _label = util.separate_items(data_ptr.iterate(batch_size))
            label = torch.tensor(_label)

            loss = train_step(model, optimizer, batch, label)
            pred = model.predict(test_batch)
            correct = (pred == test_label).sum().item()
            accuracy = correct / len(test_batch)
            log.debug(f'loss = {loss}')
            log.debug(f'accuracy = {accuracy}')

            if last_loss and abs(loss - last_loss) < epsilon:
                log.debug('Converged.')
                break
            last_loss = loss

        roc_auc, prc_auc = evaluate_model(model, data[2])
        log.info(f'ROC-AUC: {roc_auc}')
        log.info(f'PRC-AUC: {prc_auc}')

def require_device(prefer_cuda: bool) -> torch.device:
    if prefer_cuda and not torch.cuda.is_available():
        log.warn('CUDA not available.')
        prefer_cuda = False
    return torch.device('cuda' if prefer_cuda else 'cpu')

def train_step(
    model: ModelInterface,
    # `torch.optim.optimizer.Optimizer` does not exist.
    # WHY DOES MYPY NOT RECOGNIZE `torch.optim.Optimizer`?
    optimizer: 'torch.optim.optimizer.Optimizer',
    batch: Sequence[Any],
    label: torch.Tensor
) -> float:
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
    pred = model.predict(data)
    return util.evaluate_auc(label, pred.tolist())

if __name__ == '__main__':
    cli()