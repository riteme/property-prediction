from random import shuffle
from pathlib import Path
from collections import namedtuple
from typing import Text, List, Sequence, Any, Tuple

import click
import torch

import log
import util
from model import BaseModel
from interface import ModelInterface

Item = namedtuple('Item', ['obj', 'activity'])

@click.group()
@click.option('-v', '--verbose', is_flag=True, help='Show debug messages.')
def cli(verbose: bool) -> None:
    if verbose:
        log.LOG_LEVEL = 0

def require_device(prefer_cuda: bool) -> torch.device:
    if prefer_cuda and not torch.cuda.is_available():
        log.warn('CUDA not available.')
        prefer_cuda = False
    return torch.device('cuda' if prefer_cuda else 'cpu')

def train_step(
    model: ModelInterface,
    optimizer: torch.optim.optimizer.Optimizer,
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

def evaluate_model(model: ModelInterface, data: List[Item]) -> Tuple[float, float]:
    batch = []
    label = []
    for x in data:
        batch.append(x.obj)
        label.append(x.activity)

    pred = model.predict(data)
    return util.evaluate_auc(label, pred)

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
    model = ModelInterface(BaseModel, dev)
    optimizer = torch.optim.Adam(params=model.inst.parameters(), lr=learning_rate)

    for fold in sorted(data_folder.iterdir()):
        log.info(f'Processing "{fold}"...')

        raw = [
            util.load_csv(fold/name)
            for name in ['train.csv', 'test.csv', 'dev.csv']
        ]

        data: List[List[Item]] = []
        for i in range(len(raw)):
            buf: List[Item] = []
            for smiles, activity in data[i]:
                buf.append(Item(model.process(smiles), activity))
            data.append(buf)

        data_ptr = len(data[0])
        last_loss = None
        while True:
            batch = [None] * batch_size
            label = torch.zeros(batch_size)
            for i in range(batch_size):
                if data_ptr >= len(data[0]):
                    data_ptr = 0
                    shuffle(data[0])
                batch[i] = data[0][i].obj
                label[i] = data[0][i].activity

            loss = train_step(model, optimizer, batch, label)

            log.debug(f'loss = {loss}')
            if last_loss and abs(loss - last_loss) < epsilon:
                log.debug('Converged.')
                break
            last_loss = loss

        roc_auc, prc_auc = evaluate_model(model, data[2])
        log.info(f'ROC-AUC: {roc_auc}')
        log.info(f'PRC-AUC: {prc_auc}')

if __name__ == '__main__':
    cli()