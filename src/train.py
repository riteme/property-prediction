from typing import List, Sequence, Any, Text, Tuple

import time
from math import ceil

import torch

import log
import util
from util import Item
from interface import ModelInterface

def train_fold(
    model: ModelInterface,
    sampler: util.Sampler,
    train_data_size: int,
    val_batch: List[Any],
    val_label: List[int],
    *,
    no_reset: bool,
    learning_rate: float,
    batch_size: int,
    beta: float,
    epsilon: float,
    score_expression: Text,
    maximal_count: int,
    min_iteration: int,
    max_iteration: int,
    **kwargs
):
    if no_reset:
        log.warn('Model reset disabled.')
    else:
        model.reset()

    optimizer = torch.optim.Adam(params=model.inst.parameters(), lr=learning_rate)
    watcher = util.MaximalCounter()

    # training iterations
    min_loss = 1e99  # track history minimal loss
    batch_per_epoch = ceil(train_data_size / batch_size)
    log.debug(f'batch_per_epoch={batch_per_epoch}')
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
        roc_auc, prc_auc, pred_label = evaluate_model(model, val_batch, val_label)
        f_score = util.metrics.fbeta_score(val_label, pred_label, beta=beta)
        watcher.record(eval(score_expression, None, {
            'prc_auc': prc_auc,
            'roc_auc': roc_auc,
            'f_score': f_score,
            'loss': loss
        }))
        time_used = time.time() - epoch_start

        log.debug(f'[{i}] train:    loss={loss},min={min_loss}')
        log.debug(f'[{i}] validate: {util.stat_string(val_label, pred_label)}. roc={roc_auc},prc={prc_auc},fβ={f_score}')
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
    label: List[int],
    show_stats: bool = False
) -> Tuple[float, float, List[int]]:
    with torch.no_grad():
        pred = model.predict(batch)
        pred_label = pred.argmax(dim=1)
        index = pred_label.to(torch.bool)

        if show_stats:
            stats = torch.cat([
                pred[index].to('cpu'),
                torch.tensor(label, dtype=torch.float).reshape(-1, 1)[index]
            ], dim=1)
            log.debug(stats)

    return *util.evaluate_auc(label, pred[:, 1]), pred_label.tolist()