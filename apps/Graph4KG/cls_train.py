# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import time
import warnings
from collections import defaultdict

import numpy as np
import paddle
from paddle.nn import BCELoss
from paddle.optimizer.lr import StepDecay

from dataset.reader import read_trigraph
from dataset.dataset import create_dataloaders
from models.ke_model import KGEModel
from models.loss_func import LossFunction
from utils import set_seed, set_logger, print_log
from utils import evaluate
from config import prepare_config


def main():
    """Main function for shallow knowledge embedding methods.
    """
    args = prepare_config()
    set_seed(args.seed)
    set_logger(args)

    trigraph = read_trigraph(args.data_path, args.data_name)
    if args.use_symmetry:
        trigraph.add_symmetric_train_triplets()
    if args.valid_percent < 1:
        trigraph.sampled_subgraph(args.valid_percent, dataset='valid')

    use_filter_set = args.filter_sample or args.filter_eval or args.weighted_loss
    if use_filter_set:
        filter_dict = trigraph.true_cands_for_ent_rel()
    else:
        filter_dict = None

    model = KGEModel(args.model_name, trigraph, args)

    if args.async_update:
        model.start_async_update()

    if len(model.parameters()) > 0:
        optimizer = paddle.optimizer.Adam(
            learning_rate=args.lr,
            epsilon=1e-10,
            parameters=model.parameters())
    else:
        warnings.warn('There is no model parameter on gpu, optimizer is None.',
                      RuntimeWarning)
        optimizer = None

    loss_func = BCELoss()

    train_loader, valid_loader, test_loader = create_dataloaders(
        trigraph,
        args,
        mode='cls')

    timer = defaultdict(int)
    log = defaultdict(int)
    ts = t_step = time.time()
    step = 1
    stop = False
    for epoch in range(args.num_epoch):
        for ent_index, rel_index, label in train_loader:
            timer['sample'] += (time.time() - ts)
            model.set_train_mode()
            label = ((1.0 - args.label_smoothing) * label) + (1.0 / label.shape[1])

            ts = time.time()
            ent_emb, rel_emb, cand_emb = model.prepare_inputs(
                [ent_index, rel_index], data_mode='cls')

            score = model.forward(ent_emb, rel_emb, cand_emb)

            loss = loss_func(score, label)

            log['loss'] += loss.numpy()[0]
            timer['forward'] += (time.time() - ts)

            ts = time.time()
            loss.backward()
            timer['backward'] += (time.time() - ts)

            ts = time.time()

            if args.mix_cpu_gpu:
                ent_trace, rel_trace = model.create_trace(
                    paddle.arange(trigraph.num_ents), cand_emb, rel_index, rel_emb)
                model.step(ent_trace, rel_trace)
            else:
                model.step()

            if optimizer is not None:
                optimizer.step()
                optimizer.clear_grad()
            timer['update'] += (time.time() - ts)

            if args.log_interval > 0 and (step + 1) % args.log_interval == 0:
                print_log(step, args.log_interval, log, timer,
                          time.time() - t_step)
                timer = defaultdict(int)
                log = defaultdict(int)
                t_step = time.time()

            if args.valid and (step + 1) % args.eval_interval == 0:
                model.set_eval_mode()
                evaluate(
                    model,
                    valid_loader,
                    'valid',
                    filter_dict if args.filter_eval else None,
                    data_mode=args.data_mode)

            step += 1
            if args.save_interval > 0 and step % args.save_interval == 0:
                model.save(args.step_path)
            if step >= args.max_steps:
                stop = True
                break

            ts = time.time()
        if stop:
            break

    if args.async_update:
        model.finish_async_update()

    if args.test:
        evaluate(
            model,
            test_loader,
            'test',
            filter_dict if args.filter_eval else None,
            os.path.join(args.save_path, 'test.pkl'),
            data_mode=args.data_mode)


if __name__ == '__main__':
    main()
