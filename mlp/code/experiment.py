"""Paired-seed MLP localization experiments; no edits to source data."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
import pandas as pd
import torch

SOURCE = Path(__file__).parent / 'data'
TASKS = {
    'cylinder': ('Cylinder localization', 'sum_cylinder.csv', 'test_data_c.csv', 525, 147, 66),
    'fish': ('Fixed robotic fish localization', 'sumfish.csv', 'test_data_f.csv', 425, 131, 58),
}
CHANNELS = {
    'V6': ['VAUD', 'VBUD', 'VCUD', 'VALR', 'VBLR', 'VCLR'],
    'P12': ['PXAL', 'PXAM', 'PXAR', 'PXBL', 'PXBM', 'PXBR', 'PYAL', 'PYAM', 'PYAR', 'PYBL', 'PYBM', 'PYBR'],
    'P6': ['PXAL', 'PXAM', 'PXBM', 'PYAM', 'PYBM', 'PYBR'],
    'V4+P2': ['VAUD', 'VALR', 'VCUD', 'VCLR', 'PXAM', 'PYBM'],
}
CHANNELS['V6+P12'] = CHANNELS['V6'] + CHANNELS['P12']
CONFIGS = ['V6', 'P12', 'V6+P12', 'P6', 'V4+P2']


def columns(config):
    return [f'{c}_{stat}' for stat in ('mean', 'std') for c in CHANNELS[config]]


def standardize(train, other):
    mean = train.mean(axis=0)
    scale = train.std(axis=0, ddof=0)
    scale = np.where(scale == 0, 1., scale)
    return ((train - mean) / scale).astype('float32'), ((other - mean) / scale).astype('float32'), mean, scale


def load_task(task):
    folder, train, test, n_train, n_test, n_near = TASKS[task]
    a, b = [pd.read_csv(SOURCE / folder / f) for f in (train, test)]
    assert (len(a), len(b), int((b.x <= 27).sum())) == (n_train, n_test, n_near)
    assert list(a.columns) == list(b.columns)
    assert np.isfinite(a.to_numpy()).all() and np.isfinite(b.to_numpy()).all()
    for config in CONFIGS:
        assert set(columns(config)) <= set(a.columns)
    return a, b


def initial_parameters(dim, seeds):
    """Glorot uniform, zero biases; identical dim + seed gives identical weights."""
    sizes = [dim, 64, 48, 32, 2]
    weights, biases = [], []
    rngs = [np.random.default_rng(seed) for seed in seeds]
    for d_in, d_out in zip(sizes[:-1], sizes[1:]):
        limit = np.sqrt(6. / (d_in + d_out))
        w = np.stack([r.uniform(-limit, limit, (d_in, d_out)) for r in rngs]).astype('float32')
        weights.append(torch.nn.Parameter(torch.from_numpy(w)))
        biases.append(torch.nn.Parameter(torch.zeros(len(seeds), 1, d_out)))
    return weights, biases


def forward(x, weights, biases):
    for i, (w, b) in enumerate(zip(weights, biases)):
        x = torch.bmm(x, w) + b
        if i < 3:
            x = torch.relu(x)
    return x


def fit(x, y, other, seeds, epochs, batch_size, log=None):
    """Independent models vectorized on a seed axis, without averaging gradients.

    Summing per-model losses preserves each model's standalone MSE + L2 gradient.
    L2 applies only to the four Dense weight matrices, including output weights.
    Each seed's RNG for sample permutations starts afresh for every configuration.
    """
    weights, biases = initial_parameters(x.shape[1], seeds)
    initial_hash = hashlib.sha256(b''.join(w.detach().numpy().tobytes() for w in weights)).hexdigest()
    optimizer = torch.optim.Adam(weights + biases, lr=.002, betas=(.9, .999), eps=1e-7, weight_decay=0)
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    rngs = [np.random.default_rng(s) for s in seeds]
    losses = []
    order_hash=hashlib.sha256()
    started = time.monotonic()
    for epoch in range(epochs):
        order = np.stack([r.permutation(len(x)) for r in rngs])
        order_hash.update(order.tobytes())
        epoch_mse = np.zeros(len(seeds))
        for start in range(0, len(x), batch_size):
            idx = torch.from_numpy(order[:, start:start + batch_size])
            pred = forward(xt[idx], weights, biases)
            per_model_mse = (pred - yt[idx]).square().mean(dim=(1, 2))
            l2 = .0001 * sum(w.square().sum(dim=(1, 2)) for w in weights)
            loss = (per_model_mse + l2).sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_mse += per_model_mse.detach().numpy() * idx.shape[1]
        losses.append(epoch_mse / len(x))
        if log and (epoch + 1) % 100 == 0:
            print(f'{log}: epoch {epoch+1}/{epochs}, {time.monotonic()-started:.1f}s', flush=True)
    with torch.no_grad():
        prediction = forward(torch.from_numpy(other).unsqueeze(0).expand(len(seeds), -1, -1), weights, biases).numpy()
    assert np.isfinite(prediction).all()
    return prediction, {
        'weights': [w.detach() for w in weights], 'biases': [b.detach() for b in biases],
        'seeds': list(seeds), 'initial_weights_sha256': initial_hash,
        'sample_order_sha256':order_hash.hexdigest(),
    }, np.array(losses)


def metrics(true, pred, task, config, seeds):
    rows = []
    for region, mask in [('full', np.ones(len(true), dtype=bool)), ('near', true[:, 0] <= 27)]:
        error = pred[:, mask] - true[mask][None, :, :]
        values = {'mae_x': np.abs(error[:, :, 0]).mean(axis=1),
                  'mae_y': np.abs(error[:, :, 1]).mean(axis=1),
                  'mean_distance': np.linalg.norm(error, axis=2).mean(axis=1)}
        for metric, vals in values.items():
            rows.extend(dict(task=task, config=config, seed=int(seed), region=region,
                             metric=metric, error_cm=float(val), n_test=int(mask.sum()))
                        for seed, val in zip(seeds, vals))
    return rows


def train_task(task, out, seeds, epochs, batch_size, run_ids=None):
    run_ids = list(range(1001,1021)) if run_ids is None else run_ids
    assert len(seeds)==len(run_ids)==20 and len(set(seeds))==20
    train, test = load_task(task)
    y, target, ym, ys = standardize(train[['x', 'y']].to_numpy(), test[['x', 'y']].to_numpy())
    true = test[['x', 'y']].to_numpy()
    rows, hashes = [], {}
    for config in CONFIGS:
        dest = out / task / config
        dest.mkdir(parents=True, exist_ok=True)
        x, xt, xm, xs = standardize(train[columns(config)].to_numpy(), test[columns(config)].to_numpy())
        pred, model, losses = fit(x, y, xt, seeds, epochs, batch_size, f'{task}/{config}')
        pred = pred.astype('float64') * ys + ym
        rows.extend(metrics(true, pred, task, config, run_ids))
        model['run_ids']=run_ids
        for j, run_id in enumerate(run_ids):
            pd.DataFrame({'run_id':run_id, 'test_row':np.arange(len(true)), 'true_x_cm':true[:,0],
                          'true_y_cm':true[:,1], 'pred_x_cm':pred[j,:,0], 'pred_y_cm':pred[j,:,1],
                          'error_distance_cm':np.linalg.norm(pred[j]-true, axis=1)}).to_csv(dest / f'predictions_{run_id}.csv', index=False)
        torch.save(model, dest / 'models.pt')
        (dest / 'scalers.json').write_text(json.dumps({'features':columns(config), 'input_mean':xm.tolist(),
            'input_scale':xs.tolist(), 'target_mean':ym.tolist(), 'target_scale':ys.tolist()}, indent=2))
        pd.DataFrame(losses, columns=[str(s) for s in run_ids]).rename_axis('epoch_zero_based').to_csv(dest/'training_mse.csv')
        hashes[config] = model['initial_weights_sha256']
    assert hashes['V6'] == hashes['P6'] == hashes['V4+P2']
    rows=[{('run_id' if k=='seed' else k):v for k,v in r.items()} for r in rows]
    pd.DataFrame(rows).to_csv(out / task / 'run_metrics.csv', index=False)
    return rows


def validate(out, epochs, batch_size):
    train, _ = load_task('cylinder')
    rows, splits = [], []
    for split_seed in [701, 702]:
        order = np.random.default_rng(split_seed).permutation(len(train))
        n_val = int(np.ceil(.2 * len(train)))
        vi, ti = order[:n_val], order[n_val:]
        assert len(ti) == 420 and len(vi) == 105 and not set(ti) & set(vi)
        splits.append({'split_seed': split_seed, 'train_rows':ti.tolist(), 'validation_rows':vi.tolist()})
        a, b = train.iloc[ti], train.iloc[vi]
        y, _, ym, ys = standardize(a[['x','y']].to_numpy(), b[['x','y']].to_numpy())
        for config in CONFIGS:
            x, xt, _, _ = standardize(a[columns(config)].to_numpy(), b[columns(config)].to_numpy())
            seeds = [801, 802, 803]
            pred, _, _ = fit(x, y, xt, seeds, epochs, batch_size, f'validation/{split_seed}/{config}')
            distances = np.linalg.norm(pred * ys + ym - b[['x','y']].to_numpy()[None], axis=2).mean(axis=1)
            rows.extend(dict(split_seed=split_seed, seed=s, config=config, epochs=epochs,
                             batch_size=batch_size, mean_validation_distance_cm=float(d)) for s,d in zip(seeds,distances))
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'validation_metrics.csv', index=False)
    (out/'validation_splits.json').write_text(json.dumps(splits,indent=2))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['audit','benchmark','validate'])
    p.add_argument('--output', type=Path, default=Path(__file__).parent/'results')
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--batch-size', type=int, default=32)
    a = p.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if a.mode == 'audit':
        for task in TASKS:
            tr, te = load_task(task)
            print(task, len(tr), len(te), int((te.x<=27).sum()), {c:len(columns(c)) for c in CONFIGS})
        w, b = initial_parameters(12, [1001,1002])
        x = torch.randn(2,32,12)
        y = torch.randn(2,32,2)
        loss = (forward(x,w,b)-y).square().mean((1,2)).sum() + .0001 * sum(v.square().sum() for v in w)
        loss.backward()
        for j in range(2):
            sw = [v[j].detach().clone().requires_grad_() for v in w]
            sb = [v[j].detach().clone().requires_grad_() for v in b]
            z=x[j]
            for i,(v,c) in enumerate(zip(sw,sb)):
                z=z@v+c
                if i<3: z=torch.relu(z)
            sloss=(z-y[j]).square().mean()+.0001*sum(v.square().sum() for v in sw)
            sloss.backward()
            for v,s in zip(w,sw): torch.testing.assert_close(v.grad[j],s.grad,rtol=1e-5,atol=1e-6)
            for v,s in zip(b,sb): torch.testing.assert_close(v.grad[j],s.grad,rtol=1e-5,atol=1e-6)
        print('PASS: vectorized and standalone gradients agree; independent models retain loss scale.')
    elif a.mode == 'benchmark':
        tr,te=load_task('cylinder')
        x,xt,_,_=standardize(tr[columns('V6')].to_numpy(),te[columns('V6')].to_numpy())
        y,_,_,_=standardize(tr[['x','y']].to_numpy(),te[['x','y']].to_numpy())
        t=time.monotonic(); fit(x,y,xt,list(range(1001,1021)),10,32)
        print('20 models / 10 epochs seconds:',time.monotonic()-t)
    elif a.mode == 'validate':
        validate(a.output,a.epochs,a.batch_size)



if __name__ == '__main__':
    main()

