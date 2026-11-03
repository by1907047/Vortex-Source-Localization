"""Run saved localization models without retraining or overwriting data."""

from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd
import torch


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from experiment import CONFIGS, TASKS, forward, metrics


def predict(task, config, input_csv=None):
    folder = ROOT / 'results' / task / config
    scalers = json.loads((folder / 'scalers.json').read_text())
    model = torch.load(folder / 'models.pt', map_location='cpu', weights_only=True)
    source_folder, _, test_file, *_ = TASKS[task]
    source = Path(input_csv) if input_csv is not None else ROOT / 'data' / source_folder / test_file
    data = pd.read_csv(source)
    if data.empty:
        raise ValueError('The input CSV contains no data rows.')
    missing = sorted(set(scalers['features']) - set(data.columns))
    if missing:
        raise ValueError('Missing feature columns: ' + ', '.join(missing))
    features = data[scalers['features']].to_numpy(dtype='float64')
    if not np.isfinite(features).all():
        raise ValueError('Features must contain finite numeric values.')
    x = ((features - scalers['input_mean']) / scalers['input_scale']).astype('float32')
    with torch.no_grad():
        pred = forward(
            torch.from_numpy(x).unsqueeze(0).expand(len(model['run_ids']), -1, -1),
            model['weights'], model['biases'],
        ).numpy().astype('float64')
    pred = pred * np.array(scalers['target_scale']) + np.array(scalers['target_mean'])
    if not np.isfinite(pred).all():
        raise ValueError('Prediction contains non-finite values; check the input scale.')
    if input_csv is None:
        # Run-level metrics avoid packaging a second copy of every prediction.
        keys = ['task', 'config', 'run_id', 'region', 'metric']
        computed = pd.DataFrame(metrics(
            data[['x', 'y']].to_numpy(), pred, task, config, model['run_ids'],
        )).rename(columns={'seed': 'run_id'}).set_index(keys).sort_index()
        reference = pd.read_csv(ROOT / 'results' / 'run_metrics.csv')
        reference = reference.loc[(reference.task == task) & (reference.config == config)]
        reference = reference.set_index(keys).sort_index()
        pd.testing.assert_index_equal(computed.index, reference.index)
        np.testing.assert_array_equal(computed.n_test, reference.n_test)
        np.testing.assert_allclose(computed.error_cm, reference.error_cm, rtol=0, atol=1e-4)
        print('PASS: full/near-field run-level metrics match within 0.0001 cm.')
        print('Max metric difference (cm):', np.abs(computed.error_cm - reference.error_cm).max())
    return model['run_ids'], data, pred


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=list(TASKS), default='cylinder')
    parser.add_argument('--config', choices=CONFIGS, default='V6+P12')
    parser.add_argument('--input', type=Path, help='Optional custom feature CSV.')
    parser.add_argument('--output', type=Path, help='New prediction CSV; existing files are refused.')
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        raise FileExistsError(f'Output already exists: {args.output}')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    run_ids, data, pred = predict(args.task, args.config, args.input)
    print('Models:', len(run_ids), '| Test rows:', len(data))
    print('Prediction shape:', pred.shape, '| Coordinates: cm')
    if {'x', 'y'}.issubset(data.columns):
        truth = data[['x', 'y']].to_numpy(dtype='float64')
        if not np.isfinite(truth).all():
            raise ValueError('Coordinates must contain finite numeric values.')
        error = pred - truth[None, :, :]
        per_run = pd.DataFrame({
            'MAE_x_cm': np.abs(error[:, :, 0]).mean(axis=1),
            'MAE_y_cm': np.abs(error[:, :, 1]).mean(axis=1),
            'Mean_distance_cm': np.linalg.norm(error, axis=2).mean(axis=1),
        })
        print(per_run.agg(['mean', 'std']).round(6).to_string())
    else:
        print('First three predictions for the first saved run (cm):')
        print(pred[0, :3])
    if args.output is not None:
        export = pd.DataFrame({
            'run_id': np.repeat(run_ids, len(data)),
            'test_row': np.tile(np.arange(len(data)), len(run_ids)),
            'pred_x_cm': pred[:, :, 0].reshape(-1),
            'pred_y_cm': pred[:, :, 1].reshape(-1),
        })
        with args.output.open('x', encoding='utf-8', newline='') as stream:
            export.to_csv(stream, index=False)
        print('Saved:', args.output.resolve())


if __name__ == '__main__':
    main()
