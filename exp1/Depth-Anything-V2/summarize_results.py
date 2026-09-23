"""读取六组 metrics.csv，自动生成 outputs/summary 的逐图表和均值表。"""

import csv
import math

from experiment_paths import (
    GROUPS, OUTPUTS, ALIGNED_PROTOCOL, RAW_PROTOCOL,
    prediction_dir, result_dir, relative_path,
)


METRICS = ('abs_rel', 'depth_rmse_m', 'xyz_rmse_m')
FIELDS = ('model', 'input_size', 'protocol', 'image_id', 'valid_pixels',
          'depth_processing', 'prediction_dir', 'output_dir',
          'alignment_a', 'alignment_b', *METRICS)


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def normalize_row(row, model, size, protocol):
    """统一列名；路径存仓库相对路径，移动工作区后仍可使用。"""
    return {
        'model': model, 'input_size': size, 'protocol': protocol,
        'image_id': row['image_id'], 'valid_pixels': row['valid_pixels'],
        'depth_processing': ALIGNED_PROTOCOL if protocol == 'aligned' else RAW_PROTOCOL,
        'prediction_dir': relative_path(prediction_dir(model, size)),
        'output_dir': relative_path(result_dir(model, size, protocol)),
        'alignment_a': row.get('alignment_a', ''),
        'alignment_b': row.get('alignment_b', ''),
        'abs_rel': row.get('abs_rel', row.get('aligned_abs_rel')),
        'depth_rmse_m': row['depth_rmse_m'], 'xyz_rmse_m': row['xyz_rmse_m'],
    }


def save_group(rows, model, size, protocol, merge=False):
    path = result_dir(model, size, protocol) / 'metrics.csv'
    previous = read_csv(path) if merge and path.is_file() else []
    indexed = {r['image_id']: r for r in previous}
    indexed.update({r['image_id']: normalize_row(r, model, size, protocol) for r in rows})
    write_csv(path, [indexed[sid] for sid in sorted(indexed)], FIELDS)
    return path


def rebuild_summary():
    all_rows, means = [], []
    for model, size, protocol in GROUPS:
        path = result_dir(model, size, protocol) / 'metrics.csv'
        if not path.is_file():
            continue
        group = read_csv(path)
        if not group or len({r['image_id'] for r in group}) != len(group):
            raise ValueError(f'{path} 为空或有重复图片')
        for row in group:
            if (row['model'], int(row['input_size']), row['protocol']) != (model, size, protocol):
                raise ValueError(f'{path} 中存在其他实验组的记录')
            expected = normalize_row(row, model, size, protocol)
            for key in ('depth_processing', 'prediction_dir', 'output_dir'):
                if row[key] != expected[key]:
                    raise ValueError(f'{path} 的 {key} 与当前实验组不符')
            if any(not math.isfinite(float(row[k])) or float(row[k]) < 0 for k in METRICS):
                raise ValueError(f'{path} 的指标必须是有限非负值')
        all_rows.extend(group)
        means.append({
            'model': model, 'input_size': size, 'protocol': protocol,
            'image_count': len(group), 'averaging_method': 'equal_weight_per_image',
            'depth_processing': group[0]['depth_processing'],
            **{k + '_mean': sum(float(r[k]) for r in group) / len(group) for k in METRICS},
        })
    if not all_rows:
        raise ValueError('没有评估明细，请先运行几何评估脚本')
    write_csv(OUTPUTS / 'summary/all_metrics.csv', all_rows, FIELDS)
    write_csv(OUTPUTS / 'summary/means.csv', means)
    print(f'汇总完成：{len(all_rows)} 条逐图记录，{len(means)} 组均值 → {OUTPUTS / "summary"}')
    return all_rows, means


if __name__ == '__main__':
    rebuild_summary()
