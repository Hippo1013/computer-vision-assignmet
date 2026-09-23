"""课程实验的统一路径；所有相对文件路径以本仓库为基准。"""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'nyu_data'
RGB = DATA / 'rgb'
REFERENCE = DATA / 'reference'
OUTPUTS = ROOT / 'outputs'
SIZES = (280, 518)
GROUPS = tuple((model, size, protocol) for model, protocol in
               [('relative', 'aligned'), ('metric', 'raw'), ('metric', 'aligned')]
               for size in SIZES)
ALIGNED_PROTOCOL = 'inverse_depth_affine_gt_aligned_clipped_0.1_10m'
RAW_PROTOCOL = 'raw_metric_no_alignment_no_clipping'


def resolve_path(path):
    path = Path(path).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def relative_path(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def prediction_dir(model, size):
    if model not in ('relative', 'metric') or size <= 0:
        raise ValueError('模型须为 relative/metric，输入尺寸须为正数')
    return OUTPUTS / model / str(size) / 'prediction'


def result_dir(model, size, protocol):
    if (model, size, protocol) not in GROUPS:
        raise ValueError(f'不支持的实验组：{model}/{size}/{protocol}')
    return OUTPUTS / model / str(size) / protocol


def checkpoint(model, encoder):
    if model == 'relative':
        return ROOT / 'checkpoints' / f'depth_anything_v2_{encoder}.pth'
    return ROOT / 'metric_depth/checkpoints' / f'depth_anything_v2_metric_hypersim_{encoder}.pth'


def configure_inference_paths(args, model):
    """设置推理默认路径；自选输入须指定输出，避免混入 NYU 结果。"""
    args.img_path = str(resolve_path(args.img_path))
    if args.input_size <= 0:
        raise ValueError('input-size 必须为正数')
    if args.outdir is None and Path(args.img_path) != RGB:
        raise ValueError('自选输入请指定 --outdir，例如 outputs/examples/classroom')
    args.outdir = str(resolve_path(args.outdir) if args.outdir else
                      prediction_dir(model, args.input_size))
    if hasattr(args, 'load_from'):
        args.load_from = str(resolve_path(args.load_from) if args.load_from else
                             checkpoint(model, args.encoder))
    return args


def collect_predictions(model, sizes):
    """从计时记录读取原始预测清单，不扫描派生深度文件。"""
    jobs = []
    for size in dict.fromkeys(sizes):
        folder = prediction_dir(model, size)
        with (folder / 'inference_timing.json').open(encoding='utf-8') as handle:
            metadata = json.load(handle)
        if metadata['input_size'] != size:
            raise ValueError(f'{folder} 的输入尺寸与参数不符')
        if model == 'metric' and metadata.get('depth_unit') != 'meters':
            raise ValueError(f'{folder} 缺少米制深度标记')
        if metadata.get('model', model) != model:
            raise ValueError(f'{folder} 的模型标记不匹配')
        if model == 'relative' and metadata.get('depth_unit') == 'meters':
            raise ValueError(f'{folder} 不能使用米制预测作为相对逆深度')
        ids = [Path(item['filename']).stem for item in metadata['images']]
        if not ids or len(set(ids)) != len(ids) or metadata['image_count'] != len(ids):
            raise ValueError(f'{folder} 的图片清单为空、重复或数量不符')
        for sid in sorted(ids):
            path = folder / f'{sid}.npy'
            for required in (path, REFERENCE / f'{sid}.npz', RGB / f'{sid}.png'):
                if not required.is_file():
                    raise FileNotFoundError(required)
            jobs.append((path, size))
    return jobs
