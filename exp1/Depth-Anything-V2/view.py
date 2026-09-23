"""按模型、尺寸、评估方式和图片编号查看点云。"""

import argparse

import open3d as o3d

from experiment_paths import SIZES, result_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['relative', 'metric'], default='relative')
    parser.add_argument('--input-size', type=int, choices=SIZES, default=518)
    parser.add_argument('--protocol', choices=['aligned', 'raw'], default='aligned')
    parser.add_argument('--image-id', default='nyu_0001')
    parser.add_argument('--kind', choices=['pred', 'gt'], default='pred')
    args = parser.parse_args()
    try:
        folder = result_dir(args.model, args.input_size, args.protocol)
    except ValueError as exc:
        parser.error(str(exc))
    if not args.image_id.startswith('nyu_') or not args.image_id[4:].isdigit():
        parser.error('图片编号格式应为 nyu_0001')
    path = folder / f'{args.image_id}_{args.kind}.ply'
    if not path.is_file():
        parser.error(f'点云不存在，请先运行相应评估：{path}')
    cloud = o3d.io.read_point_cloud(str(path))
    if cloud.is_empty():
        parser.error(f'点云为空：{path}')
    o3d.visualization.draw_geometries(
        [cloud], front=[0, 0, -1], up=[0, -1, 0],
        lookat=cloud.get_center(), zoom=0.7)


if __name__ == '__main__':
    main()
