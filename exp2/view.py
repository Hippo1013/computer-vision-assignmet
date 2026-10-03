#!/usr/bin/env python3
"""交互点云查看：鼠标拖动旋转、滚轮缩放。"""
import argparse
import numpy as np
from mvsnet_local.io import ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--views", type=int, default=5, choices=[3, 5])
    p.add_argument("--kind", choices=["raw", "fusion"], default="fusion")
    p.add_argument("--prob-threshold", type=float, default=0.8)
    p.add_argument("--image-id", type=int, help="仅 raw 使用；省略则显示全部视角")
    p.add_argument("--screenshot", help="保存当前默认视角截图后关闭")
    args = p.parse_args()
    root = ROOT / "outputs" / f"views{args.views}"
    path = (
        root / "raw" / ("all.ply" if args.image_id is None else f"{args.image_id:08d}.ply")
        if args.kind == "raw"
        else root / "fusion" / f"prob_{round(args.prob_threshold*100):03d}" / "fused.ply"
    )
    if not path.exists():
        raise FileNotFoundError(f"请先生成点云：{path}")
    import open3d as o3d

    cloud = o3d.io.read_point_cloud(str(path))
    reference = np.load(root / "prediction/00000012.npz")
    rotation = reference["E"][:3, :3]
    front = -rotation[2]
    up = -rotation[1]
    lookat = cloud.get_center()
    if args.screenshot:
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1280, height=960, visible=False)
        vis.add_geometry(cloud)
        control = vis.get_view_control()
        control.set_front(front)
        control.set_up(up)
        control.set_lookat(lookat)
        control.set_zoom(0.72)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(args.screenshot, do_render=True)
        vis.destroy_window()
    else:
        o3d.visualization.draw_geometries(
            [cloud],
            window_name=str(path.relative_to(ROOT)),
            width=1280,
            height=960,
            front=front,
            up=up,
            lookat=lookat,
            zoom=0.72,
        )


if __name__ == "__main__":
    main()
