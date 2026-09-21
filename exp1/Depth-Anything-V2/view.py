import open3d as o3d
p = o3d.io.read_point_cloud('results_518/nyu_0001_pred.ply')
o3d.visualization.draw_geometries(
    [p], front=[0, 0, -1], up=[0, -1, 0],
    lookat=p.get_center(), zoom=0.7)