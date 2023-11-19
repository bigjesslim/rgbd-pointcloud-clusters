cd ..
MSBuild viz_cluster.sln /property:Configuration=Release
cd Release
cmd /K viz_cluster.exe ../../coords/fr1_xyz_last_fps.csv ../../clusters/clusters_fr1_xyz_last_020_10.txt

cd ..
MSBuild viz_cluster_2d.vcxproj /property:Configuration=Release
cd Release
cmd /K viz_cluster_2d.exe ../../coords/point_cloud_3_step5.csv ../../clusters/clusters_3_step5_002_10_pw_080.txt  ../../images/point_cloud_rgb_3.png

cd ..
MSBuild to_fp_cloud.vcxproj /property:Configuration=Release
cd Release
cmd /K to_fp_cloud.exe ../../fps/fr1_xyz_last_more.csv  ../../images/fr1_xyz_last_depth.png  ../../coords/fps_fr1_xyz_last_more.csv

cd ..
MSBuild to_point_cloud.vcxproj /property:Configuration=Release
cd Release
cmd /K to_point_cloud.exe ../../images/fr1_xyz_last_rgb.png ../../images/fr1_xyz_last_depth.png ../../coords/point_cloud_fr1_xyz_last.csv