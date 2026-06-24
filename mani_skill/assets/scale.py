from stl import mesh
import numpy as np

# 원본 STL 경로
input_path = "/home/por1329/maniskill/ManiSkill/mani_skill/assets/cuchillo_cocina_v2.stl"
# 저장할 STL 경로
output_path = "/home/por1329/maniskill/ManiSkill/mani_skill/assets/knifef1.stl"

# scale factor
scale = 0.005

# STL 읽기
m = mesh.Mesh.from_file(input_path)

# 좌표 스케일 줄이기
m.vectors *= scale

# 저장
m.save(output_path)
print(f"Scaled STL saved to: {output_path}")
