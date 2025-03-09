from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import torch
matplotlib.use('Qt5Agg')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def refract_ray_vectorized(incident, normal, n1, n2):
    """
    Compute the refracted ray direction (normalized) using torch vectorized operations.
    Parameters:
      incident: Tensor of shape (N,3) representing incident ray directions (normalized)
      normal:   Tensor of shape (3,) representing the interface normal (normalized, pointing upward)
      n1:       Refractive index of the incident medium
      n2:       Refractive index of the transmission medium
    Returns:
      refracted: Tensor of shape (M,3) representing valid refracted ray directions (normalized), M <= N;
      valid_idx: Indices of valid refracted rays in the original incident array (for subsequent intersection association)
    """
    cos_i = -torch.sum(incident * normal, dim=1)  # shape (N,)
    mask_flip = cos_i < 0
    cos_i_adjusted = torch.where(mask_flip, -cos_i, cos_i)
    normal_eff = normal.expand_as(incident)
    normal_eff = torch.where(mask_flip.unsqueeze(1), -normal_eff, normal_eff)

    ratio = n1 / n2
    sin_i_sq = torch.maximum(torch.tensor(
        0.0, device=device), 1.0 - cos_i_adjusted**2)
    full_reflection = (ratio**2 * sin_i_sq) > 1.0
    valid_mask = ~full_reflection

    incident_valid = incident[valid_mask]
    cos_i_valid = cos_i_adjusted[valid_mask]
    normal_eff_valid = normal_eff[valid_mask]
    sin_i_sq_valid = sin_i_sq[valid_mask]
    cos_t_valid = torch.sqrt(1.0 - ratio**2 * sin_i_sq_valid)
    refracted = ratio * incident_valid + \
        ((ratio * cos_i_valid - cos_t_valid).unsqueeze(1)) * normal_eff_valid
    norm_r = torch.linalg.norm(refracted, dim=1)
    norm_mask = norm_r >= 1e-12
    refracted = refracted[norm_mask] / norm_r[norm_mask].unsqueeze(1)
    valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
    final_indices = valid_indices[norm_mask]
    return refracted, final_indices


# Parameters
n_water = 1.33
n_air = 1.0
num_rings = 5
tol = 1e-4
n_crossing_rays = 3

source = np.array([-0.5, 0.0, -1.0], dtype=np.float64)
source_t = torch.tensor(source, dtype=torch.float64, device=device)

main_dir = np.array([0.0, 0.8, 1.0], dtype=np.float64)
main_dir = main_dir / np.linalg.norm(main_dir)
main_dir_t = torch.tensor(main_dir, dtype=torch.float64, device=device)

cone_angle_deg = 5
cone_angle = np.radians(cone_angle_deg)

# Rays
temp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
if abs(np.dot(temp, main_dir)) > 0.99:
    temp = np.array([0.0, 1.0, 0.0], dtype=np.float64)
u = np.cross(main_dir, temp)
u = u / np.linalg.norm(u)
v = np.cross(main_dir, u)
v = v / np.linalg.norm(v)

rays_list = []
rays_list.append(main_dir)

for i in range(num_rings):
    cos_theta = 1 - (i + 0.5) * (1 - np.cos(cone_angle)) / num_rings
    theta = np.arccos(cos_theta)
    spacing = (1 - np.cos(cone_angle)) / num_rings
    N_phi = int(np.round(2 * np.pi * np.sin(theta) / spacing))
    if N_phi < 1:
        N_phi = 1
    phi_vals = np.linspace(0, 2 * np.pi, N_phi, endpoint=False)
    for phi in phi_vals:
        direction = np.cos(theta) * main_dir + np.sin(theta) * \
            (np.cos(phi) * u + np.sin(phi) * v)
        direction = direction / np.linalg.norm(direction)
        rays_list.append(direction)

rays_np = np.array(rays_list)
rays = torch.tensor(rays_np, dtype=torch.float64, device=device)

# Refraction
ray_z = rays[:, 2]
mask_z = torch.abs(ray_z) >= 1e-12
rays_filtered = rays[mask_z]
t_water = -source_t[2] / rays_filtered[:, 2]  # Intersection parameter t
mask_t = t_water > 0
rays_water = rays_filtered[mask_t]
t_water = t_water[mask_t]
point_on_plane = source_t + t_water.unsqueeze(1) * rays_water

water_start = source_t.expand_as(point_on_plane)
water_segments = torch.stack([water_start, point_on_plane], dim=1)

plane_normal = torch.tensor(
    [0.0, 0.0, 1.0], dtype=torch.float64, device=device)
refracted, valid_idx = refract_ray_vectorized(
    rays_water, plane_normal, n_water, n_air)
p_on_plane_valid = point_on_plane[valid_idx]

# In-Air Path
r_z = refracted[:, 2]
p_z = p_on_plane_valid[:, 2]
t_air = torch.where(torch.abs(r_z) < 1e-12,
                    torch.tensor(1.0, device=device), (1.0 - p_z) / r_z)
t_air = torch.where(t_air < 0, torch.tensor(1.0, device=device), t_air)
end_in_air = p_on_plane_valid + t_air.unsqueeze(1) * refracted
air_segments = torch.stack([p_on_plane_valid, end_in_air], dim=1)

virtual_z = (n_air / n_water) * source[2]  # -0.75
t_virt = -virtual_z / refracted[:, 2]
virtual_pt = p_on_plane_valid - t_virt.unsqueeze(1) * refracted
virtual_segments = torch.stack([p_on_plane_valid, virtual_pt], dim=1)


# water_segments_cpu = water_segments.detach().cpu()
water_segments_cpu = water_segments.cpu().numpy()
air_segments_cpu = air_segments.cpu().numpy()
virtual_segments_cpu = virtual_segments.cpu().numpy()
virtual_pt_cpu = virtual_segments_cpu[:, 1, :]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# plane_size = 1
plane_size = 1.5
gridN = 10
xx = np.linspace(-plane_size, plane_size, gridN)
yy = np.linspace(-plane_size, plane_size, gridN)
XX, YY = np.meshgrid(xx, yy)
ZZ = np.zeros_like(XX)
ax.plot_surface(XX, YY, ZZ, alpha=0.3)

# underwater rays
for seg in water_segments_cpu:
    A, B = seg[0], seg[1]
    ax.plot([A[0], B[0]], [A[1], B[1]], [A[2], B[2]], color='blue')

# in-air rays
for seg in air_segments_cpu:
    A, B = seg[0], seg[1]
    ax.plot([A[0], B[0]], [A[1], B[1]], [A[2], B[2]], color='red')

# extended segments
for seg in virtual_segments_cpu:
    A, B = seg[0], seg[1]
    ax.plot([A[0], B[0]], [A[1], B[1]], [A[2], B[2]],
            linestyle='dashed', color='gray', alpha=0.03)

ax.scatter([source[0]], [source[1]], [source[2]], s=50, color='black')

# Intersection Points
def line_line_intersection(P1, d1, P2, d2):
    """
    Compute the closest point between two lines.
    Line 1: P1 + t * d1
    Line 2: P2 + s * d2
    Returns: (intersection point, t, s, distance error)
    If the lines are parallel, returns (None, None, None, None)
    """
    r = P1 - P2
    A = np.dot(d1, d1)
    B = np.dot(d1, d2)
    C = np.dot(d2, d2)
    D = np.dot(d1, r)
    E = np.dot(d2, r)
    denom = A * C - B * B
    if np.abs(denom) < 1e-8:
        return None, None, None, None
    t = (B * E - C * D) / denom
    s = (A * E - B * D) / denom
    Q1 = P1 + t * d1
    Q2 = P2 + s * d2
    midpoint = (Q1 + Q2) / 2.0
    error = np.linalg.norm(Q1 - Q2)
    return midpoint, t, s, error


# For each extended segment, extract the starting point P and direction d (normalized), and record its index
num_lines = virtual_segments_cpu.shape[0]
lines = []
for i in range(num_lines):
    P = virtual_segments_cpu[i, 0]
    d = virtual_segments_cpu[i, 1] - virtual_segments_cpu[i, 0]
    norm_d = np.linalg.norm(d)
    if norm_d < 1e-8:
        continue
    d = d / norm_d
    lines.append((i, P, d))  # index
    
    
intersection_records = []
for i in range(len(lines)):
    idx1, P1, d1 = lines[i]
    for j in range(i + 1, len(lines)):
        idx2, P2, d2 = lines[j]
        result = line_line_intersection(P1, d1, P2, d2)
        if result[0] is not None:
            midpoint, t_val, s_val, error = result
            if error < tol:
                intersection_records.append(
                    {'point': midpoint, 'lines': {idx1, idx2}})

# Cluster
clusters = []
for record in intersection_records:
    pt = record['point']
    lines_set = record['lines']
    found_cluster = False
    for cluster in clusters:
        if np.linalg.norm(pt - cluster['point']) < tol:
            cluster['records'].append(record)
            pts = [rec['point'] for rec in cluster['records']]
            cluster['point'] = np.mean(pts, axis=0)
            cluster['lines'].update(lines_set)
            found_cluster = True
            break
    if not found_cluster:
        clusters.append(
            {'point': pt, 'records': [record], 'lines': set(lines_set)})
for cluster in clusters:
    if len(cluster['lines']) >= n_crossing_rays:
        pt = cluster['point']
        ax.scatter(pt[0], pt[1], pt[2], color='green', s=20,
                   label='Intersection ({} rays)'.format(len(cluster['lines'])))

ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_zlim(-1.5, 1.5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Imaging of Underwater Object')

plt.show()
