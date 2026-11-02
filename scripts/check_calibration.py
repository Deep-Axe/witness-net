"""Generate local-only projection evidence. Run with .venv/bin/python.

No model APIs. Outputs contain dataset imagery: keep them under ignored data/.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from coopx.dataset.v2xreal import V2XReal, pose_to_site, project_camera, transform_points

EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7),
         (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))


def in_image(uv, valid, width, height):
    return valid & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)


def box_lines(image, points, intrinsic, color):
    """Clip box edges at the optical near plane before projecting."""
    height, width = image.shape[:2]
    drawn = False
    for first, second in EDGES:
        a, b = points[first].copy(), points[second].copy()
        if a[2] <= 0.1 and b[2] <= 0.1:
            continue
        if a[2] <= 0.1:
            a += (b - a) * ((0.101 - a[2]) / (b[2] - a[2]))
        if b[2] <= 0.1:
            b += (a - b) * ((0.101 - b[2]) / (a[2] - b[2]))
        uv, _, _ = project_camera(np.array([a, b]), intrinsic)
        ends = np.clip(np.rint(uv), -1_000_000, 1_000_000).astype(int)
        accepted, p, q = cv2.clipLine((0, 0, width, height), tuple(ends[0]), tuple(ends[1]))
        if accepted:
            cv2.line(image, p, q, color, 2, cv2.LINE_AA)
            drawn = True
    return drawn


def overlay(target, camera_name, source, output, *, cross=False):
    camera = target.cameras[camera_name]
    image = cv2.cvtColor(camera.image(), cv2.COLOR_RGB2BGR)
    height, width = image.shape[:2]
    xyz = source.lidar()[:, :3]
    xyz = xyz[np.linalg.norm(xyz, axis=1) > 1]
    pc = target.to_camera(xyz, camera_name, source_to_site=source.lidar_to_site)
    uv, depth, valid = project_camera(pc, camera.intrinsic)
    visible = in_image(uv, valid, width, height)
    indices = np.flatnonzero(visible)
    indices = indices[::max(1, len(indices) // 22_000)]
    colors = (cv2.applyColorMap(np.uint8(np.clip(depth[indices] / 70, 0, 1) * 255),
                               cv2.COLORMAP_TURBO).reshape(-1, 3)
              if len(indices) else np.empty((0, 3), dtype=np.uint8))
    for pixel, color in zip(uv[indices].astype(int), colors):
        cv2.circle(image, tuple(pixel), 1, tuple(map(int, color)), -1)
    box_count = 0
    enclosing_camera = []
    for box in source.boxes.values():
        if box.contains_site(target.camera_to_site(camera_name)[:3, 3]):
            enclosing_camera.append(box.object_id)
            continue  # Ego body surrounds the camera; its edges obscure the view.
        centre = target.to_camera(box.center_site()[None], camera_name)[0]
        if np.linalg.norm(centre) > 80:
            continue
        color = (230, 80, 230) if cross else ((0, 200, 255) if box.label == 'Pedestrian'
                                              else (80, 240, 80))
        corners = target.to_camera(box.corners_site(), camera_name)
        if box_lines(image, corners, camera.intrinsic, color):
            box_count += 1
            label_uv, _, label_valid = project_camera(centre[None], camera.intrinsic)
            if in_image(label_uv, label_valid, width, height)[0]:
                cv2.putText(image, f'{box.object_id}:{box.label}', tuple(label_uv[0].astype(int)),
                            0, 0.45, color, 1, cv2.LINE_AA)
    cv2.rectangle(image, (0, 0), (width, 92), (20, 20, 20), -1)
    title = f'{source.agent_id} LiDAR + source GT -> {target.agent_id}/{camera_name} | frame {target.frame_id}'
    for y, text, scale in ((29, title, 0.8),
                           (57, 'Points: optical depth 0-70 m | boxes: GT, not model detections', 0.65),
                           (81, 'Raw projections are NOT occlusion or visibility labels', 0.6)):
        cv2.putText(image, text, (16, y), 0, scale, (255, 255, 255), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(output), image):
        raise OSError(f'could not write {output}')
    return dict(file=str(output), target=target.agent_id, camera=camera_name,
                source=source.agent_id, projected_points=int(visible.sum()),
                drawn_points=len(indices), boxes_with_visible_edges=box_count,
                skipped_boxes_enclosing_camera=enclosing_camera)


def label_residuals(dataset, scenario, frame_id, vehicle, roadside):
    """Paired annotation discrepancies; NOT sensor timestamp measurements."""
    index = int(frame_id)
    available = dataset.aligned_frame_ids(scenario, ('1', '-1'))
    previous = dataset.read(scenario, '1', f'{max(int(available[0]), index - 1):06d}')
    following = dataset.read(scenario, '1', f'{min(int(available[-1]), index + 1):06d}')
    dt = (following.nominal_time_us - previous.nominal_time_us) / 1e6
    rows = []
    for object_id in sorted(vehicle.boxes.keys() & roadside.boxes.keys(), key=int):
        a, b = vehicle.boxes[object_id], roadside.boxes[object_id]
        delta = b.center_site() - a.center_site()
        row = dict(id=object_id, label=a.label, delta_site_m=delta.tolist(),
                   distance_3d_m=float(np.linalg.norm(delta)),
                   distance_xy_m=float(np.linalg.norm(delta[:2])))
        if dt > 0 and object_id in previous.boxes and object_id in following.boxes:
            velocity = (following.boxes[object_id].center_site()
                        - previous.boxes[object_id].center_site()) / dt
            row['nominal_speed_xy_mps'] = float(np.linalg.norm(velocity[:2]))
            if row['nominal_speed_xy_mps'] > 0.5:
                row['along_motion_equivalent_s'] = float(delta[:2] @ velocity[:2]
                                                         / (velocity[:2] @ velocity[:2]))
        pixels = {}
        for target, name in ((vehicle, 'cam1'), (roadside, 'cam2')):
            camera = target.cameras[name]
            pc = target.to_camera(np.array([a.center_site(), b.center_site()]), name)
            uv, _, valid = project_camera(pc, camera.intrinsic)
            with Image.open(camera.image_path) as image:
                visible = in_image(uv, valid, *image.size)
            if visible.all():
                pixels[f'{target.agent_id}/{name}'] = dict(
                    distance_px=float(np.linalg.norm(uv[1] - uv[0])),
                    vehicle_label_uv=uv[0].tolist(), roadside_label_uv=uv[1].tolist())
        row['paired_center_projection'] = pixels
        rows.append(row)
    return rows


def cloud_overlap(vehicle, roadside):
    """Different views and scan densities confound this overlap diagnostic."""
    base, query = vehicle.lidar()[:, :3], roadside.lidar()[:, :3]
    base = base[np.linalg.norm(base, axis=1) > 1]
    query = query[np.linalg.norm(query, axis=1) > 5][::2]
    base = transform_points(base, vehicle.lidar_to_site).astype(np.float32)
    query = transform_points(query, roadside.lidar_to_site).astype(np.float32)
    cv2.setRNGSeed(0)
    index = cv2.flann_Index(base, dict(algorithm=1, trees=4))
    _, distances = index.knnSearch(query, 1, params={'checks': 64})
    return dict(query_points=len(query), median_nearest_m=float(np.median(np.sqrt(distances))),
                fraction_within_0_5m=float((distances < 0.25).mean()),
                meaning='overlap diagnostic, not calibrated extrinsic error')


def measure_rgb_mover(vehicle, manual, output):
    """Coarse silhouette-rectangle comparison, separate from paired 3D labels."""
    box = vehicle.boxes[manual['id']]
    camera = vehicle.cameras['cam1']
    uv, _, valid = project_camera(vehicle.to_camera(box.corners_site(), 'cam1'), camera.intrinsic)
    if not valid.all():
        raise ValueError('manual mover measurement requires all box corners in front')
    projected_rect = np.r_[uv.min(axis=0), uv.max(axis=0)]
    observed_rect = np.array(manual['rgb_rect_xyxy'], dtype=float)
    observed_center = (observed_rect[:2] + observed_rect[2:]) / 2
    delta = (projected_rect[:2] + projected_rect[2:]) / 2 - observed_center
    site = transform_points(vehicle.lidar()[:, :3], vehicle.lidar_to_site)
    local = transform_points(site, np.linalg.inv(pose_to_site((*box.location, *box.angle))))
    inside = (np.abs(local - box.center) <= np.asarray(box.extent)).all(axis=1)
    projected, _, good = project_camera(vehicle.to_camera(site[inside], 'cam1'), camera.intrinsic)
    projected = projected[good]
    within_rect = ((projected >= observed_rect[:2]) & (projected <= observed_rect[2:])).all(axis=1)
    image = cv2.cvtColor(camera.image(), cv2.COLOR_RGB2BGR)
    for x, y in projected.astype(int):
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
    for rect, color in ((projected_rect, (0, 255, 0)), (observed_rect, (255, 255, 0))):
        cv2.rectangle(image, tuple(rect[:2].astype(int)), tuple(rect[2:].astype(int)), color, 2)
    low = np.maximum(np.minimum(observed_rect[:2], projected_rect[:2]).astype(int) - 45, 0)
    high = np.minimum(np.maximum(observed_rect[2:], projected_rect[2:]).astype(int) + 45,
                      [image.shape[1], image.shape[0]])
    crop = image[low[1]:high[1], low[0]:high[0]]
    crop = cv2.copyMakeBorder(crop, 72, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    for y, text in ((20, f"{vehicle.frame_id} car {box.object_id}: center offset {np.linalg.norm(delta):.1f}px"),
                    (42, 'cyan: manual RGB rectangle; green: projected GT'),
                    (63, 'red: LiDAR inside GT 3D box; edges uncertain 5-10px')):
        cv2.putText(crop, text, (8, y), 0, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(output), crop)
    return dict(id=box.object_id, rgb_rect_xyxy=observed_rect.tolist(),
                projected_gt_rect_xyxy=projected_rect.tolist(),
                projected_gt_minus_rgb_rect_center_px=delta.tolist(),
                rect_center_distance_px=float(np.linalg.norm(delta)),
                lidar_points_inside_gt_box=len(projected),
                fraction_projecting_inside_rgb_rect=float(within_rect.mean()) if len(projected) else None,
                warning='Coarse silhouette/box diagnostic, not a clock-offset or 3D error estimate',
                file=str(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data/v2x-real/val'))
    parser.add_argument('--scenario', default='2023-03-17-16-10-12_1_1')
    parser.add_argument('--frames', nargs='+', default=['000000', '000065', '000130'])
    parser.add_argument('--output', type=Path, default=Path('data/overlays'))
    parser.add_argument('--manual-movers', type=Path, default=Path('data/overlays/manual_movers.json'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    dataset = V2XReal(args.root)
    manual = json.loads(args.manual_movers.read_text()) if args.manual_movers.exists() else None
    if manual and manual['scenario'] != args.scenario:
        raise ValueError('manual mover annotations belong to a different scenario')
    result = dict(scenario=args.scenario, timestamp_basis='nominal 10Hz frame IDs only',
                  warning='No hardware timestamps: synchronization remains unproven.', frames=[])
    contacts = []
    for frame_id in args.frames:
        vehicle = dataset.read(args.scenario, '1', frame_id)
        roadside = dataset.read(args.scenario, '-1', frame_id)
        row = dict(frame=frame_id, nominal_time_us=vehicle.nominal_time_us, overlays=[])
        for source in (vehicle, roadside):
            for name in source.cameras:
                path = args.output / f'{frame_id}_agent{source.agent_id}_{name}_own.png'
                row['overlays'].append(overlay(source, name, source, path))
        tiles = [cv2.resize(cv2.imread(item['file']), (640, 360)) for item in row['overlays']]
        if len(tiles) == 6:
            cv2.imwrite(str(args.output / f'{frame_id}_all_cameras.png'),
                        np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])]))
        path = args.output / f'{frame_id}_rsu-1_to_vehicle1_cam1.png'
        row['overlays'].append(overlay(vehicle, 'cam1', roadside, path, cross=True))
        row['paired_labels'] = label_residuals(dataset, args.scenario, frame_id, vehicle, roadside)
        row['cloud_overlap'] = cloud_overlap(vehicle, roadside)
        if manual and frame_id in manual['frames']:
            row['rgb_mover_measurement'] = measure_rgb_mover(
                vehicle, manual['frames'][frame_id], args.output / f'{frame_id}_mover_alignment.png')
            row['rgb_mover_measurement']['method'] = manual['method']
        for name in (f'{frame_id}_agent1_cam1_own.png', f'{frame_id}_agent-1_cam2_own.png', path.name):
            contacts.append(cv2.resize(cv2.imread(str(args.output / name)), (960, 540)))
        result['frames'].append(row)
        print(f"{frame_id}: {len(row['overlays'])} overlays; {len(row['paired_labels'])} shared labels")
    sheet = np.vstack([np.hstack(contacts[i:i + 3]) for i in range(0, len(contacts), 3)])
    cv2.imwrite(str(args.output / 'calibration_contact_sheet.png'), sheet)
    (args.output / 'calibration_metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f'Evidence written to {args.output}')


if __name__ == '__main__':
    main()
