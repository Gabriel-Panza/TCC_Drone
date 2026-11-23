"""Show saved Baylands poses and reconstructed v21 maps, without PX4."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.pose_pb2 import Pose
from gz.transport13 import Node

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spatial_mapping.geometry import CameraIntrinsics, body_position_from_camera_transform
from spatial_mapping.navigation import SpatialNavigationConfig, SpatialNavigator
from spatial_mapping.visualization import render_top_down

WINDOW = 'Replay offline v21 - Mapa NED'
MODEL = 'x500_replay_v21'
WORLD = 'baylands_replay_v21'
ENU_FROM_NED = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
FRD_FROM_FLU = np.diag([1, -1, -1])


def prepare(source):
    run = source / 'inputs/run'
    report = json.loads((source / 'validation.json').read_text())
    metadata = json.loads((run / 'manifest.json').read_text())['metadata']
    audit = json.loads((source / 'mapping_audit.json').read_text())
    profile = json.loads((source / 'inputs/profile.json').read_text())
    navigator = SpatialNavigator(SpatialNavigationConfig(**audit['estimated_config']),
                                 temporal_revision_profile=profile)
    translation = metadata['camera_translation_body_m']
    mount = np.asarray(metadata['camera_rotation_body_from_optical'])
    plans = iter(report['planning']['plans'])
    positions, orientations, maps = [], [], []
    current_plan = None
    print('Reconstruindo o mapa monocular a partir dos 77 quadros gravados...', flush=True)
    events = (json.loads(line) for line in (run / 'events.jsonl').read_text().splitlines() if line.strip())
    for event in events:
        if event['event'] == 'frame' and event.get('file'):
            frame = Path(event['file'])
            with np.load(run / frame) as snapshot:
                transform = snapshot['camera_to_ned'].astype(float)
                intrinsics = CameraIntrinsics(*snapshot['intrinsics'])
            position = body_position_from_camera_transform(transform, translation, mount)
            rotation = ENU_FROM_NED @ transform[:3, :3] @ mount.T @ FRD_FROM_FLU
            depth = np.load(source / 'inputs/predictions' / f'{frame.stem}.npy').astype(float)
            navigator.integrate_depth(depth, intrinsics, transform, obstacle_reference_ned_z=position[2])
            positions.append(position)
            orientations.append(rotation)
            maps.append(None)
        elif (event['event'] == 'plan' and event.get('map_kind') == 'estimated'
              and positions and (event.get('plan') or {}).get('requested_goal_ned_m') is not None
              and (event['plan'].get('diagnostics') or {}).get('current_position_ned_m') is not None):
            current_plan = next(plans)
            assert current_plan['frames_integrated'] == len(positions)
        else:
            continue
        path = ()
        target = None
        if current_plan is not None:
            target = current_plan['requested_goal_ned_m']
            if (current_plan['frames_integrated'] == len(positions)
                    and current_plan['executable']):
                path = [current_plan['current_position_ned_m'], *current_plan['waypoints_ned_m']]
        image = render_top_down(navigator.grid, center_ned_m=positions[-1],
                                target_ned_m=target, path_ned_m=path, image_size=600)
        cv2.rectangle(image, (0, 0), (599, 61), (30, 25, 17), -1)
        cv2.putText(image, f'Replay offline v21 | Quadro {len(positions)}/{report["frames_integrated"]}',
                    (12, 25), cv2.FONT_HERSHEY_SIMPLEX, .61, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, 'Poses gravadas com ground truth | Mapa e planos v21',
                    (12, 48), cv2.FONT_HERSHEY_SIMPLEX, .49, (225, 225, 225), 1, cv2.LINE_AA)
        maps[-1] = image
    assert len(positions) == report['frames_integrated'] and next(plans, None) is None
    return np.asarray(positions), Rotation.from_matrix(orientations), maps


def make_world(px4, directory, first_position):
    tree = ET.parse(px4 / 'Tools/simulation/gz/worlds/baylands.sdf')
    world = tree.getroot().find('world')
    world.set('name', WORLD)
    for system, name in [('physics', 'Physics'), ('user-commands', 'UserCommands'),
                         ('scene-broadcaster', 'SceneBroadcaster')]:
        ET.SubElement(world, 'plugin', filename=f'gz-sim-{system}-system',
                      name=f'gz::sim::systems::{name}')
    drone = ET.SubElement(world, 'model', name=MODEL)
    ET.SubElement(drone, 'static').text = 'true'
    ET.SubElement(drone, 'pose').text = ' '.join(map(str, [*first_position, 0, 0, 0]))
    body = ET.SubElement(drone, 'include', merge='true')
    ET.SubElement(body, 'uri').text = 'model://x500_base'
    destination = directory / 'baylands_replay.sdf'
    tree.write(destination, encoding='utf-8')
    return destination


def set_pose(node, position, rotation):
    request = Pose(name=MODEL)
    request.position.x, request.position.y, request.position.z = map(float, position)
    (request.orientation.x, request.orientation.y,
     request.orientation.z, request.orientation.w) = map(float, rotation.as_quat())
    ok, response = node.request(f'/world/{WORLD}/set_pose', request, Pose, Boolean, 300)
    return ok and response.data


def gazebo_environment(px4):
    env = os.environ.copy()
    # The pip OpenCV build injects its private Qt plugin path on import.
    # Keep HighGUI's environment in Python and let Gazebo load system Qt.
    cv2_directory = Path(cv2.__file__).resolve().parent
    for variable in ('QT_QPA_PLATFORM_PLUGIN_PATH', 'QT_QPA_FONTDIR'):
        value = env.get(variable)
        if value and Path(value).resolve().is_relative_to(cv2_directory):
            del env[variable]
    resource_path = px4 / 'Tools/simulation/gz/models'
    env['GZ_SIM_RESOURCE_PATH'] = os.pathsep.join(filter(None, (
        str(resource_path), env.get('GZ_SIM_RESOURCE_PATH'),
    )))
    return env


def stop_process(process):
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def run(source, px4, seconds_per_frame):
    positions, rotations, maps = prepare(source)
    enu = positions @ ENU_FROM_NED.T
    interpolation = Slerp(np.arange(len(enu)), rotations)
    os.environ['GZ_PARTITION'] = f'tcc_v21_replay_{os.getpid()}'
    env = gazebo_environment(px4)
    with tempfile.TemporaryDirectory(prefix='tcc-v21-replay-') as directory:
        world = make_world(px4, Path(directory), enu[0])
        server = subprocess.Popen(['gz', 'sim', '-s', '-r', str(world)], env=env)
        gui = None
        try:
            node = Node()
            deadline = time.monotonic() + 60
            while not set_pose(node, enu[0], rotations[0]):
                if server.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError('O serviço de pose do Gazebo não ficou disponível.')
                time.sleep(.25)
            gui = subprocess.Popen(['gz', 'sim', '-g'], env=env)
            try:
                exit_code = gui.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
            else:
                raise RuntimeError(f'A janela do Gazebo encerrou durante a abertura (código {exit_code}).')
            cv2.imshow(WINDOW, maps[0])
            print('Organize as janelas, inicie a gravação e pressione ESPAÇO na janela NED.', flush=True)
            while gui.poll() is None and server.poll() is None:
                key = cv2.waitKey(50) & 255
                if key == ord(' '):
                    break
                if key in (27, ord('q')):
                    return
            started = time.monotonic()
            frame = 0
            while gui.poll() is None and server.poll() is None:
                progress = min((time.monotonic() - started) / seconds_per_frame, len(enu) - 1)
                index = int(progress)
                end = min(index + 1, len(enu) - 1)
                position = enu[index] + (progress - index) * (enu[end] - enu[index])
                if not set_pose(node, position, interpolation(progress)):
                    raise RuntimeError('O Gazebo rejeitou uma pose do replay.')
                if index != frame:
                    cv2.imshow(WINDOW, maps[index])
                    frame = index
                if (cv2.waitKey(45) & 255) in (27, ord('q')):
                    return
                if index == len(enu) - 1:
                    print('Replay concluído. Pressione Q no mapa NED para sair.', flush=True)
                    while (gui.poll() is None and server.poll() is None
                           and (cv2.waitKey(60) & 255) not in (27, ord('q'))):
                        pass
                    return
            if gui.poll() not in (None, 0) or server.poll() not in (None, 0):
                raise RuntimeError(f'O Gazebo encerrou durante o replay (GUI {gui.poll()}, servidor {server.poll()}).')
        finally:
            cv2.destroyAllWindows()
            stop_process(gui)
            stop_process(server)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'logs/reproducao_portavel_final_20261031')
    parser.add_argument('--px4', type=Path, default=Path.home() / 'PX4-Autopilot')
    parser.add_argument('--seconds-per-frame', type=float, default=.55)
    args = parser.parse_args()
    if args.seconds_per_frame <= 0:
        parser.error('--seconds-per-frame deve ser positivo')
    run(args.source.resolve(), args.px4.resolve(), args.seconds_per_frame)
