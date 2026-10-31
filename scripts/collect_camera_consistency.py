#!/usr/bin/env python3
"""Isolated, no-flight Gazebo RGB/depth collection with commanded static poses."""
import argparse
from collections import OrderedDict, Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PX4 = ROOT.parent/'PX4-Autopilot'


def stamp_ns(header):
    return int(header.stamp.sec)*1_000_000_000+int(header.stamp.nsec)


def camera_transform(position, quaternion):
    x,y,z,w = quaternion
    rotation = np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                         [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                         [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
    # Gazebo camera link: X forward, Y left, Z up. Optical: Z forward,
    # X right, Y down. The experiment world is Gazebo world, not NED.
    optical = np.array([[0,0,1],[-1,0,0],[0,-1,0]])
    result = np.eye(4)
    result[:3,:3] = rotation@optical
    result[:3,3] = position
    return result


def pose_quaternion(pose):
    return np.array([0.,0.,np.sin(pose[3]/2),np.cos(pose[3]/2)]) if len(pose)==4 else np.asarray(pose[3:7],dtype=float)


def make_world(scene, world_name, destination=None):
    if scene=='baylands':
        tree = ET.parse(PX4/'Tools/simulation/gz/worlds/baylands.sdf')
        root = tree.getroot(); world = root.find('world'); world.set('name',world_name)
        # Local cached assets only. No download or modification of PX4 assets.
        for include in world.findall('include'):
            uri = include.find('uri')
            model = 'baylands' if 'baylands' in uri.text.lower() else 'coast water'
            cached = Path.home()/'.gz/fuel/fuel.gazebosim.org/openrobotics/models'/model/'3/model.sdf'
            if not cached.exists():
                raise FileNotFoundError(cached)
            model_tree=ET.parse(cached)
            for item in model_tree.getroot().iter():
                if item.tag not in ['uri','vertex','fragment'] or not item.text:continue
                value=item.text.strip()
                if '/files/' in value:
                    local=cached.parent/value.split('/files/',1)[1]
                elif value.startswith(('meshes/','materials/','media/')):
                    local=cached.parent/value
                else:continue
                if not local.exists():raise FileNotFoundError(local)
                item.text=str(local)
            local_sdf=destination/'assets'/model.replace(' ','_')/'model.sdf'
            local_sdf.parent.mkdir(parents=True,exist_ok=True)
            for resource in cached.parent.iterdir():
                if resource.is_dir():
                    link=local_sdf.parent/resource.name
                    if not link.exists():link.symlink_to(resource,target_is_directory=True)
            model_tree.write(local_sdf,encoding='unicode')
            uri.text = str(local_sdf)
    else:
        root = ET.Element('sdf',version='1.9'); world = ET.SubElement(root,'world',name=world_name)
        world.append(ET.fromstring('<physics name="audit" type="ignored"><max_step_size>0.004</max_step_size><real_time_factor>1</real_time_factor></physics>'))
        world.append(ET.fromstring('<scene><ambient>1 1 1 1</ambient><background>0.1 0.1 0.1 1</background><shadows>false</shadows></scene>'))
        for name,pose,size,color in [
            ('wall','10.05 0 1.5 0 0 0','0.1 60 40','0.9 0.1 0.1 1'),
            ('foreground','6.25 0 1.5 0 0 0','0.5 1 1','0.1 0.9 0.1 1')]:
            world.append(ET.fromstring(f'<model name="{name}"><static>true</static><pose>{pose}</pose><link name="body"><visual name="shape"><geometry><box><size>{size}</size></box></geometry><material><ambient>{color}</ambient><diffuse>{color}</diffuse></material></visual></link></model>'))
    for filename,name in [('physics','Physics'),('sensors','Sensors'),('user-commands','UserCommands'),('scene-broadcaster','SceneBroadcaster')]:
        plugin = ET.SubElement(world,'plugin',filename='gz-sim-'+filename+'-system',name='gz::sim::systems::'+name)
        if name=='Sensors': ET.SubElement(plugin,'render_engine').text='ogre2'
    rig = ET.SubElement(world,'model',name='camera_consistency_rig')
    ET.SubElement(rig,'static').text='true'
    ET.SubElement(rig,'pose').text='0 0 1.5 0 0 0.08'
    link = ET.SubElement(rig,'link',name='camera_link')
    for label,kind in [('rgb','camera'),('depth','depth_camera')]:
        sensor = ET.SubElement(link,'sensor',name=label,type=kind)
        ET.SubElement(sensor,'pose').text='0 0 0 0 0 0'
        ET.SubElement(sensor,'gz_frame_id').text='camera_consistency_rig/camera_link'
        ET.SubElement(sensor,'topic').text='/tcc_audit/'+label+'/image'
        ET.SubElement(sensor,'always_on').text='true'
        ET.SubElement(sensor,'update_rate').text='30'
        camera = ET.SubElement(sensor,'camera')
        ET.SubElement(camera,'camera_info_topic').text='/tcc_audit/'+label+'/camera_info'
        ET.SubElement(camera,'horizontal_fov').text='1.74'
        image = ET.SubElement(camera,'image')
        ET.SubElement(image,'width').text='1280'; ET.SubElement(image,'height').text='960'
        if label=='depth':ET.SubElement(image,'format').text='R_FLOAT32'
        clip = ET.SubElement(camera,'clip'); ET.SubElement(clip,'near').text='0.1'; ET.SubElement(clip,'far').text='3000'
    pose = ET.SubElement(rig,'plugin',filename='gz-sim-pose-publisher-system',name='gz::sim::systems::PosePublisher')
    for key,value in dict(publish_model_pose='true',publish_link_pose='true',use_pose_vector_msg='true',
                          static_publisher='true',static_update_frequency='30',update_frequency='30').items():
        ET.SubElement(pose,key).text=value
    return ET.tostring(root,encoding='unicode')


def decode_gz(msg):
    if msg.pixel_format_type==13:
        return np.frombuffer(msg.data,dtype='<f4').reshape(msg.height,msg.step//4)[:,:msg.width].copy()
    channels = {3:3,4:4,5:4,8:3}.get(msg.pixel_format_type)
    if channels is None:raise ValueError('unsupported Gazebo pixel format '+str(msg.pixel_format_type))
    rgb = np.frombuffer(msg.data,dtype=np.uint8).reshape(msg.height,msg.step)[:,:msg.width*channels].reshape(msg.height,msg.width,channels)[:,:,:3]
    if msg.pixel_format_type in [5,8]:rgb=rgb[:,:,::-1]
    return rgb.copy()


class Collector:
    def __init__(self, bridge, native_rpc=False):
        self.native_rpc=native_rpc
        from gz.transport13 import Node
        from gz.msgs10.image_pb2 import Image
        from gz.msgs10.camera_info_pb2 import CameraInfo
        from gz.msgs10.pose_v_pb2 import Pose_V
        self.node=Node(); self.lock=threading.Lock(); self.buffers={k:OrderedDict() for k in ['rgb','depth','ros_rgb','ros_depth']}
        self.seen={k:set() for k in self.buffers}
        self.info={};self.poses=[];self.stats=Counter();self.ros_node=None;self.ros_thread=None
        self.capture=False
        for label in ['rgb','depth']:
            self.node.subscribe(Image,'/tcc_audit/'+label+'/image',lambda m,key=label:self.image(m,key))
            self.node.subscribe(CameraInfo,'/tcc_audit/'+label+'/camera_info',lambda m,key=label:self.camera_info(m,key))
        self.node.subscribe(Pose_V,'/model/camera_consistency_rig/pose',self.pose)
        self.node.subscribe(Pose_V,'/model/camera_consistency_rig/pose_static',self.pose)
        self.bridge=bridge
        if bridge:
            import rclpy
            from sensor_msgs.msg import Image as RosImage
            from rclpy.qos import qos_profile_sensor_data
            rclpy.init(args=[])
            self.ros_node=rclpy.create_node('camera_consistency_capture')
            for label in ['rgb','depth']:
                self.ros_node.create_subscription(RosImage,'/tcc_audit/'+label+'/image',lambda m,key=label:self.ros_image(m,key),qos_profile_sensor_data)
            self.ros_thread=threading.Thread(target=rclpy.spin,args=(self.ros_node,),daemon=True);self.ros_thread.start()

    def image(self,msg,key):
        stamp=stamp_ns(msg.header)
        with self.lock:
            self.stats[key+'_messages']+=1
            self.seen[key].add(stamp)
            self.stats['latest_stamp_ns']=max(stamp,self.stats['latest_stamp_ns'])
            if not self.capture:return
            self.buffers[key][stamp]=(msg,time.perf_counter())
            while len(self.buffers[key])>16:self.buffers[key].popitem(last=False)

    def ros_image(self,msg,key):
        stamp=int(msg.header.stamp.sec)*1_000_000_000+int(msg.header.stamp.nanosec)
        with self.lock:
            self.stats['ros_'+key+'_messages']+=1
            self.seen['ros_'+key].add(stamp)
            if not self.capture:return
            self.buffers['ros_'+key][stamp]=(msg,time.perf_counter())
            while len(self.buffers['ros_'+key])>16:self.buffers['ros_'+key].popitem(last=False)

    def camera_info(self,msg,key):
        with self.lock:self.info[key]=msg

    def pose(self,msg):
        with self.lock:
            for p in msg.pose:
                if p.name=='camera_consistency_rig':
                    pose_stamp=stamp_ns(p.header) or stamp_ns(msg.header)
                    self.poses.append(dict(stamp_ns=pose_stamp,position=[p.position.x,p.position.y,p.position.z],
                        parent_frame=next((d.value[0] for d in p.header.data if d.key=='frame_id' and d.value),''),
                        quaternion_xyzw=[p.orientation.x,p.orientation.y,p.orientation.z,p.orientation.w]))
            self.poses=self.poses[-600:]

    def request(self,world,pose):
        from gz.msgs10.pose_pb2 import Pose
        msg=Pose(name='camera_consistency_rig')
        msg.position.x,msg.position.y,msg.position.z=pose[:3]
        q=pose_quaternion(pose)
        msg.orientation.x,msg.orientation.y,msg.orientation.z,msg.orientation.w=map(float,q)
        if self.native_rpc:
            from gz.msgs10.boolean_pb2 import Boolean
            for attempt in range(2):
                ok,response=self.node.request('/world/'+world+'/set_pose',msg,Pose,Boolean,300)
                if ok and response.data:
                    self.stats['native_pose_rpc_success']+=1
                    return
            self.stats['native_pose_rpc_fallback']+=1
        last=''
        for attempt in range(3):
            reply=subprocess.run(['gz','service','-s','/world/'+world+'/set_pose',
                '--reqtype','gz.msgs.Pose','--reptype','gz.msgs.Boolean',
                '--timeout','4000','--req',str(msg)],capture_output=True,text=True,timeout=8)
            last=reply.stdout+reply.stderr
            if reply.returncode==0 and 'data: true' in reply.stdout:return
            time.sleep(.2)
        raise RuntimeError('set_pose failed after bounded retries: '+last[-500:])

    def acquire(self,out,index,pose,count,timeout,settle_seconds=1.):
        time.sleep(settle_seconds)
        with self.lock:
            self.capture=True
            for b in self.buffers.values():b.clear()
            cutoff=self.stats['latest_stamp_ns']
        rows=[];deadline=time.monotonic()+timeout
        keys=['rgb','depth']+(['ros_rgb','ros_depth'] if self.bridge else [])
        while len(rows)<count and time.monotonic()<deadline:
            with self.lock:
                stamps=set(self.buffers[keys[0]])
                for key in keys[1:]:stamps &= set(self.buffers[key])
                stamps=sorted(s for s in stamps if s>cutoff)
                candidate=None
                for stamp in stamps:
                    matching=[p for p in self.poses if abs(p['stamp_ns']-stamp)<=100_000_000
                              and np.linalg.norm(np.asarray(p['position'])-pose[:3])<1e-4
                              and abs(abs(np.dot(p['quaternion_xyzw'],pose_quaternion(pose)))-1)<1e-5]
                    if matching:
                        candidate=(stamp,{k:self.buffers[k][stamp] for k in keys},min(matching,key=lambda p:abs(p['stamp_ns']-stamp)))
                        break
            if candidate is None:
                time.sleep(.02);continue
            stamp,matched,actual=candidate;cutoff=stamp
            rgb=decode_gz(matched['rgb'][0]);depth=decode_gz(matched['depth'][0])
            arrays=dict(rgb_rgb=rgb,reference_depth_m=depth,camera_to_world=camera_transform(actual['position'],actual['quaternion_xyzw']))
            row=dict(phase=index,stamp_ns=stamp,rgb_stamp_ns=stamp_ns(matched['rgb'][0].header),depth_stamp_ns=stamp_ns(matched['depth'][0].header),
                     commanded_pose=pose,commanded_pose_format='xyz_yaw' if len(pose)==4 else 'xyz_quaternion_xyzw',
                     measured_pose=actual,pose_stamp_delta_ns=actual['stamp_ns']-stamp,
                     rgb_pixel_format=matched['rgb'][0].pixel_format_type,depth_pixel_format=matched['depth'][0].pixel_format_type,
                     rgb_frame_id=next((d.value[0] for d in matched['rgb'][0].header.data if d.key=='frame_id' and d.value),''),
                     depth_frame_id=next((d.value[0] for d in matched['depth'][0].header.data if d.key=='frame_id' and d.value),''),
                     rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),depth_sha256=hashlib.sha256(depth.tobytes()).hexdigest())
            if self.bridge:
                for key,array in [('rgb',rgb),('depth',depth)]:
                    m=matched['ros_'+key][0]
                    dtype='<f4' if key=='depth' else 'u1'
                    received=np.frombuffer(m.data,dtype=dtype)
                    row['ros_'+key+'_encoding']=m.encoding
                    row['ros_'+key+'_frame_id']=m.header.frame_id
                    row['ros_'+key+'_payload_identical']=bytes(m.data)==matched[key][0].data
                    row['ros_'+key+'_arrival_delta_ms']=(matched['ros_'+key][1]-matched[key][1])*1000
                    if not row['ros_'+key+'_payload_identical']:arrays['ros_'+key+'_raw']=received.copy()
            name='phase_'+str(index)+'_pair_'+str(len(rows))+'.npz'
            np.savez_compressed(out/name,**arrays)
            row['file']=name;rows.append(row)
        with self.lock:self.capture=False
        if len(rows)<count:
            raise RuntimeError('capture timed out: '+json.dumps(dict(stats=dict(self.stats),buffers={k:len(v) for k,v in self.buffers.items()},poses=self.poses[-3:])))
        return rows

    def close(self):
        if self.ros_node is not None:
            import rclpy
            rclpy.shutdown();self.ros_thread.join(timeout=3);self.ros_node.destroy_node()


def stop(process):
    if process is None or process.poll() is not None:return
    os.killpg(process.pid,signal.SIGINT)
    try:process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGTERM)
        try:process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL);process.wait()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene',choices=['controlled','baylands'],default='controlled')
    parser.add_argument('--ros-bridge',action='store_true')
    parser.add_argument('--pairs-per-pose',type=int,default=5)
    parser.add_argument('--trajectory-run',type=Path,
                        help='Re-render every saved camera pose; stationary capture at each pose, never a flight.')
    parser.add_argument('--max-poses',type=int,help='Development smoke test only; records the truncated protocol.')
    parser.add_argument('--settle-seconds',type=float,default=1.)
    parser.add_argument('--pose-rpc',choices=['cli','native'],default='cli')
    args=parser.parse_args()
    if args.pairs_per_pose<1 or args.settle_seconds<0:
        parser.error('positive pair count and nonnegative settling time required')
    if args.trajectory_run and args.scene!='baylands':
        parser.error('trajectory replay requires --scene baylands')
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    out=ROOT/'logs/camera_consistency'/(stamp+'_'+args.scene);out.mkdir(parents=True,exist_ok=False)
    world='tcc_consistency_'+stamp
    os.environ['GZ_PARTITION']='tcc_consistency_'+stamp
    os.environ['ROS_DOMAIN_ID']='77'
    sdf=make_world(args.scene,world,out);(out/'world.sdf').write_text(sdf)
    poses=[[0.,0.,1.5,.08],[0.,-1.,1.5,0.],[.75,0.,1.5,.03],[0.,1.,1.5,-.03],[0.,0.,1.5,.08]]
    if args.scene=='baylands':
        from scipy.spatial.transform import Rotation
        legacy=ROOT/'datasets/spatial_mapping/run_20260827_165430_607620/frames'
        ned_to_world=np.array([[0,1,0],[1,0,0],[0,0,-1]])
        link_from_optical=np.array([[0,0,1],[-1,0,0],[0,-1,0]])
        known=[]
        for frame in [84,154,167]:
            with np.load(legacy/('frame_'+str(frame).zfill(6)+'.npz')) as sample:
                transform=sample['camera_to_ned']
            position=ned_to_world@transform[:3,3]
            rotation=ned_to_world@transform[:3,:3]@link_from_optical.T
            known.append([*position.tolist(),*Rotation.from_matrix(rotation).as_quat().tolist()])
        first=known[0]
        nearby_x=list(first);nearby_x[0]+=.75
        nearby_y=list(first);nearby_y[1]+=.75
        poses=[first,nearby_x,nearby_y,known[1],known[2],first]
    trajectory_frames=[]
    if args.trajectory_run:
        from scipy.spatial.transform import Rotation
        events=[json.loads(line) for line in (args.trajectory_run/'events.jsonl').read_text().splitlines() if line.strip()]
        trajectory_frames=[event['file'] for event in events if event.get('event')=='frame' and event.get('file')]
        if args.max_poses is not None:trajectory_frames=trajectory_frames[:args.max_poses]
        if not trajectory_frames:raise ValueError('trajectory has no frames')
        poses=[]
        for name in trajectory_frames:
            with np.load(args.trajectory_run/name) as sample:
                transform=sample['camera_to_ned']
            position=ned_to_world@transform[:3,3]
            rotation=ned_to_world@transform[:3,:3]@link_from_optical.T
            poses.append([*position.tolist(),*Rotation.from_matrix(rotation).as_quat().tolist()])
    (out/'protocol.json').write_text(json.dumps(dict(scene=args.scene,poses=poses,resolution=[1280,960],horizontal_fov=1.74,
        trajectory_run=None if args.trajectory_run is None else str(args.trajectory_run.resolve()),
        trajectory_frames=trajectory_frames,settle_seconds=args.settle_seconds,
        acquisition_mode='stationary_at_each_commanded_pose',
        pose_rpc=args.pose_rpc,
        original_sensor_model_sha256=hashlib.sha256((PX4/'Tools/simulation/gz/models/mono_cam/model.sdf').read_bytes()).hexdigest(),
        sdf_sha256=hashlib.sha256(sdf.encode()).hexdigest(),gz_partition=os.environ['GZ_PARTITION'],ros_domain=77,
        no_px4=True,no_flight=True,exact_rgb_depth_stamps_required=True),indent=2)+'\n')
    print('Collection: '+str(out),flush=True)
    (out/'collector_source.py').write_text(Path(__file__).read_text())
    collector=Collector(args.ros_bridge,args.pose_rpc=='native');server=bridge=None;rows=[]
    logs=[]
    try:
        handle=(out/'gazebo.log').open('w');logs.append(handle)
        server=subprocess.Popen(['gz','sim','-s','-r','--headless-rendering','--seed','42',str(out/'world.sdf')],stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        if args.ros_bridge:
            handle=(out/'bridge.log').open('w');logs.append(handle)
            bridge=subprocess.Popen(['/opt/ros/humble/lib/ros_gz_bridge/parameter_bridge',
                '/tcc_audit/rgb/image@sensor_msgs/msg/Image[gz.msgs.Image',
                '/tcc_audit/depth/image@sensor_msgs/msg/Image[gz.msgs.Image'],stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        deadline=time.monotonic()+60
        while len(collector.info)<2 and time.monotonic()<deadline:
            if server.poll() is not None:raise RuntimeError('Gazebo exited; see gazebo.log')
            time.sleep(.1)
        if len(collector.info)<2:raise RuntimeError('camera info not received')
        info={k:dict(width=int(v.width),height=int(v.height),k=list(v.intrinsics.k)) for k,v in collector.info.items()}
        (out/'camera_info.json').write_text(json.dumps(info,indent=2)+'\n')
        for key,msg in collector.info.items():(out/(key+'_camera_info.pb')).write_bytes(msg.SerializeToString())
        for index,pose in enumerate(poses):
            collector.request(world,pose)
            phase=collector.acquire(out,index,pose,args.pairs_per_pose,45,args.settle_seconds)
            if trajectory_frames:
                for row in phase:row['trajectory_frame_file']=trajectory_frames[index]
            rows.extend(phase)
            (out/'pairs.json').write_text(json.dumps(rows,indent=2)+'\n')
            print(json.dumps(dict(phase=index,pairs=len(phase),pose=pose)),flush=True)
        errors=[line for line in (out/'gazebo.log').read_text(errors='replace').splitlines()
                if any(marker in line for marker in ['does not exist','texture not found','Failed to load','Unable to load'])]
        stamps={k:sorted(v) for k,v in collector.seen.items()}
        (out/'stream_stamps.json').write_text(json.dumps(stamps,indent=2)+'\n')
        (out/'summary.json').write_text(json.dumps(dict(completed=True,pairs=len(rows),phases=len(poses),flight=False,
            stats=dict(collector.stats),render_resources_complete=not errors,render_resource_errors=errors,
            unique_stamps={k:len(v) for k,v in collector.seen.items()},
            native_exact_pair_count=len(collector.seen['rgb']&collector.seen['depth'])),indent=2)+'\n')
        if errors:raise RuntimeError('render resource errors; collection retained but not accepted')
        print('Completed without flight: '+str(out),flush=True)
    finally:
        stop(bridge);stop(server);collector.close()
        for handle in logs:handle.close()


if __name__=='__main__':main()
