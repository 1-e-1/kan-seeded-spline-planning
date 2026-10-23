"""Decode the full MP4 and verify dimensions, frame count and nonblank images."""
from pathlib import Path
import argparse,json,hashlib
import cv2,numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);args=p.parse_args()
    info=json.loads((args.directory/'manifest.json').read_text())
    video=args.directory/'kan_spline_motion_demo.mp4'
    assert hashlib.sha256(video.read_bytes()).hexdigest()==info['video_sha256']
    expected=sum(len(np.r_[np.arange(0,c['motion_duration_s'],1/info['fps']),c['motion_duration_s']])+2*info['fps'] for c in info['cases'])
    capture=cv2.VideoCapture(str(video));assert capture.isOpened()
    fps=capture.get(cv2.CAP_PROP_FPS);count=0;minimum_std=float('inf');different=0;previous=None
    snapshots={0,expected//2,expected-1};dimensions=set()
    while True:
        ok,frame=capture.read()
        if not ok:break
        dimensions.add((frame.shape[1],frame.shape[0]));minimum_std=min(minimum_std,float(frame.std()))
        if previous is not None and np.mean(abs(frame.astype(float)-previous))>.05:different+=1
        previous=frame.astype(float)
        if count in snapshots:assert cv2.imwrite(str(args.directory/f'decoded_frame_{count:04d}.png'),frame)
        count+=1
    capture.release()
    gate=count==expected and dimensions=={(info['width'],info['height'])} and abs(fps-info['fps'])<1e-6 and minimum_std>10 and different>expected*.35
    report=dict(gate=gate,frames=count,expected_frames=expected,fps=fps,duration_s=count/fps,
        dimensions=list(dimensions),nonstatic_frame_transitions=different,minimum_frame_std=minimum_std,
        file_bytes=video.stat().st_size,video_sha256=info['video_sha256'])
    (args.directory/'video_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(report);assert gate


if __name__=='__main__':main()
