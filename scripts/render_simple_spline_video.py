"""Evidence-linked MP4 replay of the latest spline reference trajectories.

No planning, training, scene editing, payload attachment or new success trial.
All displayed meshes are the portable scene's cooked collision hulls.
"""
from pathlib import Path
import sys,json,argparse,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import imageio_ffmpeg
from src.geometry.portable_scene import PortableScene
from src.planning.cubic_execution_curve import CubicExecutionCurve
from src.planning.c2_trajectory_buffer import C2TrajectoryBuffer

plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
    'axes.unicode_minus':False,'font.size':10,'figure.facecolor':'#f5f7fb',
    'axes.facecolor':'#ffffff','animation.ffmpeg_path':imageio_ffmpeg.get_ffmpeg_exe()})


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def load_case(model,trial,row,fps):
    index=row['index'];assert row['handoff']['accepted'] and row['updated_goal_reached']
    original=ROOT/'results/sota_simple_v1/kan_normalized_tcp01_k32_i200_dev350'/f'trajectory_{index:04d}.npz'
    with np.load(original) as a:
        old=CubicExecutionCurve(a['native_knots']+model.q_ref,a['native_start']+model.q_ref,a['native_goal']+model.q_ref,
            (len(a['q'])-1)*float(a['dt']))
    with np.load(trial/f'candidate_{index:04d}.npz') as a:
        new=CubicExecutionCurve(a['native_knots']+model.q_ref,a['native_start']+model.q_ref,a['native_goal']+model.q_ref,
            (len(a['q'])-1)*float(a['dt']),a['native_start_velocity'],a['native_start_acceleration'])
    buffer=C2TrajectoryBuffer(old);switch=row['requested_switch_s']
    assert buffer.schedule(new,switch,row['ready_s'],independently_verified=True)['accepted']
    duration=switch+new.duration;times=np.r_[np.arange(0,duration,1/fps),duration]
    states=[buffer.reference_at(float(t)) for t in times]
    arrays={k:np.array([s[k] for s in states]) for k in states[0]}
    metric_time=np.r_[np.arange(0,duration,1/240),duration]
    metric_states=[buffer.reference_at(float(t)) for t in metric_time]
    metrics={k:np.array([s[k] for s in metric_states]) for k in ('velocity','acceleration','jerk')}
    transforms=np.array([model.forward(q)[0] for q in arrays['position']])
    tool=model.index[model.meta['tool_path']]
    return dict(index=index,row=row,time=times,state=arrays,transforms=transforms,
        tcp=transforms[:,tool,:3,3],duration=duration,metric_time=metric_time,metrics=metrics,
        source_sha256={str(original.relative_to(ROOT)):sha(original),
                      str((trial/f'candidate_{index:04d}.npz').relative_to(ROOT)):sha(trial/f'candidate_{index:04d}.npz')})


def meshes(model):
    dynamic=[];fixed=[]
    for path,hulls in model.hulls.items():
        if path.startswith(('/Bolt_','/Nut_','/Cube_')) or path=='/World/GroundPlane':continue
        index=model.index[path];moving=path.startswith('/World/Base_01/') or path==model.meta['tool_path']
        for vertices in hulls:
            item=dict(path=path,index=index,vertices=vertices,faces=ConvexHull(vertices).simplices)
            (dynamic if moving else fixed).append(item)
    return dynamic,fixed


def triangle(mesh,pose):
    vertices=mesh['vertices']@pose[:3,:3].T+pose[:3,3]
    return vertices[mesh['faces']]


def prepare(case,model,dynamic,fixed,ordinal):
    fig=plt.figure(figsize=(14.4,8.1),dpi=100)
    grid=fig.add_gridspec(3,2,width_ratios=[1.9,1],left=.025,right=.97,bottom=.13,top=.84,wspace=.12,hspace=.46)
    world=fig.add_subplot(grid[:,0],projection='3d');charts=[fig.add_subplot(grid[i,1]) for i in range(3)]
    for mesh in fixed:
        path=mesh['path']
        color='#9ba9bb' if 'Astronaut' in path else '#4b637a' if 'lanzi' in path else '#b0bac7'
        alpha=.12 if 'Astronaut' in path else .28 if 'lanzi' in path else .32
        world.add_collection3d(Poly3DCollection(triangle(mesh,model.home[mesh['index']]),
            facecolor=color,edgecolor='none',linewidth=0,alpha=alpha))
    artists=[]
    for mesh in dynamic:
        tool=mesh['path']==model.meta['tool_path']
        artist=Poly3DCollection(triangle(mesh,case['transforms'][0,mesh['index']]),
            facecolor='#2563eb' if tool else '#eb963b',edgecolor='#1d4ed8' if tool else '#b76f22',linewidth=.1,alpha=.96)
        world.add_collection3d(artist);artists.append(artist)
    # Derive camera extent from actual moving volumes, not an origin-to-origin skeleton.
    points=[]
    for frame in range(0,len(case['time']),max(1,len(case['time'])//24)):
        for mesh in dynamic:
            pose=case['transforms'][frame,mesh['index']]
            points.append(mesh['vertices']@pose[:3,:3].T+pose[:3,3])
    points=np.concatenate(points);lo=points.min(0)-.12;hi=points.max(0)+.12
    lo=np.minimum(lo,[-.35,.0,.63]);hi=np.maximum(hi,[.35,.85,1.5])
    center=(lo+hi)/2;size=np.maximum(hi-lo,.7)
    world.set(xlim=(center[0]-size[0]/2,center[0]+size[0]/2),
              ylim=(center[1]-size[1]/2,center[1]+size[1]/2),zlim=(center[2]-size[2]/2,center[2]+size[2]/2),
              xlabel='世界 X / m',ylabel='世界 Y / m',zlabel='世界 Z / m')
    world.set_box_aspect(size);world.view_init(elev=24,azim=38);world.tick_params(labelsize=8)
    world.set_title('左臂：橙色   电磁铁：蓝色   固定结构/右臂：灰色',fontsize=11,pad=10)
    world.plot(*case['tcp'].T,color='#95a5ba',lw=1,ls='--',label='工具原点参考路径')
    trail,=world.plot([],[],[],color='#2563eb',lw=2.5)
    marker,=world.plot([],[],[],marker='o',ms=5,color='#2563eb')
    world.scatter(*case['tcp'][-1],marker='*',s=100,color='#16a34a',label='更新后的工具原点目标')
    world.legend(loc='lower left',fontsize=8)
    cursors=[];lines=[];labels=[]
    for ax,key,color,title,unit,limit in zip(charts,['velocity','acceleration','jerk'],
            ['#2563eb','#d97706','#7c3aed'],['关节速度','关节加速度','关节 jerk'],['rad/s','rad/s²','rad/s³'],[1.5,4,30]):
        values=np.max(abs(case['metrics'][key]),axis=1)
        style='steps-post' if key=='jerk' else 'default'
        ax.plot(case['metric_time'],values,color=color,lw=1.2,alpha=.22,drawstyle=style)
        line,=ax.plot([],[],color=color,lw=1.8,drawstyle=style);lines.append((line,values))
        ax.axhline(limit,color='#dc2626',ls='--',lw=.9,label='限值')
        ax.axvline(case['row']['requested_switch_s'],color='#16a34a',ls=':',lw=1.2)
        ax.set(xlim=(0,case['duration']),ylim=(0,limit*1.16),ylabel=unit,xlabel='运动时间 / s')
        labels.append(ax.set_title(title+'：各关节绝对值的最大值',fontsize=10,loc='left'))
        ax.grid(alpha=.17);cursors.append(ax.axvline(0,color='#334155',lw=.8))
    fig.text(.035,.955,'KAN-Transformer 初值 → 单一约束样条优化 → 关节轨迹执行',fontsize=17,weight='bold',color='#15304d')
    status=fig.text(.035,.905,'',fontsize=12,color='#334155')
    fig.text(.035,.07,'原始碰撞几何显示｜右臂固定为 0｜无携物｜曲线及衔接已做密集采样几何与导数复核',fontsize=10,color='#475569')
    fig.text(.035,.037,'已验证记录的 FK 参考回放，不是 Isaac/实物录像或抓取成功试验。运动片段约 1×，案例首尾各停留 1 s。',fontsize=9,color='#64748b')
    def update(frame):
        time=float(case['time'][frame])
        for artist,mesh in zip(artists,dynamic):artist.set_verts(triangle(mesh,case['transforms'][frame,mesh['index']]))
        points=case['tcp'][:frame+1];trail.set_data(points[:,0],points[:,1]);trail.set_3d_properties(points[:,2])
        current=case['tcp'][frame];marker.set_data([current[0]],[current[1]]);marker.set_3d_properties([current[2]])
        for cursor,(line,values) in zip(cursors,lines):
            last=int(np.searchsorted(case['metric_time'],time,side='right'))
            cursor.set_xdata([time,time]);line.set_data(case['metric_time'][:last],values[:last])
        phase=('求解与几何复核；继续原轨迹' if time<case['row']['ready_s'] else
               '候选验收完成；等待衔接' if time<case['row']['requested_switch_s'] else
               '沿更新曲线运动（q/v/a 连续衔接）' if time<case['duration'] else '到达更新目标并停稳')
        status.set_text(f"案例 {ordinal}/3 · task {case['index']}   |   t={time:.2f} / {case['duration']:.2f} s   |   {phase}")
    return fig,update


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--preview-only',action='store_true')
    p.add_argument('--fps',type=int,default=24);args=p.parse_args()
    if args.preview_only:assert not args.output.exists()
    args.output.mkdir(parents=True,exist_ok=True)
    video=args.output/'kan_spline_motion_demo.mp4';assert not video.exists()
    trial=ROOT/'results/sota_simple_v1/wall_clock_handoff_dev30_parallel_v1'
    status=json.loads((trial/'status.json').read_text());audit=json.loads((ROOT/'results/sota_simple_v1/wall_clock_reference_geometry_audit_v1.json').read_text())
    assert status['status']=='complete' and audit['gate'] and sha(trial/'status.json')==audit['trial_status_sha256']
    model=PortableScene(ROOT/'results/sota_simple_v1/scene_model.json');assert sha(ROOT/'finally.usd')==model.meta['scene_sha256']
    rows={r['index']:r for r in status['rows']};cases=[load_case(model,trial,rows[i],args.fps) for i in (13,227,63)]
    dynamic,fixed=meshes(model);panels=[];manifest=[]
    for ordinal,case in enumerate(cases,1):
        fig,update=prepare(case,model,dynamic,fixed,ordinal);panels.append((fig,update,case))
        for label,frame in [('start',0),('handoff',int(np.argmin(abs(case['time']-1.6)))),('finish',len(case['time'])-1)]:
            update(frame);fig.savefig(args.output/f"case_{case['index']}_{label}.png",dpi=100)
        manifest.append(dict(index=case['index'],motion_duration_s=case['duration'],source_sha256=case['source_sha256']))
    if not args.preview_only:
        writer=FFMpegWriter(fps=args.fps,codec='libx264',bitrate=4500,extra_args=['-pix_fmt','yuv420p','-movflags','+faststart'],
                           metadata={'title':'Verified KAN spline reference replay','comment':'No grasp or hardware execution claim'})
        # FFMpegWriter reads the figure assigned to it; all panels have identical dimensions.
        with writer.saving(panels[0][0],str(video),dpi=100):
            for fig,update,case in panels:
                writer.fig=fig
                frames=[0]*args.fps+list(range(len(case['time'])))+[len(case['time'])-1]*args.fps
                for n,frame in enumerate(frames):
                    update(frame);writer.grab_frame()
                    if n%120==0:print('RENDER',case['index'],n,len(frames),flush=True)
        manifest_data=dict(status='rendered',fps=args.fps,width=1440,height=810,cases=manifest,
            scene_sha256=model.meta['scene_sha256'],reference_audit_sha256=sha(ROOT/'results/sota_simple_v1/wall_clock_reference_geometry_audit_v1.json'),
            video_sha256=sha(video),scope='FK reference replay of verified development records; collision hull visualization; not grasp/Isaac/hardware video')
        (args.output/'manifest.json').write_text(json.dumps(manifest_data,indent=2),encoding='utf-8')
    for fig,_,_ in panels:plt.close(fig)
    print('PREVIEWS_READY' if args.preview_only else 'VIDEO_READY',str(args.output),flush=True)


if __name__=='__main__':main()
