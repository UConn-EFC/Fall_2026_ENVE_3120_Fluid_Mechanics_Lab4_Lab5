"""Transparent teaching calculations for ENVE 3120 Labs 4 and 5.
SI units internally. Darcy (not Fanning) friction factors. No proprietary engine.
Read the notebooks for model scope and the provenance of each assumption.
"""
from pathlib import Path
import copy, json, logging, sys, zipfile
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq, least_squares

G = 9.81
NU = 1.004e-6
RHO = 998.2
NAVY, BLUE, GOLD, PURPLE, RED = '#17345b', '#2c70b7', '#b77712', '#8765a8', '#b44852'
plt.rcParams.update({'figure.dpi':110,'font.size':11,'axes.spines.top':False,
    'axes.spines.right':False,'axes.labelcolor':NAVY,'text.color':NAVY,
    'axes.titleweight':'bold','axes.titlesize':13,'savefig.bbox':'tight',
    'axes.prop_cycle':plt.cycler(color=[BLUE,GOLD,PURPLE,RED,NAVY]),
    'mathtext.fontset':'stix','svg.fonttype':'none','figure.facecolor':'white'})

from IPython.display import FileLink as _FileLink

class DownloadLink(_FileLink):
    """Keep notebook download URLs and plain-text previews portable."""
    def __init__(self,path):
        super().__init__(Path(path).as_posix())
    def __repr__(self):
        return self.path


def start_log(name, folder='outputs'):
    """A fresh, rerun-safe console and file logger for this notebook session."""
    folder=Path(folder);folder.mkdir(exist_ok=True,parents=True)
    log=logging.getLogger(name);log.setLevel(logging.INFO);log.propagate=False
    for handler in log.handlers[:]:
        handler.close();log.removeHandler(handler)
    fmt=logging.Formatter('%(asctime)s | %(levelname)-7s | %(message)s',datefmt='%H:%M:%S')
    for h in [logging.StreamHandler(sys.stdout),logging.FileHandler(folder/f'{name}.log',mode='w',encoding='utf8')]:
        h.setFormatter(fmt);log.addHandler(h)
    log.info('Session started. SI units; Darcy friction factor; logs also saved to %s.',folder/f'{name}.log')
    return log

def step(log, number, title):
    log.info('STEP %s | %s',number,title)

def friction(q, D, eps, nu=NU):
    """Laminar 64/Re; turbulent Colebrook; a declared linear transition blend.
    Returns NaN f at exactly zero flow (head loss is evaluated separately).
    """
    if not np.isfinite([q,D,eps,nu]).all() or D<=0 or nu<=0 or eps<0 or eps>=D:
        raise ValueError('Flow must be finite; D and nu positive; roughness between zero and D.')
    re=4*abs(float(q))/(np.pi*D*nu)
    if re==0:return np.nan,0.
    if re<=2000:return 64/re,re
    def turbulent(r):
        x=brentq(lambda x:x+2*np.log10(eps/(3.7*D)+2.51*x/r),1.,100.)
        return 1/x**2
    if re>=4000:return turbulent(re),re
    w=(re-2000)/2000
    return (1-w)*64/re+w*turbulent(4000),re

def pipe_loss(q,L,D,eps=0.,f=None,K=0.,nu=NU):
    if not np.isfinite([q,L,D,eps,K,nu]).all() or L<=0 or D<=0 or nu<=0 or K<0 or eps<0 or eps>=D:
        raise ValueError('Invalid pipe input: use finite values, positive length/diameter/viscosity, and nonnegative losses/roughness.')
    if f is not None and (not np.isfinite(f) or f<=0):raise ValueError('Darcy f must be finite and positive.')
    if q==0:return 0.
    if f is None:f,_=friction(q,D,eps,nu)
    area=np.pi*D**2/4
    return (f*L/D+K)*q*abs(q)/(2*G*area**2)

def lab4_config():
    return dict(reservoir_heads={'R-1':42.6,'R-2':21.3,'R-3':8.5},
        junction2_elevation=21.3,
        pipes=pd.DataFrame({
          'pipe':['P-1','P-2','P-3','P-4','P-5','P-6'],
          'start':['R-1','J-1','J-2','J-3','J-2','J-4'],
          'end':['J-1','J-2','J-3','R-2','J-4','R-3'],
          'L_m':[1420.3,1065.3,327.8,355.1,710.2,852.2],
          'D_m':[.8,1.,.6,.4,.8,.6],
          'f_D':[.02,.015,.025,.03,.02,.025]}).set_index('pipe'),
        # These K values are illustrative assumptions, not supplied measurements.
        bend_K={'R-1':.35,'R-2':.20,'R-3':.55},
        entrance_K=.5,exit_K=1.,contraction_K=.5,tee_K=.9,
        include_minor=True,friction_mode='fixed',roughness_m={},nu=NU,
        atmosphere_Pa=101325.,vapor_Pa=2339.,density=RHO)

# Each branch below is ordered from its reservoir toward J-2.
BRANCHES={'R-1':['P-1','P-2'],'R-2':['P-4','P-3'],'R-3':['P-6','P-5']}

def validate4(cfg):
    p=cfg['pipes']
    if list(p.index)!=['P-1','P-2','P-3','P-4','P-5','P-6']:
        raise ValueError('Keep all six pipe labels, in the original order.')
    if not np.isfinite(p[['L_m','D_m','f_D']].to_numpy()).all() or (p[['L_m','D_m','f_D']]<=0).any().any():
        raise ValueError('Lengths, diameters, and Darcy factors must be finite and positive.')
    reference=lab4_config()['pipes']
    if not p[['start','end']].equals(reference[['start','end']]):
        raise ValueError('Keep the assigned Lab 4 connections; this branch solver has a fixed topology.')
    if set(cfg['reservoir_heads'])!=set(BRANCHES) or set(cfg['bend_K'])!=set(BRANCHES):
        raise ValueError('Provide exactly R-1, R-2, and R-3 heads and bend coefficients.')
    if not np.isfinite([cfg['nu'],cfg['density']]).all() or cfg['nu']<=0 or cfg['density']<=0:
        raise ValueError('Viscosity and density must be finite and positive.')
    if cfg['atmosphere_Pa']<=0 or cfg['vapor_Pa']<0:raise ValueError('Absolute atmospheric pressure must be positive and vapor pressure nonnegative.')
    for val in [*cfg['reservoir_heads'].values(),cfg['junction2_elevation'],cfg['atmosphere_Pa'],cfg['vapor_Pa']]:
        if not np.isfinite(val):raise ValueError('Heads and pressures must be finite.')
    if 'geometry_elevations' in cfg:
        expected={'R-1','R-2','R-3','J-1','J-2','J-3','J-4'}
        if set(cfg['geometry_elevations'])!=expected or not np.isfinite(list(cfg['geometry_elevations'].values())).all():
            raise ValueError('Provide finite physical elevations for all seven Lab 4 nodes.')
        for res,h in cfg['reservoir_heads'].items():
            if h < cfg['geometry_elevations'][res]-1e-10:
                raise ValueError(f'{res}: water level is below the assumed pipe connection elevation. Revise geometry or use a different outlet boundary model.')
    for val in [*cfg['bend_K'].values(),cfg['entrance_K'],cfg['exit_K'],cfg['contraction_K'],cfg['tee_K']]:
        if not np.isfinite(val) or val<0:raise ValueError('Loss coefficients must be finite and nonnegative.')
    if cfg['friction_mode'] not in ['fixed','roughness']:raise ValueError('Choose fixed or roughness friction.')
    if cfg['friction_mode']=='roughness':
        for pid in p.index:
            e=cfg['roughness_m'].get(pid,-1)
            if not np.isfinite(e) or e<0 or e>=p.loc[pid,'D_m']:
                raise ValueError(f'{pid}: provide a nonnegative roughness smaller than its diameter.')

def elevations4(cfg):
    if 'geometry_elevations' in cfg:return dict(cfg['geometry_elevations'])
    p=cfg['pipes'];z2=cfg['junction2_elevation']
    z=dict(cfg['reservoir_heads']);z['J-2']=z2
    for res,mid,first in [('R-1','J-1','P-1'),('R-2','J-3','P-4'),('R-3','J-4','P-6')]:
        total=p.loc[BRANCHES[res],'L_m'].sum()
        z[mid]=z[res]+(z2-z[res])*p.loc[first,'L_m']/total
    return z

def branch4(cfg,res,q):
    """Return loss events in actual flow direction; q>0 means reservoir -> J-2.
    Common total head is imposed at the idealized central junction.
    """
    p=cfg['pipes'];outward=q<0
    ids=BRANCHES[res][::-1] if outward else BRANCHES[res]
    nodes=([res,{'R-1':'J-1','R-2':'J-3','R-3':'J-4'}[res],'J-2'])
    if outward:nodes=nodes[::-1]
    events=[]
    def local(label,k,pid,at):
        if cfg['include_minor'] and k:
            v=abs(q)/(np.pi*p.loc[pid,'D_m']**2/4)
            events.append(dict(kind='local',label=label,pipe=pid,loss_m=k*v*v/(2*G),K=k,at=at))
    if not outward:
        local('reservoir entrance',cfg['entrance_K'],ids[0],nodes[0])
        local('reservoir bend',cfg['bend_K'][res],ids[0],nodes[0])
    elif res=='R-1':local('J-2 tee, P-2 only',cfg['tee_K'],'P-2','J-2')
    for j,pid in enumerate(ids):
        row=p.loc[pid];D=row.D_m;A=np.pi*D**2/4;v=abs(q)/A
        f=row.f_D if cfg['friction_mode']=='fixed' else friction(q,D,cfg['roughness_m'][pid],cfg['nu'])[0]
        hf=0. if q==0 else f*row.L_m/D*v*v/(2*G)
        events.append(dict(kind='pipe',label=pid,pipe=pid,loss_m=hf,L_m=row.L_m,
                           start=nodes[j],end=nodes[j+1],V_m_s=v,f_D=f,Re=4*abs(q)/(np.pi*D*cfg['nu'])))
        if j==0:
            Dnext=p.loc[ids[1],'D_m'];vnext=abs(q)/(np.pi*Dnext**2/4)
            if Dnext>D:
                k=(1-(D/Dnext)**2)**2
                local('sudden expansion',k,pid,nodes[1])
                if cfg['include_minor'] and k:
                    events[-1].update(v_before=v,v_after=vnext)
            elif Dnext<D:
                local('sudden contraction',cfg['contraction_K'],ids[1],nodes[1])
                if cfg['include_minor'] and cfg['contraction_K']:
                    events[-1].update(v_before=v,v_after=vnext)
    if outward:
        local('reservoir bend',cfg['bend_K'][res],ids[-1],res)
        local('reservoir exit',cfg['exit_K'],ids[-1],res)
    elif res=='R-1':local('J-2 tee, P-2 only',cfg['tee_K'],'P-2','J-2')
    return events

def solve4(cfg,log=None):
    validate4(cfg)
    heads=cfg['reservoir_heads']
    def flow(res,h):
        dh=heads[res]-h
        if abs(dh)<1e-13:return 0.
        s=np.sign(dh)
        def loss(q):return sum(e['loss_m'] for e in branch4(cfg,res,s*q))
        hi=.01
        while loss(hi)<abs(dh):
            hi*=2
            if hi>1e5:raise RuntimeError('Cannot bracket branch flow; check the model.')
        return s*brentq(lambda q:loss(q)-abs(dh),0.,hi,xtol=1e-13)
    count=0
    def balance(h):
        nonlocal count
        count+=1;qs=[flow(r,h) for r in BRANCHES];res=sum(qs)
        if log:log.info('J-2 trial %02d: total head %.6f m; inflow minus outflow %+.6e m3/s',count,h,res)
        return res
    lo,hi=min(heads.values()),max(heads.values())
    h=lo if lo==hi else brentq(balance,lo,hi,xtol=1e-12)
    qs={r:flow(r,h) for r in BRANCHES}
    rows=[];loss_rows=[];events={}
    for res,q in qs.items():
        events[res]=branch4(cfg,res,q)
        for e in events[res]:
            loss_rows.append(dict(branch=res,**e))
            if e['kind']=='pipe':
                pid=e['pipe'];row=cfg['pipes'].loc[pid]
                sign=1 if row.start==e['start'] else -1
                rows.append(dict(pipe=pid,Q_m3_s=sign*abs(q),Q_L_s=1000*sign*abs(q),
                    direction=(f"{e['start']} -> {e['end']}" if abs(q)>1e-12 else "no flow"),V_m_s=e['V_m_s'],Re=e['Re'],
                    f_D=e['f_D'],friction_loss_m=e['loss_m']))
    table=pd.DataFrame(rows).set_index('pipe').loc[cfg['pipes'].index]
    energy=max(abs(abs(heads[r]-h)-sum(e['loss_m'] for e in events[r])) for r in BRANCHES)
    mass=abs(sum(qs.values()))
    if mass>1e-9 or energy>1e-7:raise RuntimeError('Mass/energy validation failed; do not use these results.')
    if log:
        log.info('CONVERGED: J-2 total head %.6f m; mass %.2e m3/s; energy %.2e m.',h,mass,energy)
        log.warning('Bend and some fitting coefficients are teaching assumptions; report them with your results.')
    return dict(config=copy.deepcopy(cfg),head_J2=h,branch_Q=qs,pipes=table,
                losses=pd.DataFrame(loss_rows),events=events,mass_residual=mass,energy_residual=energy,trials=count)

def profile4(result):
    """Pipe-side pressure profile, including both sides of local loss jumps.
    Straight centerline interpolation; no claim about pressure inside fittings.
    """
    cfg=result['config'];z=elevations4(cfg);rows=[]
    for res,q in result['branch_Q'].items():
        ev=result['events'][res];start='J-2' if q<0 else res
        E=result['head_J2'] if q<0 else cfg['reservoir_heads'][res];x=0.;node=start
        def add(label,v,zz,pid='',location='pipe side'):
            hgl=E-v*v/(2*G);pabs=cfg['atmosphere_Pa']+cfg['density']*G*(hgl-zz)
            rows.append(dict(branch=res,x_m=x,location=location,label=label,pipe=pid,EGL_m=E,
                HGL_m=hgl,z_m=zz,p_abs_kPa=pabs/1000,vapor_margin_kPa=(pabs-cfg['vapor_Pa'])/1000))
        if start==res:add('reservoir free surface',0,cfg['reservoir_heads'][res],location='free surface')
        for e in ev:
            pid=e['pipe'];v=abs(q)/(np.pi*cfg['pipes'].loc[pid,'D_m']**2/4)
            if e['kind']=='local':
                before=e.get('v_before',v);after=e.get('v_after',v)
                if e['label']=='reservoir entrance':before=0.
                if e['label']=='reservoir exit':after=0.
                add(e['label']+' before',before,z[e['at']],pid)
                E-=e['loss_m'];add(e['label']+' after',after,z[e['at']],pid)
            else:
                add(pid+' inlet',v,z[e['start']],pid)
                E0=E;x0=x
                for fraction in np.linspace(0,1,25)[1:]:
                    E=E0-fraction*e['loss_m'];x=x0+fraction*e['L_m']
                    zz=z[e['start']]+fraction*(z[e['end']]-z[e['start']])
                    add(pid,v,zz,pid)
                node=e['end']
        if q<0:add('reservoir free surface',0,cfg['reservoir_heads'][res],location='free surface')
    return pd.DataFrame(rows)

def plot_profiles4(result):
    data=profile4(result)
    fig,axes=plt.subplots(3,1,figsize=(10,10.5),layout='constrained')
    for ax,(res,df) in zip(axes,data.groupby('branch',sort=False)):
        ax.plot(df.x_m,df.EGL_m,color=NAVY,lw=2.4,label='EGL: total head')
        ax.plot(df.x_m,df.HGL_m,color=BLUE,lw=1.9,ls='--',label='HGL: static head')
        ax.plot(df.x_m,df.z_m,color=GOLD,lw=1.5,ls=':',label='assumed pipe elevation')
        events=[e for e in result['events'][res] if e['kind']=='pipe']
        distance=0.;ticks=[0.]
        for event in events:
            ax.text((distance+event['L_m']/2)/df.x_m.max(),.06,event['pipe'],
                    transform=ax.transAxes,ha='center',fontsize=10,color=MUTED_COLOR,
                    bbox=dict(fc='white',ec='none',alpha=.85))
            distance+=event['L_m'];ticks.append(distance)
        ax.axvline(ticks[1],color='#a8b3c6',lw=.8,ls=':')
        q=result['branch_Q'][res]
        direction=(f'{res} → J-2' if q>1e-12 else f'J-2 → {res}' if q< -1e-12 else f'{res} — J-2: no net flow')
        ax.set(title=f'{direction}  |  {abs(q)*1000:.2f} L/s',xlabel='Distance in actual flow direction (m)',ylabel='Head / elevation (m)')
        ax.set_xticks(ticks);ax.set_xticklabels([f'{x:,.1f}' for x in ticks]);ax.grid(alpha=.16)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside upper center',ncol=3,fontsize=10)
    return fig

MUTED_COLOR='#5d6e88'

def plot_pressure4(result):
    """Absolute pressure and vapor threshold, with explicit minimum locations."""
    data=profile4(result);pv=result['config']['vapor_Pa']/1000
    fig,axes=plt.subplots(1,3,figsize=(12,4.2),layout='constrained',sharey=True)
    upper=max(data.p_abs_kPa.max()*1.12,pv*1.2,1.)
    lower=min(0.,data.p_abs_kPa.min()*1.1)
    for ax,(res,df) in zip(axes,data.groupby('branch',sort=False)):
        ax.plot(df.x_m,df.p_abs_kPa,color=BLUE,lw=2.2)
        ax.axhline(pv,color=RED,ls='--',lw=1.6,label='vapor pressure')
        ax.axhspan(lower,pv,color='#f8e7ea',alpha=.7)
        point=df.loc[df.p_abs_kPa.idxmin()]
        ax.scatter([point.x_m],[point.p_abs_kPa],color=NAVY,s=35,zorder=5)
        ax.set_title(f"{res} | minimum {point.p_abs_kPa:.2f} kPa\nVapor margin {point.vapor_margin_kPa:.2f} kPa",fontsize=11)
        ax.set(xlabel='Distance along flow (m)',ylim=(lower,upper));ax.grid(alpha=.15)
    axes[0].set_ylabel('Absolute pressure (kPa)')
    fig.suptitle('Pipe-section pressure screen • fitting interiors are not resolved',fontsize=13)
    axes[-1].legend(loc='lower right',fontsize=9)
    return fig


def network_plot4(result=None,cfg=None):
    cfg=cfg or result['config'];xy={'R-1':(0,2.8),'J-1':(1.5,2),'J-2':(3,1.2),
        'J-3':(4.5,1.2),'R-2':(6,1.2),'J-4':(2.3,.2),'R-3':(1.6,-.8)}
    fig,ax=plt.subplots(figsize=(11,6),layout='constrained')
    for pid,row in cfg['pipes'].iterrows():
        a=np.array(xy[row.start]);b=np.array(xy[row.end]);q=None if result is None else result['pipes'].loc[pid,'Q_m3_s']
        ax.plot(*np.array([a,b]).T,color=BLUE,lw=3)
        if q is not None and abs(q)>1e-10:
            u,v=(a,b) if q>0 else (b,a)
            ax.annotate('',xy=.35*u+.65*v,xytext=.65*u+.35*v,arrowprops=dict(arrowstyle='-|>',lw=2,color=NAVY))
        mid=(a+b)/2;label=f'{pid}  |  {row.D_m*1000:g} mm'
        if q is not None:label+=f'\n{abs(q)*1000:.1f} L/s'
        ax.text(mid[0]+.04,mid[1]+.18,label,ha='center',fontsize=10,bbox=dict(fc='white',ec='none',alpha=.9))
    for name,(x,y) in xy.items():
        res=name.startswith('R');ax.scatter(x,y,s=230 if res else 85,marker='s' if res else 'o',color=NAVY,zorder=5)
        label=name+(f"\nwater level {cfg['reservoir_heads'][name]:g} m" if res else '')
        if name=='J-2':ax.text(x+.15,y+.30,label,ha='left',va='bottom',fontsize=10)
        elif name=='J-4':ax.text(x+.25,y,label,ha='left',va='center',fontsize=10)
        else:ax.text(x,y-.23,label,ha='center',va='top',fontsize=10)
    ax.set_title('Lab 4 • network connections and actual flow directions' if result else 'Lab 4 • model layout (not to scale)')
    ax.set(xlim=(-.6,6.7),ylim=(-1.5,3.3));ax.axis('off');return fig

def roughness4(result):
    """Invert Colebrook at the fixed-f operating point, not at arbitrary Re."""
    rows=[]
    for pid,p in result['config']['pipes'].iterrows():
        re=result['pipes'].loc[pid,'Re'];f=p.f_D
        if re<4000:raise ValueError(f'{pid}: turbulent roughness calibration needs Re >= 4000.')
        eps=3.7*p.D_m*(10**(-1/(2*np.sqrt(f)))-2.51/(re*np.sqrt(f)))
        eps_sj=3.7*p.D_m*(10**(-1/(2*np.sqrt(f)))-5.74/re**.9)
        rows.append([pid,re,f,eps,eps*1000,eps_sj*1000,eps>=0])
    return pd.DataFrame(rows,columns=['pipe','Re','target_f','eps_m','eps_mm','Swamee_Jain_eps_mm','physical_eps']).set_index('pipe')

# Lab 5: names are pipe identifiers; reference directions are explicit below.
NODES5=['A','B','C','D','E','F']
CYCLES=np.array([[0,1,0,-1,-1,-1,0,0,0],
                 [0,0,0,0,1,0,-1,-1,-1],
                 [-1,1,1,0,0,0,0,0,0]],dtype=float)
LOOP_NAMES=['upper loop A-B-D-C-A','lower loop C-D-F-E-C','reservoir path R1-A-B-R2']

def lab5_config(demand_mode='table'):
    demand=np.array([7.81,7.81,16.27,16.27,14.97,14.97])/1000
    if demand_mode=='population':demand*=((15000*300/1000/86400)/demand.sum())
    elif demand_mode!='table':raise ValueError('Demand mode must be table or population.')
    p=pd.DataFrame({'pipe':['AR1','AB','BR2','AC','CD','DB','CE','EF','FD'],
        'start':['A','A','B','A','C','D','C','E','F'],
        'end':['R1','B','R2','C','D','B','E','F','D'],
        'L_m':[2000,1200,1500,300,1200,300,300,1200,300],
        'D_m':[.25]*9,'eps_m':[.00015]*9}).set_index('pipe')
    return dict(pipes=p,heads={'R1':160.,'R2':180.},demand=dict(zip(NODES5,demand)),
        elevations=dict(zip(NODES5,[100,100,94,94,88,88])),nu=NU,demand_mode=demand_mode,
        diameter_catalog_m=[.10,.125,.15,.20,.25,.30,.35,.40,.45,.50,.60])

def incidence5(cfg):
    B=np.zeros((6,9))
    for j,(_,p) in enumerate(cfg['pipes'].iterrows()):
        if p.start in NODES5:B[NODES5.index(p.start),j]=1
        if p.end in NODES5:B[NODES5.index(p.end),j]=-1
    return B

def initial5(cfg):
    """Minimum-Euclidean-norm seed satisfying all six continuity equations."""
    B=incidence5(cfg);d=np.array([cfg['demand'][n] for n in NODES5])
    return np.linalg.lstsq(B,-d,rcond=None)[0]

def size5(cfg,q0,target=1.):
    q0=np.asarray(q0,dtype=float)
    if not np.isfinite(target) or target<=0:raise ValueError('Target speed must be finite and positive.')
    if q0.shape!=(9,) or not np.isfinite(q0).all():raise ValueError('Provide nine finite sizing flows.')
    new=copy.deepcopy(cfg);catalog=np.asarray(cfg['diameter_catalog_m'],dtype=float)
    if catalog.ndim!=1 or len(catalog)==0 or not np.isfinite(catalog).all() or np.any(catalog<=0) or np.any(np.diff(catalog)<=0):
        raise ValueError('Diameter catalog must be finite, positive, and strictly increasing.')
    req=np.sqrt(4*np.abs(q0)/(np.pi*target))
    if max(req)>max(catalog):raise ValueError('Extend the diameter catalog for the requested flow.')
    chosen=np.array([catalog[catalog>=x][0] for x in req]);new['pipes']['D_m']=chosen
    return new,pd.DataFrame({'Q_seed_L_s':q0*1000,'D_at_target_mm':req*1000,
        'D_selected_mm':chosen*1000,'seed_speed_m_s':np.abs(q0)/(np.pi*chosen**2/4)},index=cfg['pipes'].index)

def validate5(cfg,q):
    p=cfg['pipes'];v=p[['L_m','D_m','eps_m']].to_numpy()
    if list(p.index)!=['AR1','AB','BR2','AC','CD','DB','CE','EF','FD']:raise ValueError('Keep the nine original pipe labels and order.')
    if not np.isfinite(v).all() or (p[['L_m','D_m']]<=0).any().any() or (p.eps_m<0).any() or (p.eps_m>=p.D_m).any():raise ValueError('Check positive lengths/diameters and valid roughness.')
    if not p[['start','end']].equals(lab5_config()['pipes'][['start','end']]):
        raise ValueError('Keep the assigned Lab 5 connections and pipe reference directions.')
    if set(cfg['heads'])!={'R1','R2'} or set(cfg['demand'])!=set(NODES5) or set(cfg['elevations'])!=set(NODES5):
        raise ValueError('Provide exactly R1/R2 heads and all six A-F demands/elevations.')
    if not np.isfinite(list(cfg['elevations'].values())).all():raise ValueError('Junction elevations must be finite.')
    if cfg['nu']<=0 or not np.isfinite(cfg['nu']):raise ValueError('Viscosity must be finite and positive.')
    if np.shape(q)!=(9,) or not np.isfinite(q).all():raise ValueError('Supply nine finite initial flows in m3/s.')
    d=np.array([cfg['demand'][n] for n in NODES5])
    if not np.isfinite(d).all() or (d<0).any():raise ValueError('Demands must be finite and nonnegative.')
    if not np.isfinite(list(cfg['heads'].values())).all():raise ValueError('Reservoir heads must be finite.')
    B=incidence5(cfg)
    if np.max(abs(B@CYCLES.T))>1e-12:raise ValueError('Topology differs from the loop definitions.')
    err=np.max(abs(B@q+d))
    if err>1e-9:raise ValueError(f'Initial flow violates continuity by {err:.3g} m3/s. Use initial5 or repair your seed.')

def losses5(cfg,q):
    return np.array([pipe_loss(qi,p.L_m,p.D_m,p.eps_m,nu=cfg['nu']) for qi,(_,p) in zip(q,cfg['pipes'].iterrows())])

def derivatives5(cfg,q):
    # Include df/dQ through a symmetric numerical derivative of the full law.
    dq=np.maximum(abs(q)*1e-5,1e-9)
    return (losses5(cfg,q+dq)-losses5(cfg,q-dq))/(2*dq)

def targets5(cfg):return np.array([0.,0.,cfg['heads']['R1']-cfg['heads']['R2']])

def first_correction5(cfg,q,loop=0):
    c=CYCLES[loop];h=losses5(cfg,q);dh=derivatives5(cfg,q)
    f=[friction(qi,p.D_m,p.eps_m,cfg['nu'])[0] for qi,(_,p) in zip(q,cfg['pipes'].iterrows())]
    table=pd.DataFrame({'Q_L_s':q*1000,'loop_sign':c,'f_D':f,'signed_h_m':h,
        'loop_contribution_m':c*h,'derivative_s_m2':c*c*dh},index=cfg['pipes'].index)
    residual=c@h-targets5(cfg)[loop];correction=-residual/np.sum(c*c*dh)
    return table.loc[c!=0],residual,correction

def hardy_cross5(cfg,q0=None,log=None,tol=1e-8,max_sweeps=1500):
    if not np.isfinite(tol) or tol<=0 or not isinstance(max_sweeps,(int,np.integer)) or max_sweeps<1:
        raise ValueError('Use a positive finite tolerance and a positive integer sweep limit.')
    q=initial5(cfg) if q0 is None else np.array(q0,dtype=float).copy();validate5(cfg,q)
    history=[];target=targets5(cfg);B=incidence5(cfg);d=np.array([cfg['demand'][n] for n in NODES5])
    for sweep in range(1,max_sweeps+1):
        max_dq=0.
        for k,c in enumerate(CYCLES):
            residual=c@losses5(cfg,q)-target[k];den=np.sum(c*c*derivatives5(cfg,q))
            if den<=0:raise RuntimeError('Nonpositive loss derivative; cannot compute a loop correction.')
            dq=-residual/den;alpha=1.
            while abs(c@losses5(cfg,q+alpha*dq*c)-target[k])>abs(residual)+1e-12 and alpha>2**-20:alpha*=.5
            q+=alpha*dq*c;max_dq=max(max_dq,abs(alpha*dq))
        residuals=CYCLES@losses5(cfg,q)-target;err=max(abs(residuals));mass=max(abs(B@q+d))
        history.append([sweep,*residuals,err,max_dq,mass])
        if log and (sweep<=5 or sweep%10==0 or err<tol):
            log.info('Sweep %03d | max energy residual %.3e m | largest correction %.3e L/s | mass %.2e m3/s',sweep,err,max_dq*1000,mass)
        if err<tol:break
    else:raise RuntimeError(f'Hardy-Cross did not converge in {max_sweeps} sweeps. Last residual {err:.4g} m.')
    if mass>1e-9:raise RuntimeError('Continuity was lost during the loop corrections.')
    return finish5(cfg,q,pd.DataFrame(history,columns=['sweep','upper_m','lower_m','reservoir_path_m','max_energy_m','max_dQ_m3_s','mass_m3_s']),log)

def heads5(cfg,q):
    # Overdetermined reconstruction checks agreement of all routes.
    B=incidence5(cfg);h=losses5(cfg,q);rhs=h.copy()
    for j,(_,p) in enumerate(cfg['pipes'].iterrows()):
        if p.start in cfg['heads']:rhs[j]-=cfg['heads'][p.start]
        if p.end in cfg['heads']:rhs[j]+=cfg['heads'][p.end]
    H=np.linalg.lstsq(B.T,rhs,rcond=None)[0]
    return dict(zip(NODES5,H)),np.max(abs(B.T@H-rhs))

def finish5(cfg,q,history=None,log=None):
    H,energy=heads5(cfg,q);d=np.array([cfg['demand'][n] for n in NODES5]);mass=max(abs(incidence5(cfg)@q+d))
    rows=[]
    for qi,(pid,p) in zip(q,cfg['pipes'].iterrows()):
        f,re=friction(qi,p.D_m,p.eps_m,cfg['nu']);v=abs(qi)/(np.pi*p.D_m**2/4)
        direction=('no flow' if abs(qi)<=1e-12 else (f'{p.start} -> {p.end}' if qi>0 else f'{p.end} -> {p.start}'))
        rows.append([pid,qi,qi*1000,direction,p.D_m*1000,v,re,f,abs(pipe_loss(qi,p.L_m,p.D_m,p.eps_m,nu=cfg['nu']))])
    pipes=pd.DataFrame(rows,columns=['pipe','Q_m3_s','Q_L_s','direction','D_mm','V_m_s','Re','f_D','loss_m']).set_index('pipe')
    nodes=pd.DataFrame({'z_m':cfg['elevations'],'HGL_m':H,'demand_m3_s':cfg['demand']})
    nodes['pressure_head_m']=nodes.HGL_m-nodes.z_m;nodes['gauge_pressure_kPa']=nodes.pressure_head_m*RHO*G/1000
    supplies={'R1':-q[0],'R2':-q[2]}
    if log:
        log.info('CONVERGED | pipe energy %.2e m; node continuity %.2e m3/s; total demand %.4f L/s',energy,mass,d.sum()*1000)
        for name,supply in supplies.items():log.info('%s: %s %.4f L/s',name,'supplies' if supply>=0 else 'receives',abs(supply)*1000)
        if nodes.pressure_head_m.min()<0:log.warning('Negative gauge pressure: prescribed-demand solution may be physically infeasible.')
        if ((pipes.Re>2000)&(pipes.Re<4000)).any():log.warning('A pipe is transitional; friction-model uncertainty is important.')
    if energy>1e-6 or mass>1e-9:raise RuntimeError('Final solution failed conservation checks.')
    return dict(config=copy.deepcopy(cfg),q=q,heads=H,pipes=pipes,nodes=nodes,supplies=supplies,
                history=history,mass_residual=mass,energy_residual=energy)

def independent5(cfg,qstart=None):
    """Solve all pipe-energy and node-continuity equations simultaneously.
    Uses no loop corrections; same constitutive law for a numerical cross-check.
    """
    B=incidence5(cfg);d=np.array([cfg['demand'][n] for n in NODES5]);q0=initial5(cfg) if qstart is None else np.asarray(qstart,dtype=float)
    validate5(cfg,q0)
    fixed=np.zeros(9)
    for j,(_,p) in enumerate(cfg['pipes'].iterrows()):
        fixed[j]=cfg['heads'].get(p.start,0)-cfg['heads'].get(p.end,0)
    def residual(x):
        q=x[:9]*.1;H=x[9:]*100
        return np.r_[(B@q+d)/.1,(B.T@H+fixed-losses5(cfg,q))/100]
    sol=least_squares(residual,np.r_[q0/.1,np.ones(6)*1.4],xtol=1e-13,ftol=1e-13,gtol=1e-13,max_nfev=1000)
    if not sol.success or max(abs(residual(sol.x)))>1e-9:raise RuntimeError('Independent solve did not close the equations.')
    return finish5(cfg,sol.x[:9]*.1)

def network_plot5(result=None,cfg=None,loops=False):
    cfg=cfg or result['config'];xy={'R1':(-1,2.7),'R2':(3,2.7),'A':(0,2),'B':(2,2),'C':(0,1),'D':(2,1),'E':(0,0),'F':(2,0)}
    fig,ax=plt.subplots(figsize=(11,7),layout='constrained')
    for j,(pid,p) in enumerate(cfg['pipes'].iterrows()):
        a=np.array(xy[p.start]);b=np.array(xy[p.end]);q=None if result is None else result['q'][j]
        ax.plot(*np.array([a,b]).T,color=BLUE,lw=3)
        u,v=(a,b) if q is None or q>=0 else (b,a)
        if q is None or abs(q)>1e-12:
            ax.annotate('',xy=.35*u+.65*v,xytext=.65*u+.35*v,arrowprops=dict(arrowstyle='-|>',color=NAVY,lw=1.7))
        mid=(a+b)/2;txt=pid+(f'\n{abs(q)*1000:.2f} L/s' if q is not None else f'\n{p.L_m:g} m')
        dx=.24 if abs(a[0]-b[0])<.1 else 0
        ax.text(mid[0]+dx,mid[1]+.13,txt,ha='center',fontsize=10,bbox=dict(fc='white',ec='none',alpha=.95))
    for n,(x,y) in xy.items():
        reservoir=n.startswith('R');ax.scatter(x,y,s=180 if reservoir else 80,marker='s' if reservoir else 'o',color=NAVY,zorder=5)
        txt=n+(f"\nH = {cfg['heads'][n]:g} m" if reservoir else f"\nz = {cfg['elevations'][n]:g} m\nd = {cfg['demand'][n]*1000:.2f} L/s")
        ax.text(x+(-.43 if x<1 else .43),y,txt,ha='center',va='center',fontsize=10)
    if loops:
        for y,label in [(1.5,'Upper loop'),(.5,'Lower loop')]:
            ax.annotate('',xy=(1.35,y),xytext=(.65,y),arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=-.8',color=PURPLE,lw=2))
            ax.text(1,y-.13,label,color=PURPLE,ha='center',fontsize=10)
        ax.text(1,3.0,'Third constraint: R1 → A → B → R2,  ΔH = −20 m',ha='center',color=PURPLE)
    ax.set(xlim=(-1.9,3.9),ylim=(-.6,3.35));ax.axis('off')
    ax.set_title('Lab 5 • actual flows' if result else 'Lab 5 • reference directions and loop structure');return fig

def plot_history5(result):
    hist=result['history'];fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for name,color,style in zip(['upper_m','lower_m','reservoir_path_m'],[BLUE,GOLD,PURPLE],['-','--','-.']):
        axes[0].semilogy(hist.sweep,np.maximum(abs(hist[name]),1e-16),color=color,ls=style,lw=1.9,label=name.replace('_m','').replace('_',' '))
    axes[0].axhline(1e-8,color=RED,ls=':',label='stopping threshold')
    axes[0].set(xlabel='Sweep',ylabel='Absolute energy residual (m)',title='Energy closure');axes[0].legend(fontsize=9)
    axes[1].plot(hist.sweep,hist.max_dQ_m3_s*1000,color=BLUE)
    axes[1].set(xlabel='Sweep',ylabel='Largest correction (L/s)',title='Flow corrections become small')
    for ax in axes:ax.grid(alpha=.2)
    return fig

def profile5(result):
    cfg=result['config'];heads={**cfg['heads'],**result['heads']};z={**cfg['heads'],**cfg['elevations']};rows=[]
    for j,(pid,p) in enumerate(cfg['pipes'].iterrows()):
        start,end=(p.start,p.end) if result['q'][j]>=0 else (p.end,p.start)
        vh=result['pipes'].loc[pid,'V_m_s']**2/(2*G)
        for t in np.linspace(0,1,21):
            h=heads[start]+t*(heads[end]-heads[start]);zz=z[start]+t*(z[end]-z[start])
            rows.append(dict(pipe=pid,x_m=t*p.L_m,start=start,end=end,HGL_m=h,EGL_m=h+vh,z_m=zz))
    return pd.DataFrame(rows)

def plot_profiles5(result):
    data=profile5(result);fig,axes=plt.subplots(3,3,figsize=(14,10),layout='constrained')
    for ax,(pid,df) in zip(axes.flat,data.groupby('pipe',sort=False)):
        ax.plot(df.x_m,df.EGL_m,color=NAVY,lw=2,label='EGL (pipe side)')
        ax.plot(df.x_m,df.HGL_m,color=BLUE,ls='--',lw=1.8,label='HGL')
        ax.plot(df.x_m,df.z_m,color=GOLD,ls=':',label='assumed elevation')
        vh=float(df.EGL_m.iloc[0]-df.HGL_m.iloc[0])
        ax.text(.97,.08,rf'$V^2/(2g)={vh:.3f}$ m',transform=ax.transAxes,ha='right',fontsize=9,
                bbox=dict(fc='white',ec='none',alpha=.9))
        ax.set_title(f'{pid}: {df.start.iloc[0]} → {df.end.iloc[0]}',fontsize=11)
        ax.set(xlabel='Distance along flow (m)',ylabel='Head (m)');ax.grid(alpha=.2)
    handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside upper center',ncol=3)
    return fig

def load_saved_inputs(path):
    """Restore an exported JSON configuration and validate its assigned network."""
    cfg=json.loads(Path(path).read_text(encoding='utf8'))
    cfg['pipes']=pd.DataFrame(cfg['pipes']).set_index('pipe')
    if 'reservoir_heads' in cfg:validate4(cfg)
    else:validate5(cfg,initial5(cfg))
    return cfg

def export_result(result,name,figures=None,folder='outputs'):
    """Export only files from this run; never include stale figures from an earlier run."""
    folder=Path(folder);folder.mkdir(exist_ok=True,parents=True);target=folder/name;target.mkdir(exist_ok=True)
    written=[]
    def record(filename):
        path=target/filename;written.append(path);return path
    cfg=copy.deepcopy(result['config']);cfg['pipes']=cfg['pipes'].reset_index().to_dict(orient='records')
    record('inputs.json').write_text(json.dumps(cfg,indent=2),encoding='utf8')
    for key in ['pipes','nodes','history','losses']:
        value=result.get(key)
        if isinstance(value,pd.DataFrame):value.to_csv(record(f'{key}.csv'),index=True)
    profile=(profile4(result) if 'head_J2' in result else profile5(result));profile.to_csv(record('profiles.csv'),index=False)
    record('checks.json').write_text(json.dumps({'created_UTC':datetime.now(timezone.utc).isoformat(),
        'mass_residual_m3_s':result['mass_residual'],'energy_residual_m':result['energy_residual']},indent=2),encoding='utf8')
    for label,fig in (figures or {}).items():
        fig.savefig(record(f'{label}.png'),dpi=170);fig.savefig(record(f'{label}.svg'))
    # Logs may contain earlier scenarios in the same session, so label that scope explicitly.
    logname='lab4.log' if 'head_J2' in result else 'lab5.log'
    logfile=folder/logname
    record('manifest.json').write_text(json.dumps({
        'result_files':[f.name for f in written],
        'session_log':logname if logfile.exists() else None,
        'log_scope':'Whole lab session; inputs.json identifies the exact exported case.',
        'reproduce':'load_saved_inputs(inputs.json), then solve4(config) or hardy_cross5(config).'
    },indent=2),encoding='utf8')
    archive=folder/f'{name}.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for f in written:z.write(f,arcname=f'{name}/{f.name}')
        if logfile.exists():z.write(logfile,arcname=f'{name}/{logname}')
    return archive


def plot_pipe_detail5(result,pid='AB'):
    """Three complementary views avoid hiding a small velocity head on a large elevation axis."""
    if pid not in result['pipes'].index:raise ValueError(f'Unknown pipe {pid}.')
    df=profile5(result);df=df[df['pipe']==pid];row=result['pipes'].loc[pid]
    vh=row.V_m_s**2/(2*G);q=abs(row.Q_L_s)
    fig,axes=plt.subplots(1,3,figsize=(12,4.3),layout='constrained')
    axes[0].plot(df.x_m,df.EGL_m,color=NAVY,lw=2.5,label='EGL')
    axes[0].plot(df.x_m,df.HGL_m,color=BLUE,ls='--',lw=1.8,label='HGL')
    axes[0].set(title='Pipe-side grade lines',xlabel='Distance along flow (m)',ylabel='Head (m)')
    span=max(float(df.EGL_m.max()-df.HGL_m.min()),.02)
    axes[0].set_ylim(df.HGL_m.min()-.12*span,df.EGL_m.max()+.12*span);axes[0].legend(fontsize=10)
    axes[1].bar(['EGL − HGL'],[vh],color=PURPLE,width=.45)
    axes[1].set(title='The velocity-head gap',ylabel='Velocity head (m)',ylim=(0,max(1.4*vh,.01)))
    axes[1].text(0,vh,rf'$V^2/(2g)={vh:.4f}$ m',ha='center',va='bottom',fontsize=11)
    axes[1].text(.5,.12,f'V = {row.V_m_s:.3f} m/s',transform=axes[1].transAxes,ha='center')
    axes[2].plot(df.x_m,df.HGL_m-df.z_m,color=GOLD,lw=2.2)
    axes[2].axhline(0,color='#a8b3c6',ls=':',lw=.8)
    axes[2].set(title='Pressure relative to local elevation',xlabel='Distance along flow (m)',ylabel='Gage pressure head (m)')
    for ax in axes:ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
    direction=row.direction.replace(' -> ',' → ')
    fig.suptitle(f'{pid}: {direction}  |  D = {row.D_mm:g} mm  |  |Q| = {q:.3f} L/s',fontsize=13)
    return fig
