"""One rest-to-rest quintic spline optimizer with exact signed hull costs.

No RRT, waypoint repair, terminal servo or external smoothing. The phase curve
and its duration constitute one trajectory. Duration is the analytic feasible
V/A/J scaling of that same curve, not a geometry-changing postprocessor.
Experimental sampled collision constraints are NOT a continuous certificate.
"""
from dataclasses import dataclass
import time
import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import minimize


@dataclass(frozen=True)
class SplineConfig:
    controls:int=12
    collocation:int=41
    clearance:float=.002
    collision_weight:float=1e6
    acceleration_weight:float=.1
    iterations:int=100
    refinement_rounds:int=3
    speed_metric:bool=True
    solver:str='lbfgs'
    velocity:float=1.5
    acceleration:float=4.
    jerk:float=30.
    dt:float=1/30


class ExactSpline:
    def __init__(self,sensor,config=SplineConfig()):
        self.sensor=sensor;self.config=config
        n=config.controls;k=5
        if n<8 or config.collocation<3:raise ValueError('Insufficient spline controls/samples')
        self.knots=np.r_[np.zeros(k+1),np.arange(1,n-k)/(n-k),np.ones(k+1)]
        self.basis=BSpline(self.knots,np.eye(n),k)
        self.grid=np.linspace(0,1,config.collocation)
        self.B=self.basis(self.grid)
        # Exact Gaussian quadrature for squared derivatives of degree-five spans.
        nodes,weights=np.polynomial.legendre.leggauss(5)
        t=np.concatenate([(i+(nodes+1)/2)/(n-k) for i in range(n-k)])
        w=np.tile(weights/(2*(n-k)),n-k)
        D3=self.basis.derivative(3)(t);D2=self.basis.derivative(2)(t)
        self.Q=D3.T@(w[:,None]*D3)/720 +config.acceleration_weight*D2.T@(w[:,None]*D2)/(120/7)
        self.free=np.arange(3,n-3)
        self.fixed=np.r_[np.arange(3),np.arange(n-3,n)]

    def initialize(self,start,goal,proposal=None):
        n=self.config.controls
        C=np.empty((n,8));C[:3]=start;C[-3:]=goal
        if proposal is None:
            C[self.free]=-np.linalg.solve(self.Q[np.ix_(self.free,self.free)],self.Q[np.ix_(self.free,self.fixed)]@C[self.fixed])
        else:
            proposal=np.asarray(proposal)
            basis=self.basis(np.linspace(0,1,len(proposal)))
            C[self.free]=np.linalg.lstsq(basis[:,self.free],proposal-basis[:,self.fixed]@C[self.fixed],rcond=None)[0]
        return C

    def solve(self,start,goal,proposal=None):
        begun=time.perf_counter();C=self.initialize(start,goal,proposal);evaluations=0;last={}
        grid=self.grid.copy();basis=self.B.copy();derivative=self.basis.derivative(1)(grid)
        reference_length=max(float(np.linalg.norm(np.asarray(goal)-start)),1e-3)
        def objective(x):
            nonlocal evaluations,last
            C[self.free]=x.reshape(-1,8)
            value=float(np.sum(C*(self.Q@C)));gradient=2*self.Q@C
            penalty=0.;minimum=float('inf');count=0
            weights=np.r_[np.diff(grid)[0]/2,(grid[2:]-grid[:-2])/2,np.diff(grid)[-1]/2]
            for t,B,D,q,dq,w in zip(grid,basis,derivative,basis@C,derivative@C,weights):
                d,g,_,_=self.sensor.query(q)
                if len(d):minimum=min(minimum,float(d.min()))
                # Endpoint clearance cannot be changed by the optimizer. Its
                # actual collision validity is a prerequisite, never relaxed.
                if t==0 or t==1:continue
                violation=np.minimum(d-self.config.clearance,0.)
                active=violation<0
                if active.any():
                    if self.config.speed_metric:
                        value_c,gq,gv=arc_collision_cost(d,g,dq,self.config.clearance)
                        penalty+=w*value_c/reference_length
                        gradient+=self.config.collision_weight*w/reference_length*(B[:,None]*gq+D[:,None]*gv)
                    else:
                        penalty+=w*float(violation@violation)
                        gradient+=B[:,None]*(2*self.config.collision_weight*w*(violation@g))[None]
                    count+=int(active.sum())
            value+=self.config.collision_weight*penalty
            evaluations+=1;last=dict(collocation_minimum_m=minimum,active_cost_pairs=count,cost=value)
            return value,gradient[self.free].ravel()
        exchanges=[]
        for round_index in range(self.config.refinement_rounds+1):
            if self.config.solver=='slsqp':
                pairs={tuple(p[:2]):i for i,p in enumerate(self.sensor.model.pairs)}
                keep=np.flatnonzero((grid>0)&(grid<1));constraint_cache={}
                def quad(x):
                    C[self.free]=x.reshape(-1,8)
                    return float(np.sum(C*(self.Q@C))),(2*self.Q@C)[self.free].ravel()
                def constraints(x):
                    if 'x' in constraint_cache and np.array_equal(x,constraint_cache['x']):
                        return constraint_cache['d'],constraint_cache['g']
                    C[self.free]=x.reshape(-1,8)
                    d=np.full((len(keep),len(pairs)),self.sensor.activation_distance-self.config.clearance)
                    g=np.zeros((len(keep),len(pairs),len(self.free),8))
                    for k,phase_index in enumerate(keep):
                        self.sensor.query(basis[phase_index]@C)
                        for r in self.sensor.last_rows:
                            j=pairs[r['pair']];d[k,j]=r['distance']-self.config.clearance
                            g[k,j]=basis[phase_index,self.free,None]*r['gradient']
                    constraint_cache.update(x=x.copy(),d=d.ravel(),g=g.reshape(-1,len(self.free)*8))
                    return constraint_cache['d'],constraint_cache['g']
                result=minimize(quad,C[self.free].ravel().copy(),jac=True,method='SLSQP',
                                bounds=[(-2*np.pi,2*np.pi)]*(len(self.free)*8),
                                constraints=[dict(type='ineq',fun=lambda x:constraints(x)[0],jac=lambda x:constraints(x)[1])],
                                options=dict(maxiter=self.config.iterations,ftol=1e-8))
            else:
                result=minimize(objective,C[self.free].ravel().copy(),jac=True,method='L-BFGS-B',
                                bounds=[(-2*np.pi,2*np.pi)]*(len(self.free)*8),
                                options=dict(maxiter=self.config.iterations,ftol=1e-10,gtol=1e-6,maxls=25))
            C[self.free]=result.x.reshape(-1,8)
            objective(result.x)
            if last['collocation_minimum_m'] < -1e-7:
                exchanges.append(dict(round=round_index,collocation=len(grid),optimizer_iterations=int(result.nit),
                                      reason='existing_constraints_infeasible_no_sampling_refinement',violating_pairs=[]))
                break
            curve=BSpline(self.knots,C,5)
            count=max(241,int(np.ceil(np.max(abs(curve.derivative(1)(np.linspace(0,1,2049))))/.02)))
            worst={}
            for phase,q in zip(np.linspace(0,1,count+1),curve(np.linspace(0,1,count+1))):
                self.sensor.query(q)
                for row in getattr(self.sensor,'last_rows',[]):
                    if row['distance'] < -1e-7 and (row['pair'] not in worst or row['distance'] < worst[row['pair']][1]):
                        worst[row['pair']]=(float(phase),row['distance'])
            exchanges.append(dict(round=round_index,collocation=len(grid),optimizer_iterations=int(result.nit),
                                  violating_pairs=[dict(pair=p,phase=v[0],distance_m=v[1]) for p,v in worst.items()]))
            if not worst or round_index==self.config.refinement_rounds:break
            # Constraint exchange inside the SAME optimization problem: add
            # omitted violating phases, never alter a returned path afterward.
            additions=np.clip([v[0]+offset/count for v in worst.values() for offset in (-1,0,1)],0,1)
            grid=np.unique(np.r_[grid,additions]);basis=self.basis(grid);derivative=self.basis.derivative(1)(grid)
        spline=BSpline(self.knots,C,5)
        dense=np.linspace(0,1,2049)
        maxima=[float(np.max(abs(spline.derivative(o)(dense)))) for o in (1,2,3)]
        duration=max(maxima[0]/self.config.velocity,np.sqrt(maxima[1]/self.config.acceleration),np.cbrt(maxima[2]/self.config.jerk),self.config.dt*4)
        # Guard grid peak estimation with a small common numerical margin.
        duration=np.ceil(duration*1.001/self.config.dt)*self.config.dt
        steps=int(round(duration/self.config.dt));phase=np.linspace(0,1,steps+1)
        data=dict(q=spline(phase),velocity=spline.derivative(1)(phase)/duration,
                  acceleration=spline.derivative(2)(phase)/duration**2,jerk=spline.derivative(3)(phase)/duration**3,
                  dt=self.config.dt,controls=C.copy(),knots=self.knots.copy(),duration=duration)
        info=dict(optimizer_success=bool(result.success),message=str(result.message),iterations=int(result.nit),
                  evaluations=evaluations,wall_s=time.perf_counter()-begun,constraint_exchanges=exchanges,**last)
        return data,info


def arc_collision_cost(distances,gradients,phase_velocity,clearance):
    """Configuration-space line-integral density and first derivatives.

    Penalizing path length through obstacles prevents reducing collision cost
    merely by compressing the colliding motion into a short time interval.
    """
    violation=np.minimum(np.asarray(distances)-clearance,0.)
    penalty=float(violation@violation)
    speed=max(float(np.linalg.norm(phase_velocity)),1e-12)
    return penalty*speed,2*speed*(violation@gradients),penalty*np.asarray(phase_velocity)/speed
