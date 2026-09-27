"""15-state right-error ESKF, ENU world / ROS body FLU.

Nominal p,v,R,ba,bg; delta state [p,v,theta,ba,bg]. Gravity=[0,0,-g].
White-noise densities squared times dt. Joseph position update and attitude reset.
Only synthetic validation is authorized as ready: real initialization/calibration
must pass readiness; never substitute GT or GNSS course for body yaw.
"""
import numpy as np
from scipy.spatial.transform import Rotation

G=9.80665

def mm(a,b):
    # Explicit contraction avoids spurious Accelerate BLAS floating flags on macOS.
    return np.einsum('ij,jk->ik',a,b)

def mv(a,b):return np.einsum('ij,j->i',a,b)

def skew(v):
    x,y,z=v;return np.array([[0,-z,y],[z,0,-x],[-y,x,0]],float)

class ESKF:
    def __init__(self,p=None,v=None,R=None,ba=None,bg=None,lever=None,noise=(.1,.01,.001,.0001)):
        self.p=np.zeros(3) if p is None else np.array(p,float)
        self.v=np.zeros(3) if v is None else np.array(v,float)
        self.R=np.eye(3) if R is None else np.array(R,float)
        self.ba=np.zeros(3) if ba is None else np.array(ba,float)
        self.bg=np.zeros(3) if bg is None else np.array(bg,float)
        self.lever=np.zeros(3) if lever is None else np.array(lever,float)
        self.P=np.diag([10.]*3+[3.]*3+[.1,.1,np.pi**2]+[.1]*3+[.01]*3)
        self.noise=noise
    def predict(self,force,omega,dt):
        if not (0<dt<=.1):raise ValueError('Unsupported integration interval')
        f=np.array(force)-self.ba;w=np.array(omega)-self.bg
        old=self.R.copy();mid=mm(old,Rotation.from_rotvec(w*dt/2).as_matrix())
        accel=mv(mid,f)+[0,0,-G]
        self.p+=self.v*dt+.5*accel*dt*dt;self.v+=accel*dt
        self.R=mm(old,Rotation.from_rotvec(w*dt).as_matrix())
        F=np.zeros((15,15));F[:3,3:6]=np.eye(3);F[3:6,6:9]=-mm(mid,skew(f));F[3:6,9:12]=-mid
        F[6:9,6:9]=-skew(w);F[6:9,12:15]=-np.eye(3)
        phi=np.eye(15)+F*dt+.5*mm(F,F)*dt*dt
        qa,qg,qba,qbg=np.square(self.noise);Q=np.zeros((15,15))
        Q[:3,:3]=np.eye(3)*qa*dt**3/3;Q[:3,3:6]=np.eye(3)*qa*dt**2/2;Q[3:6,:3]=Q[:3,3:6].T
        Q[3:6,3:6]=np.eye(3)*qa*dt;Q[6:9,6:9]=np.eye(3)*qg*dt
        Q[9:12,9:12]=np.eye(3)*qba*dt;Q[12:15,12:15]=np.eye(3)*qbg*dt
        self.P=mm(mm(phi,self.P),phi.T)+Q;self.P=(self.P+self.P.T)/2
    def update_gnss(self,position,covariance):
        H=np.zeros((3,15));H[:,:3]=np.eye(3);H[:,6:9]=-mm(self.R,skew(self.lever))
        N=np.diag(covariance);S=mm(mm(H,self.P),H.T)+N
        K=np.linalg.solve(S,mm(H,self.P)).T
        error=np.asarray(position)-(self.p+mv(self.R,self.lever));delta=mv(K,error)
        A=np.eye(15)-mm(K,H);P=mm(mm(A,self.P),A.T)+mm(mm(K,N),K.T)
        self.p+=delta[:3];self.v+=delta[3:6];self.R=mm(self.R,Rotation.from_rotvec(delta[6:9]).as_matrix())
        self.ba+=delta[9:12];self.bg+=delta[12:15]
        reset=np.eye(15);reset[6:9,6:9]-=.5*skew(delta[6:9]);self.P=mm(mm(reset,P),reset.T)
        self.P=(self.P+self.P.T)/2

def initialize_from_allowed_history(imu,gnss_t,gnss_p,heading_rad=None,lever=None):
    if heading_rad is None or lever is None:
        raise RuntimeError('ESKF not ready: independent permitted heading/alignment and GNSS lever arm are required; no GT/course fallback')
    imu=np.asarray(imu);f=imu[:,:3].mean(0)
    if abs(np.linalg.norm(f)-G)>.3 or np.max(imu[:,:3].std(0))>.2:
        raise RuntimeError('Initial quasi-static gravity check failed')
    tilt,_=Rotation.align_vectors(np.array([[0,0,G]]),f[None])
    R=mm(Rotation.from_euler('z',heading_rad).as_matrix(),tilt.as_matrix())
    t=np.asarray(gnss_t);p=np.asarray(gnss_p);keep=t>=t[-1]-5;t=t[keep];p=p[keep]
    tc=t-t.mean();v=(tc[:,None]*(p-p.mean(0))).sum(0)/np.sum(tc*tc)
    return ESKF(p=p[-1]-mv(R,np.asarray(lever)),v=v,R=R,lever=lever)

def readiness(cfg):
    return {'status':'not_ready','synthetic_mechanization':'tested separately',
        'blockers':cfg['eskf']['blockers'],'real_metrics_allowed':False,
        'no_gt_initialization':True,'no_course_as_yaw':True,'parameter_tuning':'none'}
