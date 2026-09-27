"""Fixed physical diagnostic, no learned model, GT input, or parameter search."""
import numpy as np

def mm(a,b):return np.einsum('ij,jk->ik',a,b)
def mv(a,b):return np.einsum('ij,j->i',a,b)
def exp_rotation(w):
    angle=np.linalg.norm(w);x,y,z=w
    k=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
    if angle<1e-8:return np.eye(3)+k+.5*mm(k,k)
    return np.eye(3)+np.sin(angle)/angle*k+(1-np.cos(angle))/angle**2*mm(k,k)

def initial_rotation(mean_force,yaw):
    up=np.asarray(mean_force)/np.linalg.norm(mean_force)
    forward=np.array([1.,0,0]);forward-=up*np.dot(up,forward)
    if np.linalg.norm(forward)<1e-8:raise ValueError('forward axis vertical: yaw not observable')
    forward/=np.linalg.norm(forward);left=np.cross(up,forward)
    body=np.column_stack([forward,left,up])
    c,s=np.cos(yaw),np.sin(yaw);world=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
    return mm(world,body.T)

def causal_gnss_velocity(t,held,source_t,source_id):
    # Exact arithmetic/order of the fixed V2 build_inputs velocity calculation.
    velocity=np.zeros_like(held);history=[];v=np.zeros(3)
    for i in range(len(t)):
        if i==0 or source_id[i]!=source_id[i-1]:
            history.append(i);history=[k for k in history if source_t[k]>=t[i]-5.]
            if len(history)>=2 and source_t[history[-1]]-source_t[history[0]]>=1.:
                x=source_t[history]-source_t[history].mean();y=held[history]
                v=(x[:,None]*(y-y.mean(0))).sum(0)/(x@x)
            else:v=np.zeros(3)
        velocity[i]=v
    return velocity.astype(np.float32).astype(float)

def estimate_initial(t,imu,held,source_t,velocity,onset,cfg):
    a=int(np.searchsorted(t,onset));anchor=float(source_t[a-1]);p=held[a-1].copy();v=velocity[a-1].copy()
    pre=(t>anchor-cfg['gravity_history_s'])&(t<=anchor)
    if pre.sum()<10:raise ValueError('insufficient strictly pre-anchor IMU')
    fmean=imu[pre,:3].mean(0);fmedian=np.median(imu[pre,:3],axis=0)
    course_pre=(t>=onset-5)&(t<onset)&(np.linalg.norm(velocity[:,:2],axis=1)>=cfg['yaw_min_horizontal_speed_m_s'])
    courses=np.unwrap(np.arctan2(velocity[course_pre,1],velocity[course_pre,0]))
    speed=np.linalg.norm(v[:2]);std=float(np.degrees(courses.std())) if len(courses) else None
    reliable=bool(speed>=cfg['yaw_min_horizontal_speed_m_s'] and std is not None and std<=cfg['yaw_max_course_std_deg'])
    yaw=float(np.arctan2(v[1],v[0])) if reliable else np.radians(cfg['yaw_fallback_deg_east_ccw'])
    reason='course-aligned assumption; true body yaw unobservable' if reliable else 'low speed or unstable/absent course; fixed east yaw fallback'
    r=initial_rotation(fmean,yaw);g=cfg['gravity_m_s2'];reference=mv(r.T,np.array([0.,0,g]))
    return {'anchor_timestamp_s':anchor,'onset_s':float(onset),'p0':p.tolist(),'v0':v.tolist(),'yaw_east_ccw_deg':float(np.degrees(yaw)),
      'horizontal_speed_m_s':float(speed),'course_std_deg':std,'course_reliable_for_course_only':reliable,'yaw_policy':reason,'R0':r.tolist(),
      'pre_sample_count':int(pre.sum()),'pre_first_s':float(t[pre][0]),'pre_last_s':float(t[pre][-1]),
      'mean_accel_xyz':fmean.tolist(),'std_accel_xyz':imu[pre,:3].std(0).tolist(),'median_accel_xyz':fmedian.tolist(),
      'mean_gyro_xyz':imu[pre,3:].mean(0).tolist(),'std_gyro_xyz':imu[pre,3:].std(0).tolist(),
      'median_gyro_xyz':np.median(imu[pre,3:],axis=0).tolist(),'mean_accel_norm_m_s2':float(np.linalg.norm(fmean)),
      'radial_gravity_discrepancy_m_s2':float(np.linalg.norm(fmean)-g),
      'accel_bias_mean':(fmean-reference).tolist(),'accel_bias_median':(fmedian-reference).tolist(),
      'R0_body_down_tilt_deg':float(np.degrees(np.arccos(np.clip(-r[2,2],-1,1)))),
      'a_reference_world':(mv(r,fmean)+[0,0,-g]).tolist()}

def integrate(t,imu,gyro_source_t,query_t,initial,method,g=9.80665):
    """IMU ZOH at interval LEFT endpoint; split at accel timestamps, no GT/GNSS arguments.

    Accel already has causal gyro attached. Orientation midpoint uses the same
    past gyro sample, not a future measurement. Actual gaps are not shortened.
    """
    r=np.array(initial['R0']);r0=r.copy();p=np.array(initial['p0']);v=np.array(initial['v0']);now=initial['anchor_timestamp_s']
    ba=np.zeros(3);bg=np.zeros(3)
    if method=='dr_mean_bias':ba=np.array(initial['accel_bias_mean']);bg=np.array(initial['mean_gyro_xyz'])
    if method=='dr_median_bias':ba=np.array(initial['accel_bias_median']);bg=np.array(initial['median_gyro_xyz'])
    gravity=np.array([0.,0,-g]);fref=mv(r0.T,-gravity)
    sensor_dp=np.zeros(3);sensor_dv=np.zeros(3);tilt_dp=np.zeros(3);tilt_dv=np.zeros(3)
    pp=[];vv=[];rr=[];aa=[];ap=[];tp=[];source_rows=[];max_orth=0.;dtmax=0.
    for endpoint in query_t:
        if endpoint<now:raise ValueError('timestamps not chronological')
        a=np.zeros(3);j=-1
        while now<endpoint-1e-10:
            j=int(np.searchsorted(t,now+1e-10,side='right')-1)
            if j<0:raise ValueError('no past IMU')
            end=min(float(endpoint),float(t[j+1]) if j+1<len(t) else float(endpoint))
            dt=end-now
            if dt<=0:raise ValueError('non-positive interval')
            assert t[j]<=now+1e-9 and gyro_source_t[j]<=t[j]+1e-9
            w=imu[j,3:]-bg;rmid=mm(r,exp_rotation(w*dt*.5))
            a=mv(rmid,imu[j,:3]-ba)+gravity
            tilt=mv(rmid,fref)+gravity;sensor=mv(rmid,imu[j,:3]-ba-fref)
            if method=='cv_acceleration_residual':a-=initial['a_reference_world']
            if method=='horizontal_dr_zero_bias':a[2]=0.
            p+=v*dt+.5*a*dt*dt;v+=a*dt
            sensor_dp+=sensor_dv*dt+.5*sensor*dt*dt;sensor_dv+=sensor*dt
            tilt_dp+=tilt_dv*dt+.5*tilt*dt*dt;tilt_dv+=tilt*dt
            r=mm(r,exp_rotation(w*dt));now=end;dtmax=max(dtmax,dt)
            max_orth=max(max_orth,float(np.max(np.abs(mm(r.T,r)-np.eye(3)))))
        pp.append(p.copy());vv.append(v.copy());rr.append(r.copy());aa.append(a.copy());ap.append(sensor_dp.copy());tp.append(tilt_dp.copy());source_rows.append(j)
    return {'pred':np.array(pp),'velocity':np.array(vv),'orientation':np.array(rr),'acceleration_world':np.array(aa),
            'force_residual_displacement':np.array(ap),'orientation_gravity_displacement':np.array(tp),
            'last_used_imu_index':np.array(source_rows),'max_rotation_orthogonality_error':max_orth,'max_integration_step_s':dtmax,
            'accel_bias':ba,'gyro_bias':bg}

def metrics(pred,gt,target,anchor):
    raw=pred-gt;motion=pred-anchor-target
    def vals(e,prefix):return {prefix+'rmse_3d_m':float(np.sqrt(np.mean(np.sum(e*e,axis=1)))),
        prefix+'rmse_horizontal_m':float(np.sqrt(np.mean(np.sum(e[:,:2]**2,axis=1)))),
        prefix+'rmse_vertical_m':float(np.sqrt(np.mean(e[:,2]**2))),
        prefix+'final_3d_m':float(np.linalg.norm(e[-1])),prefix+'final_horizontal_m':float(np.linalg.norm(e[-1,:2])),
        prefix+'final_abs_vertical_m':float(abs(e[-1,2]))}
    return {**vals(motion,'motion_'),**vals(raw,'raw_')}
