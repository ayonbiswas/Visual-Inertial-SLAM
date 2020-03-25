import numpy as np
import matplotlib.pyplot as plt
from transforms3d.euler import mat2euler
from utils import *
import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.linalg import block_diag
 
# function to read visual features, IMU measurements and calibration parameters 
def load_data(file_name):
 
  with np.load(file_name) as data:
	  t = data["time_stamps"] # time_stamps
	  features = data["features"] # 4 x num_features : pixel coordinates of features
	  linear_velocity = data["linear_velocity"] # linear velocity measured in the body frame
	  rotational_velocity = data["rotational_velocity"] # rotational velocity measured in the body frame
	  K = data["K"] # intrindic calibration matrix
	  b = data["b"] # baseline
	  cam_T_imu = data["cam_T_imu"] # Transformation from imu to camera frame
  return t,features,linear_velocity,rotational_velocity,K,b,cam_T_imu

#function to visualize only the trajectory in 2D
def visualize_only_trajectory(pose,path_name="Unknown",show_ori=False):

  fig,ax = plt.subplots(figsize=(7,7))
  n_pose = pose.shape[2]
  ax.plot(pose[0,3,:],pose[1,3,:],'r-',linewidth=2,label=path_name)
  ax.scatter(pose[0,3,0],pose[1,3,0],marker='s',label="start")
  ax.scatter(pose[0,3,-1],pose[1,3,-1],marker='o',label="end")
  
  if show_ori:
	  select_ori_index = list(range(0,n_pose,max(int(n_pose/50), 1)))
	  yaw_list = []
	  for i in select_ori_index:
		  _,_,yaw = mat2euler(pose[:3,:3,i])
		  yaw_list.append(yaw)
	  dx = np.cos(yaw_list)
	  dy = np.sin(yaw_list)
	  dx,dy = [dx,dy]/np.sqrt(dx**2+dy**2)
	  ax.quiver(pose[0,3,select_ori_index],pose[1,3,select_ori_index],dx,dy,\
		  color="b",units="xy",width=1)
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.axis('equal')
  ax.grid(False)
  ax.legend()
  plt.show(block=True)

  return fig, ax
			 
#helper functions to calculate hat map and adjoint
def get_uhat(v,w):
	return np.concatenate([np.concatenate([get_hat(w),v.reshape(-1,1)],axis = 1),np.zeros((1,4))],axis = 0)
	
def get_uadjoint(v,w):
	return np.concatenate([np.concatenate([get_hat(w),get_hat(v)],axis = 1),np.concatenate([np.zeros((3,3)),get_hat(w)],axis = 1)],axis = 0)
	
def get_hat(vec):
	return np.array([[0,-vec[2],vec[1]],[vec[2],0,-vec[0]],[-vec[1],vec[0],0]])

#fuction for computing
def jacobian_pi(q):
	J = np.array([[1,0,-q[0]/q[2],0],
			  [0,1, -q[1]/q[2],0],
			  [0,0,0,0],
			  [0,0,-q[3]/q[2],1]])/q[2]
	return J

#fuction for computing Jacobian of observation z wrt to landmark mu
def get_H_imu(mu,mj,cam_T_imu,M_intsic):
	comm = mu@mj
	term = cam_T_imu @comm
	J = jacobian_pi(term)
	P =  np.concatenate([np.concatenate([np.eye(3),-1*get_hat(comm[:3])],axis = 1),np.zeros((1,6))],axis =0)
	H = M_intsic@J@cam_T_imu @P

	return H

#fuction for computing Jacobian of observation z wrt to inverse imu pose
def get_Jacobian_H(mu,ut,cam_T_imu,M_intsic):
	q = cam_T_imu @ut@mu
	J = jacobian_pi(q)
	P = np.concatenate([np.eye(3),np.zeros((3,1))],axis =1)
	H = M_intsic@J@cam_T_imu @ut@P.T

	return H

#fuction for computing world coordinates of the landmarks from the inverse imu pose and pixel features
def transform2world(feature, ut,cam_T_imu, M):
	array = np.ones((4,np.shape(feature)[1]))
	array[0,:]  = (feature[0,:]-M[0,2])/M[0,0]
	array[1,:]  = (feature[1,:]-M[1,2])/M[1,1]
	array[3,:]  = -(feature[0,:] - feature[2,:])/M[2,3]
	array = array/array[3,:]
	W_cood = inv(cam_T_imu@ut)@array
	return W_cood

# function to visualize the trajectory in 2D with landmarks
def visualize_trajectory(pose,landmark,path_name="Unknown",show_ori=False):

  fig,ax = plt.subplots(figsize=(7,7))
  n_pose = pose.shape[2]
  ax.plot(pose[0,3,:],pose[1,3,:],'r-',linewidth=2,label=path_name)
  ax.scatter(pose[0,3,0],pose[1,3,0],marker='s',label="start")
  ax.scatter(pose[0,3,-1],pose[1,3,-1],marker='o',label="end")
  ax.plot(landmark[0,:],landmark[1,:],'o',c= 'dimgrey',markersize = 2,alpha= 0.7,label = 'landmark')
  if show_ori:
	  select_ori_index = list(range(0,n_pose,max(int(n_pose/50), 1)))
	  yaw_list = []
	  for i in select_ori_index:
		  _,_,yaw = mat2euler(pose[:3,:3,i])
		  yaw_list.append(yaw)
	  dx = np.cos(yaw_list)
	  dy = np.sin(yaw_list)
	  dx,dy = [dx,dy]/np.sqrt(dx**2+dy**2)
	  ax.quiver(pose[0,3,select_ori_index],pose[1,3,select_ori_index],dx,dy,\
		  color="b",units="xy",width=1)
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.axis('equal')
  ax.grid(False)
  ax.legend()
  plt.show(block=True)

  return fig, ax