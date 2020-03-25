from utils import *
import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.linalg import block_diag


if __name__ == '__main__':
	filename = "./data/0034.npz"
	t,features,linear_velocity,rotational_velocity,K,b,cam_T_imu = load_data(filename)


	fsu = K[0,0]
	fstheta = K[0,1]
	fsv = K[1,1] 
	cu= K[0,2]
	cv = K[1,2]

	#setting intrinsic calibration matrix
	M_intsic = np.array([[fsu,0,cu,0],
	            [0,fsv,cv,0],
	            [fsu,0,cu,-fsu*b],
	            [0,fsv,cv,0]])



	#using downsampling of features by 10
	down_features = features[:,::25,:]
	M = down_features.shape[1]

	#initialising mean and sigma, noise parameters
	V = 10
	W = np.eye(6)*0.001
	mu_map = -np.ones((4,M))
	cov_map = np.eye(3*M)
	mu_t = np.eye(4)
	sigma_t=100* np.eye(6,6)

	#set to track already visited features
	visited_landmarks = set()
	pose_ = inv(mu_t).reshape(4,4,1)

	#matrix to help with the calculation of mu of landmark
	expand = np.vstack((np.identity(3),np.zeros((1,3))))
	expand = np.kron(np.eye(np.shape(down_features)[1]),expand)


	for i in np.arange(1,t.shape[1]): 
	    
	    tau =  t[0,i]- t[0,i-1]

	############## (a) IMU-based Localization via EKF Prediction ################################

	    mu_t = expm(-tau*get_uhat(linear_velocity[:,i],rotational_velocity[:,i]))@mu_t
	    adj = expm(-tau*get_uadjoint(linear_velocity[:,i],rotational_velocity[:,i]))
	    sigma_t = adj@sigma_t@adj.T + W#tau*tau*np.diag(np.random.normal(0,1,6))
	    pose_ =np.concatenate([pose_, inv(mu_t).reshape(4,4,1)],axis =2)

	    #keeping track of features already visited and seen for the first time at timestep i 
	    present_landmarks = set(np.where(np.sum(down_features[:,:,i],axis=0) !=-4 )[0])
	    update_landmarks = present_landmarks.intersection(visited_landmarks)
	    init_landmarks = present_landmarks.difference(visited_landmarks)
	    visited_landmarks = visited_landmarks.union(present_landmarks)
	    
	############# (b) Landmark Mapping via EKF Update ############################################

	    #if observed for first time transform to world coordinates of the landmarks from the inverse imu pose and pixel features
	    mu_map[:,list(init_landmarks)] = transform2world(down_features[:,list(init_landmarks),i],mu_t,cam_T_imu,M_intsic)
	    
	    upl = list(update_landmarks)
	    if(upl):
	    	#H of landmarks
	        H_U = np.zeros((4*len(list(update_landmarks)),3*M))
	        for j in range(len(upl)):
	            H_i = get_Jacobian_H(mu_map[:,upl[j]],mu_t,cam_T_imu,M_intsic)
	            H_U[4*j:4*(j+1),3*upl[j]:3*(upl[j]+1)] = H_i

	        IxV  = V*np.eye(4*len(upl))

	        #Kalman gain for landmarks
	        Kt_obs = cov_map@H_U.T@inv(H_U@cov_map@H_U.T+IxV)
	        
	        #calcilating innovation
	        aterm = cam_T_imu @mu_t@mu_map[:,upl]
	        z_t_ = M_intsic@(aterm/aterm[2,:])        
	        zt = down_features[:,upl,i]
	        diff = zt - z_t_

	        mu_map = (mu_map.reshape(-1,1,order='F') + expand@Kt_obs@(diff).reshape(-1,1,order='F')).reshape(4,-1,order='F')  

	############# (c)Visual-Inertial SLAM: IMU and Landmark joint update ############################################ 
	        
	        #H of imu
	        H_imu = np.zeros((4*len(upl),6))
	        
	        for j in range(len(upl)):
	            H_it = get_H_imu(mu_t,mu_map[:,upl[j]],cam_T_imu, M_intsic)
	            H_imu[4*j:4*(j+1),:] = H_it
	        
	        # joint H of imu and landmarks
	        H_joint = np.concatenate([H_imu,H_U],axis = 1)
	        joint_cov = block_diag(sigma_t,cov_map)
	        IxV  = V*np.eye(4*len(upl))
	        #computing joint Kalman gain
	        Kalman_joint= joint_cov@H_joint.T@inv(H_joint@joint_cov@H_joint.T+IxV)
	        
	        product = Kalman_joint@diff.reshape(-1,order='F')

	        #updated imu inverse pose mean and coveriance
	        mu_t = expm(get_uhat(product[:3],product[3:]))@mu_t
	        updated_joint_cov = (np.eye(joint_cov.shape[0])-Kalman_joint@H_joint)@joint_cov
	        sigma_t = updated_joint_cov[:6,:6]
	        cov_map = updated_joint_cov[6:,6:]

	# save the figure to file
	fig ,ax = visualize_trajectory(pose_,mu_map,show_ori=True)
	fig.savefig('./0034_DS_25.png',dpi = 150,bbox_inches='tight')   
	plt.close(fig)
