from LRC import *

import csv
import threading 
import os
import socket
import sys
import logging
import time
import argparse
import datetime
from multiprocessing import Pool

import numpy as np
import glob
import pandas
 
import cv2 

from numpy import sin,sign,eye, zeros,cos, ones,vstack,hstack, matmul,transpose, quantile, mean, std , maximum, minimum, amax,amin
from numpy.linalg import norm,inv,det, matrix_power
from numpy.random import randn,randint,uniform
from math import pi,sqrt,atan2,sin,cos, floor
from controlpy.synthesis import controller_lqr_discrete_time as dlqr
import gurobipy as gp
from gurobipy import  *

from pytope import Polytope
from pytope.polytope import intersection, minkowski_sum
import multiprocessing

import scipy.special as sc


# The basic mujoco wrapper.
from dm_control import mujoco

# Control Suite
from dm_control import suite

# General
import copy
import os
from IPython.display import clear_output 
import scipy
import scipy.stats as stats
import scipy.sparse as sp
from scipy.linalg import block_diag

# Graphics-related
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from IPython.display import HTML
import PIL.Image 
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points
from functools import partial

from itertools import repeat
np.set_printoptions(suppress=True,precision=5,linewidth = np.inf, floatmode='maxprec')


# Initialize random seed to generate noise samples for simulation
np.random.seed(42)


total_num_processes = 20 # Number of processes
multi_process = False # Run multiprocessing instead of single loop
capture_video = False # Records frames for playback

time_start = time.time() #just for metrics of how long simulation takes 

# #  MPC Controller Parameters
Wslack = 100000;    # constraint softening, not currently used
soft_flg = 0;     #  1 for slacks on 0 for hard constraints. , not currently used
 
print("Loading parameters")
# #  Loading all system parameters
A,B,U,nx,nu,wub_true,wlb_true, y_0, Q,R,N,dt,Pinf = sys_load() #Loads relevant system dynamics, cost, and noise matrices

nsamples = 50 # Number of samples (n) used to construct V_hat(n)
num_iterations = 1000 # Number of rollouts to test for each V_hat(n)


# Simulated noise parameters for mujoco (not needed in experiment)
lower = -0.05
upper = 0.05
mu = 0.0
sigma = 0.03 
v_samples = scipy.stats.truncnorm.rvs((lower-mu)/sigma,(upper-mu)/sigma,loc=mu,scale=sigma,size=(2,20000)) # full collection of noise samples
trueV = Polytope(lb = np.ones(2)*lower, ub = np.ones(2)*upper); # true noise which is unknown in experiments

W = Polytope(lb = wlb_true*ones([nx,1]), ub = wub_true*ones([nx,1])); # additive uncertainty, not currently used

print("Loading Stabilizing Controller")
Ks, Ke = DefineStabilizingController(A, B,Q,R); # Gain for feedback controller

 
# L,_,_ = dlqr(A,eye(nx),eye(nx),eye(nx)*10);
L = block_diag(0.25,0.267) # Gain for observer


conf = 0.9975; # desired confidence value (95% confidence support)
 
print("Startin constructing V") 

# Noise samples used for V(n) 
Vstacked = np.hstack(( v_samples[:,:nsamples-2] ,amax(v_samples[:,:],axis=1).reshape(2,1),amin(v_samples[:,:],axis=1).reshape(2,1)) ) 
# Vstacked = v_samples[:,:nsamples]

# Construct support V_hat(n) using noise samplse
Vmn, v_lb, v_ub, minRTilde, minRBar, minR, Hxn_nom, hxn_nom, Hxbar, hxbar, Hubar, hubar, Xbar, Ubar, Xn_nom, polxhat0 = v_constructGauss( Vstacked, W, conf, nx,nu,A,B,Q,R,U,N,y_0, Ks, Ke, L);
polygonxhat = Polygon([polxhat0.V[0,:],polxhat0.V[1,:],polxhat0.V[2,:],polxhat0.V[3,:]]) # Used to pick an exhat from this set at t=0

# Must initialize the nominal state and the observer state
maxB = np.max([v_ub, -v_lb], axis =0);
vmnN0 = Polytope(lb = -maxB, ub = maxB);
mthX0 =  y_0 + (-vmnN0);       


# adds a little slack for numerical error during check at each step if ex-exnom \belong_to R, exhat-exnom \belong_to RBar, ex-exhat \belong_to Rtide
brb = 0.0001
setcheckeraddbound = Polytope(lb = -np.array([brb,brb]), ub = np.array([brb,brb]));
minRTildecheck = minRTilde + setcheckeraddbound
minRcheck = minR + setcheckeraddbound


def runloop(polygonxhat,num_iterations,capture_video,process_number):
    with gp.Env() as environ:
        environ.setParam('OutputFlag', 0)

        # Construct gurobi model
        e_x,e_u,previous_u_cup,m,epsilon = construct_Models(Hxn_nom, hxn_nom, Hxbar, hxbar, Hubar, hubar, Pinf, nx,nu,A,B,Q,R,U,N,y_0, soft_flg, Wslack)

        success = 0
        step = 0
        trial_number = 0

        # Arrays to store state and input data from rollouts
        array_size = 200
        ex_cl = zeros([num_iterations,nx,array_size+1]);
        ex_nom_cl = zeros([num_iterations,nx,array_size+1]);
        ex_hat_cl = zeros([num_iterations,nx,array_size+1]);
        eu_cl = zeros([num_iterations,nu,array_size]);


        # Initialize mujoco environment
        env = suite.load(domain_name="kendama_catch_simulation", task_name="catch")
        action_spec = env.action_spec()
        time_step = env.reset(process_number*num_iterations+trial_number) # Reset the environment and set the random seed (changes initial conditions)

        duration = 1  # Max duration of episode (unless terminal condition met earlier)


        frames = []
        ticks = []
        rewards = []
        observations = []
        ticks_list = []
        rewards_list = []
        observations_list = []
        exnom_list = []
        exhat_list = []
        trialnum_list = [] 

        while trial_number<num_iterations:
            np.random.seed(process_number*num_iterations+trial_number) #Setting random seed for trial (used for simulated noise on measurements)
         
            ex_cl[trial_number,:,0] = env.physics.ball_to_cup()

            observations = [time_step.observation]  
            ticks= [env.physics.data.time]
            rewards = [time_step.reward]
            curr_succ = True
            step = 0

            while env.physics.data.time < duration:
                obs = observations[-1]

                # Measurement y0
                cup_minus_ball = env.physics.ball_to_cup()+scipy.stats.truncnorm.rvs((lower-mu)/sigma,(upper-mu)/sigma,loc=mu,scale=sigma,size=2)

                if step == 0:
                    # Pick a point inside exhat that lies closest to the noisy measurement
                    point = Point(cup_minus_ball)    
                    p1, p2 = nearest_points(polygonxhat, point)
                    xhatchose0 = np.array(p1.coords[:]).flatten()


                    # newX0 =  cup_minus_ball + (-vmnN0); 
                    # try:
                    #     newpolxhat0 = newX0 - minRTilde;
                    # except:
                    #     newpolxhat0 = -(minRTilde - newX0) ;   
                    # newpolygonxhat = Polygon([newpolxhat0.V[0,:],newpolxhat0.V[1,:],newpolxhat0.V[2,:],newpolxhat0.V[3,:]])
                    # newhatp1, _ = nearest_points(newpolygonxhat, Point(cup_minus_ball))
                    # newxhatchoose = np.array(newhatp1.coords[:]).flatten()
                    # newxhatchoose = newpolxhat0.V[1,:]

                    # newpolxnom0 = newxhatchoose + (-minRBar)
                    # newpolygonxnom = Polygon([newpolxnom0.V[0,:],newpolxnom0.V[1,:],newpolxnom0.V[2,:],newpolxnom0.V[3,:]])
                    # newnomp1, _ = nearest_points(newpolygonxnom, Point(cup_minus_ball))
                    # newxnomchoose = np.array(newnomp1.coords[:]).flatten()
                    # newxnomchoose = newpolxnom0.V[1,:]

  
                    ex_hat_cl[trial_number,:,0] = xhatchose0.flatten();  

                    # Pick a point that is within Rbar away from exhat, can be picked arbitrarily or optimally
                    ex_nom_cl[trial_number,:,0] = xhatchose0.flatten()-minRBar.V[1,:]*0.6; #selected to show success improvement despite suboptimal choice

                ball_vel = obs['velocity'][2:4] # in experiment, this is estimated


                if step<N: 

                    # Check at each step if ex-exnom \belong_to R, exhat-exnom \belong_to RBar, ex-exhat \belong_to Rtide
                    if not(minRTildecheck.contains(env.physics.ball_to_cup()-ex_hat_cl[trial_number,:,step])):
                        print("BROKEN HATS",step,env.physics.ball_to_cup()-ex_hat_cl[trial_number,:,step],env.physics.ball_to_cup(),ex_hat_cl[trial_number,:,step])
                        # input("BROKEN HATS")

                    if not(minRcheck.contains(env.physics.ball_to_cup()-ex_nom_cl[trial_number,:,step])):
                        print("BROKEN NOMS",step,env.physics.ball_to_cup()-ex_nom_cl[trial_number,:,step],env.physics.ball_to_cup(),ex_nom_cl[trial_number,:,step])

                    m[step].update() 

                    # for gurobi initialization
                    x2index = str(N+1-step)
                    x0_1 = m[step].getVarByName(str(step)+ 'e_x[0]')
                    x0_2 = m[step].getVarByName(str(step)+ 'e_x['+x2index +']') 

                    optimize_ebar = False  # If we want to select exnom optimally , not currently used and doesnt change guarantees, may affect performance (not adversely)
                    rollout_optimized_ehat = False # If we want to propagate the initial optimal exnom or recalcuate (t>0), may affect performance (not adversely)

                    if rollout_optimized_ehat:
                        if optimize_ebar and step ==0:
                            abcd = -minRBar.A@ex_hat_cl[trial_number,:,step]+minRBar.b.flatten()
                            # print('constr',abcd)

                            m[step].addConstr(-x0_2<=abcd[0])
                            m[step].addConstr(x0_1<=abcd[1])
                            m[step].addConstr(-x0_1<=abcd[2])
                            m[step].addConstr(x0_2<=abcd[3])
                        else:
                            x0_1.LB = ex_nom_cl[trial_number,0,step]
                            x0_1.UB = ex_nom_cl[trial_number,0,step]
                            x0_2.LB = ex_nom_cl[trial_number,1,step]
                            x0_2.UB = ex_nom_cl[trial_number,1,step]
                    else:
                        if optimize_ebar:
                            abcd = -minRBar.A@ex_hat_cl[trial_number,:,step]+minRBar.b.flatten()
                            # print('constr',abcd)

                            m[step].addConstr(-x0_2<=abcd[0])
                            m[step].addConstr(x0_1<=abcd[1])
                            m[step].addConstr(-x0_1<=abcd[2])
                            m[step].addConstr(x0_2<=abcd[3])
                        else:
                            x0_1.LB = ex_nom_cl[trial_number,0,step]
                            x0_1.UB = ex_nom_cl[trial_number,0,step]
                            x0_2.LB = ex_nom_cl[trial_number,1,step]
                            x0_2.UB = ex_nom_cl[trial_number,1,step]


                    # affects cost by encouraging input to stay near current velocity (does not change any guarantees)
                    previous_u_cup_2index = str(N-step)
                    previous_u_cup_1 = m[step].getVarByName(str(step)+ 'e_uprevious[0]')
                    previous_u_cup_2 = m[step].getVarByName(str(step)+ 'e_uprevious['+x2index +']') 
                    previous_u_cup_1.LB = env.physics.vel_ball_to_cup()[0]
                    previous_u_cup_1.UB = env.physics.vel_ball_to_cup()[0]
                    previous_u_cup_2.LB = env.physics.vel_ball_to_cup()[1]
                    previous_u_cup_2.UB = env.physics.vel_ball_to_cup()[1]



                    m[step].optimize()

                    if m[step].status==2:

                        # Get nominal input from gurobi
                        v_hor = e_u[step].X

                        if rollout_optimized_ehat:
                            if optimize_ebar and step == 0:
                                ex_nom_cl[trial_number,:,step] = e_x[step].X[:,0] 
                            ex_nom_cl[trial_number,:,step+1]= A@ex_nom_cl[trial_number,:,step]+  B@v_hor[:,0]; 
                        else:
                            if optimize_ebar:
                                ex_nom_cl[trial_number,:,step] = e_x[step].X[:,0]
                            else:
                                ex_nom_cl[trial_number,:,step+1]= A@ex_nom_cl[trial_number,:,step]+  B@v_hor[:,0]; 

                        # compute closed loop input with feedback control
                        eu_cl[trial_number,:,step]= Ke@(ex_hat_cl[trial_number,:,step]-ex_nom_cl[trial_number,:,step])+v_hor[:,0];                  # Closed loop control 

                        # Propogate observer
                        ex_hat_cl[trial_number,:,step+1]= A@ex_hat_cl[trial_number,:,step]+ B@eu_cl[trial_number,:,step] + L@(cup_minus_ball-ex_hat_cl[trial_number,:,step]).flatten()
                        
                        # convert error frame input to cup input
                        action = eu_cl[trial_number,:,step]+ball_vel
                    else:
                        print(step,"failed optimize")
                        
                        # stay at current position if solver fails
                        action = np.array([0,0])
                else:

                    # stay at current position after t>N
                    action = np.array([0,0])


                gear_ratio = 20 #for mujoco actuator input
                
                #step environment in mujoco
                time_step = env.step(np.append(action*gear_ratio,9.86)) 

                # update closed loop input
                ex_cl[trial_number,:,step+1] = env.physics.ball_to_cup()
 
                if capture_video:
                    camera0 = env.physics.render(camera_id=2, height=480, width=640)
                    # camera1 = env.physics.render(camera_id=1, height=200, width=200)
                    # frames.append(np.hstack((camera0, camera1)))
                    frames.append(camera0)
                rew = time_step.reward
                rewards.append(rew)
                observations.append(time_step.observation) 
                ticks.append(env.physics.data.time)
                step+=1 
                if time_step.step_type==2:
                    if rew>0:
                        print("Success")
                    else:
                        
                        print("TRIAL NUMBER ", process_number*num_iterations+trial_number,'Failed')
                        curr_succ = False
                    time_step = env.reset(process_number*num_iterations+trial_number) 
                    break
            if curr_succ:
                print("TRIAL NUMBER ", process_number*num_iterations+trial_number,'Success')
                success +=1
                time_step = env.reset(process_number*num_iterations+trial_number) 
            observations_list.append(observations)
            ticks_list.append(ticks)
            rewards_list.append(rewards)

            trialnum_list.append(trial_number)
            trial_number+=1 

        return observations_list,ticks_list,rewards_list,success,trialnum_list,ex_cl,ex_nom_cl,ex_hat_cl,frames


##################################################################################################
# This section for video display settings is from dm_control tutorial.ipynb
sc.seterr(all='ignore' )
# Font sizes
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def display_video(frames, framerate=30):
    height, width, _ = frames[0].shape
    dpi = 70
    orig_backend = matplotlib.get_backend()
    # matplotlib.use('Agg')  # Switch to headless 'Agg' to inhibit figure rendering.
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    # matplotlib.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])
    def update(frame):
        im.set_data(frame)
        return [im]
    interval = 1000/framerate
    anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
                                 interval=interval, blit=True, repeat=True)
    return anim #fig,update,frames,interval #HTML(anim.to_html5_video())

##################################################################################################
 

if __name__ == '__main__':
    p = Pool(processes=total_num_processes)

    num_iterations_per_thread = int(np.ceil(num_iterations/float(total_num_processes)))

    if multi_process == True: 
        datalist =  p.starmap(runloop, zip(repeat(polygonxhat), repeat(num_iterations_per_thread), repeat(capture_video), [i for i in range(total_num_processes)]))
        p.close() 
    else:
        datalist = [runloop(polygonxhat,num_iterations,capture_video,0)]

    ticks_list = []
    rewards_list = []
    observations_list = []
    trialnum_list = []
    success = 0
    frames = []

    cshape = datalist[0][5][0].shape
    exnom_list = np.empty(np.hstack((0,cshape)))
    exhat_list = np.empty(np.hstack((0,cshape)))
    ex_list = np.empty(np.hstack((0,cshape)))

    for data in datalist:
        observations_list.extend(data[0])
        ticks_list.extend(data[1])
        rewards_list.extend(data[2])
        success+=data[3]
        trialnum_list.append(data[4])
        ex_list= np.concatenate((ex_list,data[5]),axis=0)
        exnom_list= np.concatenate((exnom_list,data[6]),axis=0)
        exhat_list= np.concatenate((exhat_list,data[7]),axis=0)
        frames.extend(data[8])

    posx_list = []
    velx_list = []
    posz_list = []
    velz_list = [] 
    for k,obs in enumerate(observations_list):
        observations = observations_list[k]
        cvel = observations[N]['velocity'][0:2]
        bvel = observations[N]['velocity'][2:4]
        dif = cvel-bvel 
        velx_list.append(dif[0])
        velz_list.append(dif[1])
        cpos = observations[N]['position'][0:2]
        bpos = observations[N]['position'][2:4]
        difp = cpos-bpos 
        posx_list.append(difp[0])
        posz_list.append(difp[1])

    capture_video = False
    if capture_video:
      # html_video = display_video(frames, framerate=1./env.control_timestep())
      html_video = display_video(frames, framerate=20)
      plt.show()


    print("Ke",Ke)
    print("L",L)
    print("minRTilde",minRTilde.V)
    print("minRBar",minRBar.V)
    print("minr",minR.V)
    print("Xbar",Xbar.V)
    print("ubar",Ubar.V)
    print("Xn",Xn_nom.V)
    print("trueV",trueV.V)
    print("vmn",Vmn.V)
    print("y0",y_0.V)
    print("mthX0",mthX0.V)
    print("polxhat0",polxhat0.V)

    print("ex",np.mean(ex_list[:,:,N],axis=0),"exnom",np.mean(exnom_list[:,:,N],axis=0),"exhat",np.mean(exhat_list[:,:,N],axis=0),)
    print("ex-exnom",np.mean(ex_list[:,:,N]-exnom_list[:,:,N],axis=0), "ex-exhat",np.mean(ex_list[:,:,N]-exhat_list[:,:,N],axis=0),)

    print("nsamples",nsamples,"num_iterations",num_iterations,"success rate",success/num_iterations)
    print("pos", np.around(np.mean(posx_list),5), np.around(np.mean(posz_list),5),"std", np.around(np.std(posx_list),5), np.around(np.std(posz_list),5))
    print("vel", np.around(np.mean(velx_list),5), np.around(np.mean(velz_list),5),"std", np.around(np.std(velx_list),5), np.around(np.std(velz_list),5))
    
    print("total time",time.time()-time_start)
