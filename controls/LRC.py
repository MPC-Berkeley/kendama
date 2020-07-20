
import csv
import threading 
import os
import socket
import sys
import logging
import time
import argparse
import datetime
 
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
import scipy.sparse as sp
from scipy.linalg import block_diag
import matplotlib.pyplot as plt

from pytope import Polytope
from pytope.polytope import intersection, minkowski_sum
import multiprocessing

import scipy.special as sc
sc.seterr(all='ignore' )

def DefineStabilizingController(A, B,Q,R):
    # UNTITLED2 Summary of this function goes here
    # Detailed explanation goes here

    Ks,_,_ = dlqr(A,B,Q,R);
    Ks = -Ks;


    Ke,_,_ = dlqr(A,B,Q,R);
    Ke = -Ke;

    return Ks, Ke
 
def stackarrays(m,n, *args):
    # m is number of rows
    # n is number of columns
    # args fill across columns first

    if len(args) != m*n:
        print("Invalid number of entries")
        return
    i = 0
    return np.vstack((np.hstack((args[0],args[1])),np.hstack((args[2],args[3]))))

def size(M,n):
    if isinstance(M,list):
        return len(M)
    else:
        return M.shape[n-1]

def computeInvariantUgo( Acl, W ):
    #computeInvariant: If the algorithm in this function converges it returns a Robust Positive Invariant (RPI) set
    #   Acl: This matrix is the closed-loop matrix given by Acl = A - BK
    #   W: This polytope defined the domain of the uncertanty. It is defined
    #   using MPC. An 2d example is W = Polytope([1 1;1 -1; -1 1; -1 -1]); 


    X = [zeros([size(Acl,1),1])]; # Initialize the set
    for i in range(10000): 
        if i == 0:
            X.append(Acl@X[i] + W);  # Propagate the uncertanty
        else:
            X.append(Acl*X[i] + W);  # Propagate the uncertanty
        # Check if the algorithm has covnerged 
        if i > 1: 
            if (X[i+1].contains(X[i])) and (X[i].contains(X[i+1])):
                invariantSet = X[i+1]; # Set invaraint to the current iterate
                print('Invariant set computed in i = ',str(i),' iterations')         
                return invariantSet  
 

# Sets a high confidence value. Uses data. Constructs support. Checks
# feasibility of MPC problem from x_0. If infeasible, lowers confidence and
# repeats. if feasible, this support is rolled out to the main code to
# complete an iteration. Then data is collected and process is repeated
# again. 

def isEmptySet(P):
    return any([dim==0 for dim in P.V.shape])

def v_constructGauss(v_samples, W, conf, nx,nu,A,B,Q, R, U,N, y_0, Ks, Ke, L):

    bsSize = 1000;   # Bootstrap copies 
    vbs = zeros([bsSize,nx,size(v_samples,2)]);                 
    ## First construct the convex hull vertices 
    mincvxH = amin(v_samples,axis=1).reshape(nx,1);
    maxcvxH = amax(v_samples,axis=1).reshape(nx,1);

    ## Start Bootstrap here
    meanEmp = mean(v_samples,1);
    stdEmp = std(v_samples,1); 

    for j in range(bsSize):
        for i in range(size(v_samples,2)):
            entr = randint(0,size(v_samples,2)); 
            vbs[j,:,i] = v_samples[:,entr];

    meanBatch  = zeros([nx,bsSize]);
    stdBatch      = zeros([nx,bsSize]); 

    for j in range(bsSize):
        meanBatch[:,j] = mean(vbs[j,:,:],1);
        stdBatch[:,j] =  std(vbs[j,:,:],1);

    ###### Take the confidence set #########
    minMu = zeros([nx,1]);
    maxMu = zeros([nx,1]);
    maxStd = zeros([nx,1]);

    ###########################
    ##
    feas_conf = 0;
    count = 0; 
    count2 = 0; 


    print("Adding Constraints")

    while feas_conf == 0:     
        print(count,"=========================================================================================================================================================")
        if conf >= 0.001:
            conf =  conf*(0.9999**count);    
            for i in range(nx):
                minMu[i,0] = quantile(meanBatch[i,:],(1-conf)/2);
                maxMu[i,0] = quantile(meanBatch[i,:],1-(1-conf)/2);
                maxStd[i,0] = quantile(stdBatch[i,:],1-(1-conf)/2);
            
            ###########################

            v_lb = minimum(mincvxH, minMu - 3.08*maxStd);   
            v_ub = maximum(maxcvxH, maxMu + 3.08*maxStd);
            

            # v_lb = minMu - 3.08*maxStd;   
            # v_ub = maxMu + 3.08*maxStd;                       # NO CVX HULL UNION 
            Vmn = Polytope(lb = v_lb, ub = v_ub);            # CVX HULL OF UNION 
       
        else :
            v_lb = min(mincvxH, meanEmp - (3.08-0.1*count2)*stdEmp);   
            v_ub = max(maxcvxH, meanEmp + (3.08-0.1*count2)*stdEmp);
             
            # v_lb = meanEmp - (3.08-0.1*count2)*stdEmp;   
            # v_ub = meanEmp + (3.08-0.1*count2)*stdEmp;                          # NO CVX HULL UNION
            
            Vmn = Polytope(lb = v_lb, ub = v_ub);                            # CVX HULL OF UNION 
            count2 = count2 + 1; 
        
        ### ALL CHECKS MUST GO THROUGH WITH THIS SET

    ####################### MAYNE APPROACH (1) ######################
        ALcl = A-L;                                                             # FOLLOW MAYNE NOTATIONS HERE 
        LVsc = (-L)*Vmn;    
        DeltaTilde = W + LVsc; 
        minRTilde  = computeInvariantUgo(ALcl, DeltaTilde);

        ### second piece 
        Acl = A + B@Ke;
        DeltaBar = L*minRTilde + L*Vmn;

        Vlist = DeltaBar.V; 
        tolV = 300;                 # after 100 vertices, max box outside 
        
        if size(Vlist,1) < tolV:
            print('***** NOT FITTING BOX. EXACT MIN INVARIANT SET ATTEMPT******')
            l = np.array([[min(Vlist[:,0])],[min(Vlist[:,1])]]);
            u = np.array([[max(Vlist[:,0])],[max(Vlist[:,1])]]);
            polOut = Polytope(lb = l, ub = u);
            minRBar = computeInvariantUgo(Acl, polOut); 
            print("UGO COMPUTED")
        else:
            print('***** FITTING BOX. APPROX MIN INVARIANT SET ******')
            minRBar = computeInvariantUgo(Acl, DeltaBar);
            print("UGO COMPUTED")
        
        minR = minRTilde + minRBar;                                              # NET PIECE
        print("minRTilde",minRTilde.V)
        print("minRBar",minRBar.V)
        print("minr",minR.V)


        # Compute the Tightened Ubar 
        Ubar = U-Ke*minRBar; 
        Hubar = Ubar.A; 
        hubar = Ubar.b;   

        # Terminal Condition = 0
        Xn_nom = Polytope(lb = zeros(nx), ub = zeros(nx))
        Hxn_nom = Xn_nom.A
        hxn_nom = Xn_nom.b


        ## Checking if x_hat exists
        maxB = np.linalg.norm([v_ub, -v_lb],np.inf,0) 
        vmnN0 = Polytope(lb = -maxB, ub = maxB);
        mthX0 =  y_0 + (-vmnN0);                                              
        try:
            polxhat0 = mthX0 - minRTilde;
        except:
            polxhat0 = -(minRTilde - mthX0) 
        

        if isEmptySet(polxhat0) == 0  and isEmptySet(Ubar) == 0: 
            gamma_hat = np.linalg.norm((polxhat0 + (-minRBar)).V,np.inf,0) 
            Xbar = Polytope(lb=-gamma_hat,ub=gamma_hat)
            Hxbar = Xbar.A
            hxbar = Xbar.b 
            if isEmptySet(Xbar) == 0:
                print("Xbar",Xbar.V)
                print("Ubar",Ubar.V)
                print("Xn_nom",Xn_nom.V)  
                print("Vmn",Vmn.V) 
                feas_conf = 1
            else:
                feas_conf = 0;  
             
        else:
            print("no v gauss - lower confidence")
            feas_conf = 0;   
        count = count + 1; 
    
        
    return Vmn, v_lb, v_ub, minRTilde, minRBar, minR, Hxn_nom, hxn_nom, Hxbar, hxbar, Hubar, hubar , Xbar, Ubar, Xn_nom, polxhat0

def sys_load():
 
    dt = 0.01 
    N =  25

    A = eye(2)
    B = dt*eye(2)

    nx = A.shape[1]
    nu = B.shape[1]  

    Q =  np.diag([500,500]) 
    R =  np.diag([0.4,0.4])

    ua = 8
    uxmin = -1.0*ua
    uxmax =  1.0*ua
    uzmin = -1.0*ua
    uzmax =  1.0*ua
    U = Polytope(lb = np.array([[uxmin],[uzmin]]),ub = np.array([[uxmax],[uzmax]]))


    # # #  Defining Process Noise Bounds 
    wlb_true = -0.00; # Lower bound of additive noise value ######
    wub_true = -wlb_true; # Upper bound of additive noise value ######
 
    y_0 = Polytope(lb=np.array([-0.3492,-0.2457]),ub=np.array([-0.3158,-0.2095]))  


    _, Pinf,_ = dlqr(A,B,Q,R);  
    return A,B,U,nx,nu,wub_true,wlb_true, y_0, Q,R,N,dt ,Pinf


def construct_Models(Hxn_nom, hxn_nom, Hxbar, hxbar, Hubar, hubar, Pinf, nx,nu,A,B,Q,R,U,N,y_0, soft_flg, Wslack,env=gp.Env()):
    env.setParam('OutputFlag', 0)

    # Initialize each as a list in case we wanted time-varying cost matrices 
    m = []
    epsilon = []
    previous_u_cup = []
    e_x = []
    e_u = []
    cost_stage = []
    e_uprevious = []
    U_penalty = []
    U_between_penalty = []
    polxnom_A = []
    polxnom_b = [] 


    for step in range(N):

        m.append(gp.Model(env=env, name=str(step)+"matrix"))
        m[step].setParam('OutputFlag', 0)


        # Decision variables
        e_x.append(m[step].addMVar(shape=(nx,N+1-step), lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name=str(step)+"e_x"))
        e_u.append(m[step].addMVar(shape=(nu,N-step),lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name=str(step)+"e_u"))
        e_uprevious.append(m[step].addMVar(shape=(nu,N),lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name=str(step)+"e_uprevious"))
        previous_u_cup.append(m[step].addMVar(shape=(nu),lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name=str(step)+"u_ball_nom"))
        epsilon.append(m[step].addMVar(shape=(1),  vtype=GRB.CONTINUOUS, name=str(step)+"epsilon"))

        # Penalty for input between steps, does not change any guarantees
        U_penalty.append(np.diag([50,500]))  
        U_between_penalty.append(np.diag([50,50]))  

        # Initialize cost
        cost_stage.append(0)
        # k=0
        # cost_stage[step] = cost_stage[step] +  e_u[step][:,k]@U_penalty[step]@e_u[step][:,k]-e_u[step][:,k]@U_penalty[step]@e_uprevious[step][:,k]*2+e_uprevious[step][:,k]@U_penalty[step]@e_uprevious[step][:,k]
        
        for k in range(N-step):

            # Dynamics
            m[step].addConstr(e_x[step][:,k+1] == A@e_x[step][:,k]+ B@(e_u[step][:,k]), name=str(step)+"xpred"+str(k))

            # State and input constraints
            m[step].addConstr(Hxbar@e_x[step][:,k]<= hxbar.flatten(), name=str(step)+"Hubar1"+str(k))
            m[step].addConstr(Hubar@e_u[step][:,k]<= hubar.flatten(), name=str(step)+"Hubar2"+str(k))

            # Add stage cost
            cost_stage[step] = cost_stage[step] + e_x[step][:,k]@Q@e_x[step][:,k]+e_u[step][:,k]@R@e_u[step][:,k]*k**2;
            
            if k<N-1-step:
                cost_stage[step] = cost_stage[step] +  e_u[step][:,k+1]@U_between_penalty[step]@e_u[step][:,k+1]-e_u[step][:,k+1]@U_between_penalty[step]@e_u[step][:,k]*2+e_u[step][:,k]@U_between_penalty[step]@e_u[step][:,k];

            cost_stage[step] = cost_stage[step] +  e_u[step][:,k]@U_penalty[step]@e_u[step][:,k]-e_u[step][:,k]@U_penalty[step]@e_uprevious[step][:,k]*2+e_uprevious[step][:,k]@U_penalty[step]@e_uprevious[step][:,k]
        

        # Terminal constraint (0 in this case)
        m[step].addConstr(Hxn_nom@e_x[step][:,N-step]<= hxn_nom.flatten(), name=str(step)+"Hxnom")
        m[step].setObjective(cost_stage[step], GRB.MINIMIZE)
        m[step].update() 

        # Add terminal cost
        cost_stage[step] = cost_stage[step] + e_x[step][:,N-step]@Pinf@e_x[step][:,N-step]  + epsilon[step]@epsilon[step]*soft_flg*Wslack
  
    
    return e_x,e_u,previous_u_cup,m,epsilon