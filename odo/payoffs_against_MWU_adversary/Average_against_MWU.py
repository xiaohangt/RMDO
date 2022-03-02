

import matplotlib.pyplot as plt
import numpy as np
import math
import time
from OMD_update import OMD
import pickle
import math
from pathlib import Path
import argparse
from numpy.random import RandomState

parser = argparse.ArgumentParser()
parser.add_argument('--game', type=str, default="", help='Game')
parser.add_argument('--prefix', type=str, default="", help='Game')

SEED = int(os.getenv('SEED')) if os.getenv('SEED') is not None else 1000


#pick the game here
GAME_NAME = parser.parse_args().game
PREFIX = parser.parse_args().prefix
PATH = 'payoffs_data/' + GAME_NAME
with open(PATH, "rb") as fh:
    payoffs = pickle.load(fh)
n=len(payoffs)
m=len(payoffs)
A=payoffs
iterate_max=10000
mu_x=0.01
mu_y=0.01

##########
rs = RandomState(SEED)
start_idx = rs.randint(n)
##########

#*****MWU
x=np.array([1/n for i in range(n)])
z1=np.array([1/n for i in range(n)])
y=np.array([1/m for i in range(m)])
z2=np.array([1/n for i in range(m)])
iterate=0
average_perform=0
yaxis_mwu=[]
while iterate <iterate_max:
    average_perform=average_perform*iterate/(iterate+1)+ np.dot(x, np.matmul(A,y))/(iterate+1)
    yaxis_mwu.append(average_perform)
    xupdate=-mu_x*(np.matmul(A,y))
    xupdate=np.exp(xupdate)
    z1=np.multiply(x,xupdate)
    yupdate=mu_y*(np.matmul(x,A))
    yupdate=np.exp(yupdate)
    z2=np.multiply(y,yupdate)
    #mwu update for both player
    x=z1/sum(z1)
    y=z2/sum(z2) 
    iterate+=1

#******OSO
x_average=np.array([1/n for i in range(n)])
y_average=np.array([1/m for i in range(m)])
x_oso=np.array([0 for i in range(n)])
z_oso=np.array([0 for i in range(n)])
z_oso[start_idx]=1
y=np.array([1/m for i in range(m)])
z2=np.array([1/m for i in range(m)]) 
iterate=0
set_of_strategies=[]
average_perform_oso=0
yaxis_oso=[]
k=0
start1=time.time()
while iterate < iterate_max:
    average_perform_oso=average_perform_oso*iterate/(iterate+1)+ np.dot(z_oso, np.matmul(A,y))/(iterate+1)
    yaxis_oso.append(average_perform_oso)
    if np.argmin(np.matmul(A,y_average)) not in set_of_strategies:
        k+=1
        set_of_strategies.append(np.argmin(np.matmul(A,y_average)))
        x_oso[np.argmin(np.matmul(A,y_average))]=1/k
    yupdate=mu_y*(np.matmul(x_oso,A))
    yupdate=np.exp(yupdate)
    z2=np.multiply(y, yupdate)
    xupdate=-mu_x*(np.matmul(A,y))
    xupdate=np.exp(xupdate)
    z1=np.multiply(x_oso, xupdate)
    x_oso=z1/sum(z1)
    y=z2/sum(z2)
    z_oso=x_oso+0
    y_average=(1-1/(iterate+2))*y_average+1/(iterate+2)*y
    iterate+=1
end1=time.time()
print(set_of_strategies)
print(average_perform_oso)


#DO method #new added
x_do=np.array([0 for i in range(n)])
x_do[start_idx]=1
z_do=np.array([1/n for i in range(n)])
y=np.array([1/m for i in range(m)])
z2=np.array([1/m for i in range(m)])
iterate=0
iterate_1=1
B1=np.identity(n)
average_perform_do=0
set_of_strategies=[0]
y_average=np.array([1/m for i in range(m)])
x_average=x_do+0
k=1
yaxis_do=[]
while iterate <iterate_max:
    average_perform_do=average_perform_do*iterate/(iterate+1)+ np.dot(x_do, np.matmul(A,y))/(iterate+1)
    yaxis_do.append(average_perform_do)
    c=np.matmul(A,y_average)
    b=c[0]+0
    for i in range(n):
        if i in set_of_strategies and c[i] <=b:
            b=c[i]+0
            
    if np.max(np.matmul(x_average,A))-b<0.01 and np.argmin(np.matmul(A,y_average)) not in set_of_strategies:
        set_of_strategies.append(np.argmin(np.matmul(A,y_average)))
        k+=1
        for i in range(n):
            if i in set_of_strategies:
                z_do[i]=1/k
            else:
                z_do[i]=0
        iterate_1=1
        x_average=np.array([0 for i in range(n)])
        y_average=np.array([0 for i in range(m)])
        x_do=z_do+0
    xupdate=-mu_x*(np.matmul(A,y))
    xupdate=np.exp(xupdate)
    z1=np.multiply(x_do,xupdate)
    yupdate=mu_y*(np.matmul(x_do,A))
    yupdate=np.exp(yupdate)
    z2=np.multiply(y,yupdate)
    x_do=z1/sum(z1)
    y=z2/sum(z2) 
    iterate+=1
    iterate_1+=1
    y_average=(1-1/(iterate_1))*y_average+1/(iterate_1)*y  
    x_average=(1-1/(iterate_1))*x_average+1/(iterate_1)*x_do 
print(set_of_strategies)
x_axis=list(range(iterate_max))

plt.plot(x_axis, yaxis_mwu, label = "MWU")
plt.plot(x_axis, yaxis_oso, label = "OSO")
plt.plot(x_axis, yaxis_do, label = "DO w/MWU")
  
plt.xlabel('iterations')
plt.ylabel('average loss')
plt.title(GAME_NAME)
plt.legend()
  

SAVE_DIR = os.path.join(os.path.join('results_OMD_update/', PREFIX, GAME_NAME))
os.makedirs(SAVE_DIR)
plt.savefig(os.path.join(SAVE_DIR, 'results_s{:d}.pdf'.format(SEED)))
pickle.dump({'x_axis': x_axis, 'yaxis_mwu': yaxis_mwu, 'yaxis_oso': yaxis_oso, 'yaxis_do': yaxis_do, 'game_name': GAME_NAME},
            open(os.path.join(SAVE_DIR, 'results_s{:d}.p'.format(SEED), 'wb')))   
