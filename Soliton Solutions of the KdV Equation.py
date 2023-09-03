#!/usr/bin/env python
# coding: utf-8

# In[15]:


import math
import numpy as np
import random
import scipy
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import fsolve
from IPython.display import HTML
import matplotlib.animation as animation
import seaborn as sns


# In[16]:


sns.set_style(style="ticks")
sns.set_context('notebook')


# In[17]:


#---start of def---
def initial_cond(x,A,x0,delta,init_type):
    if(init_type==1):
        a = np.cosh((1/delta) *(1/np.sqrt(12)) * np.sqrt(A) * (x - x0))
        return A/(np.multiply(a,a))
    else:
        b=np.sin(2*np.pi*x)
        return b
#---end of def---
#---start of def---
def U1(u0, h, k, delta):
    up1 = np.hstack([u0[1:], u0[:1]])
    up2 = np.hstack([u0[2:], u0[:2]])
    u = np.array(u0)
    um1 = np.hstack([u0[-1:], u0[:-1]])
    um2 = np.hstack([u0[-2:], u0[:-2]])

    a = (up1 - um1) / (2 * h)
    b = (up2 - 2 * up1 + 2 * um1 - um2) / (2 * h * h * h)
    d = (up1 + u + um1) / 3
    
    sol=u0-k*np.multiply(a,d)-delta*delta*k*b
    return sol
#---end of def---
#---start of def---
def U2(u0, u1, h, k, delta):    
    up1 = np.hstack([u1[1:], u1[:1]])
    up2 = np.hstack([u1[2:], u1[:2]])
    up3 = np.hstack([u1[3:], u1[:3]])
    up4 = np.hstack([u1[4:], u1[:4]])
    u = np.array(u1)
    um1 = np.hstack([u1[-1:], u1[:-1]])
    um2 = np.hstack([u1[-2:], u1[:-2]])
    um3 = np.hstack([u1[-3:], u1[:-3]])
    um4 = np.hstack([u1[-4:], u1[:-4]])
    
    a = (up1 - um1) / (2 * h)
    b = (up2 - 2 * up1 + 2 * um1 - um2) / (2 * h * h * h)
    d = (up1 + u + um1) / 3
    
    return u0 - 2 * k * np.multiply(d, a) - 2 * delta*delta * k * b
#---end of def---
#---start of def---
def solver(u0,u1,h,k,delta,steps):
    for i in range(steps-1):
        U = U2(u0,u1,h,k,delta)
        u0=u1
        u1=U
    return u1
#---end of def---
#---start of def---
def calc_k(h,delta,A):
    k=h*h*h/(4*delta*delta+h*h*A)
    return k
#---end of def---
#---start of def---
def analytical_solution(x,t,A,delta,x0):
    #a*(sech(sqrt(a/(delta*delta*12))*(x-c*t-x0)) .* sech(sqrt(a/(delta*delta*12))*(x-c*t-x0)) )
    c=A/3
    a = np.cosh((1/delta) *(1/np.sqrt(12)) * np.sqrt(A) * (x - x0 - c*t))
    return A/(np.multiply(a,a))
#---end of def---
#---start of def---
def visualise(u0,u1,h,k,delta,steps, ylim: tuple = (-1,1.5), xlim = (-8,8), anim_interval = 100):
    
    fig, ax = plt.subplots();
    line, = ax.plot([], [])
    frames_req=steps+1
    def init():
        line.set_data([], [])
    def animate(i):
        global u0
        global u1
        global U
        if(i==0):
            line.set_data(x, u0)
        if(i==1):
            line.set_data(x, u1)
        else:
            for j in range(1):
                U = U2(u0,u1,h,k,delta)
                u0 = u1
                u1 = U
        fig.suptitle("Time t="+str(i*k))
        line.set_data(x, U)

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=frames_req, interval = anim_interval, repeat=False) 

    plt.close()

    return HTML(anim.to_jshtml())
#---end of def---


# ### Question 2

# In[144]:


A=2
x0=0.25
h=0.002
t=0.5
delta=0.03
x=np.arange(-4,4,h)
N=int(np.ceil(16/h))
k=0.5*calc_k(h,delta,A)
M=int(np.ceil(t/k))

initial_cond_param={'A':A,'x0':x0,'delta':delta,'init_type':1}
solution_params_first_step = {'h':h, 'k':k, 'delta':delta}
num_solution_params_final = {'h':h, 'k':k, 'delta':delta,'steps':M}
analytical_solution_params_final={'A':A,'delta':delta,'x0':x0}

u_prev = initial_cond(x, **initial_cond_param)
u_curr = U1(u_prev, **solution_params_first_step)
u_final_num=solver(u_prev,u_curr,**num_solution_params_final)
u_final_anal=analytical_solution(x,t,**analytical_solution_params_final)

print("h="+str(h)+"; k="+str(k))

plt.plot(x,u_final_anal,label='Analytical Solution')
plt.plot(x,u_final_num,'m--',label='Numerical Solution')
plt.title("Time, t="+str(k*2+k*(M-2)))
plt.legend()
plt.ylabel('$u(x,t)$')
plt.xlabel('$x$')
plt.savefig('fig1.eps')
plt.show()

error=abs(u_final_num-u_final_anal)
print('error:',error)
print('max error:', max(error))
print('mean error:', np.mean(error))


# In[156]:


print(max(u_final_num))
print(min(u_final_num))


# In[154]:


for i in range(int(0),int(0.8*len(x))):
    print(x[i], u_final_num[i])


# In[155]:


for i in range(int(0.8*len(x))+1,int(len(x))):
    print(x[i], u_final_num[i])


# In[ ]:





# In[405]:


A=2
x0=0.25
h=0.005
t=0.5
delta=0.03
x=np.arange(-5,5,h)
N=int(np.ceil(16/h))
k=0.5*calc_k(h,delta,A)
M=int(np.ceil(t/k))

initial_cond_param={'A':A,'x0':x0,'delta':delta,'init_type':1}
solution_params_first_step = {'h':h, 'k':k, 'delta':delta}
num_solution_params_final = {'h':h, 'k':k, 'delta':delta,'steps':M}
analytical_solution_params_final={'A':A,'delta':delta,'x0':x0}

u_prev = initial_cond(x, **initial_cond_param)
u_curr = U1(u_prev, **solution_params_first_step)
u_final_num=solver(u_prev,u_curr,**num_solution_params_final)
u_final_anal=analytical_solution(x,t,**analytical_solution_params_final)
print(max(u_final_num))

print(max(u_final_anal))
print(u_final_num[list(u_final_num).index(max(u_final_num))],x[list(u_final_num).index(max(u_final_num))])
print("h="+str(h)+"; k="+str(k))

plt.plot(x,u_final_anal,label='Analytical Solution')
plt.plot(x,u_final_num,'m--',label='Numerical Solution')
plt.axis([0.57, 0.6, 1.8, 2.01])
plt.title("Zoomed in plot, Time, t="+str(k*2+k*(M-2)))
plt.legend()
plt.ylabel('$u(x,t)$')
plt.xlabel('$x$')
plt.savefig('fig2.eps')
plt.show()


# In[162]:


A=2
x0=0.25
h=0.005
t=0.5
delta=0.03
x=np.arange(-5,5,h)
N=int(np.ceil(16/h))
k=0.5*calc_k(h,delta,A)
M=int(np.ceil(t/k))

initial_cond_param={'A':A,'x0':x0,'delta':delta,'init_type':1}
solution_params_first_step = {'h':h, 'k':k, 'delta':delta}
num_solution_params_final = {'h':h, 'k':k, 'delta':delta,'steps':M}
analytical_solution_params_final={'A':A,'delta':delta,'x0':x0}

u_prev = initial_cond(x, **initial_cond_param)
u_curr = U1(u_prev, **solution_params_first_step)
u_final_num=solver(u_prev,u_curr,**num_solution_params_final)
u_final_anal=analytical_solution(x,t,**analytical_solution_params_final)
print(max(u_final_num))

print(max(u_final_anal))
print(u_final_num[list(u_final_num).index(max(u_final_num))],x[list(u_final_num).index(max(u_final_num))])
print("h="+str(h)+"; k="+str(k))

plt.plot(x,u_final_anal,label='Analytical Solution')
plt.plot(x,u_final_num,'m--',label='Numerical Solution')
plt.axis([0.57, 0.6, 1.8, 2.01])
plt.title("Zoomed in plot, Time, t="+str(k*2+k*(M-2)))
plt.legend()
plt.ylabel('$u(x,t)$')
plt.xlabel('$x$')
plt.savefig('fig2.eps')
plt.show()


# In[143]:


A=2
x0=0.25
h=0.007
t=0.5
delta=0.03
x=np.arange(-5,5,h)
N=int(np.ceil(16/h))
k=0.5*calc_k(h,delta,A)
M=int(np.ceil(t/k))

initial_cond_param={'A':A,'x0':x0,'delta':delta,'init_type':1}
solution_params_first_step = {'h':h, 'k':k, 'delta':delta}
num_solution_params_final = {'h':h, 'k':k, 'delta':delta,'steps':M}
analytical_solution_params_final={'A':A,'delta':delta,'x0':x0}

u_prev = initial_cond(x, **initial_cond_param)
u_curr = U1(u_prev, **solution_params_first_step)
u_final_num=solver(u_prev,u_curr,**num_solution_params_final)
u_final_anal=analytical_solution(x,t,**analytical_solution_params_final)

print("h="+str(h)+"; k="+str(k))

plt.plot(x,u_final_anal,label='Analytical Solution')
plt.plot(x,u_final_num,'m--',label='Numerical Solution')
plt.axis([0, 0.3, -0.01, 0.01])
plt.title("Zoomed in plot, Time, t="+str(k*2+k*(M-2)))
plt.legend()
plt.ylabel('$u(x,t)$')
plt.xlabel('$x$')
plt.savefig('fig5.eps')
plt.show()


# In[45]:


A=2
x0=0.25
h=0.005
x=np.arange(-8,8,h)
N=int(np.ceil(16/h))
delta=0.03
k=0.5*calc_k(h,delta,A)

initial_cond_param={'A':A,'x0':x0,'delta':delta,'init_type':1}
solution_params_first_step = {'h':h, 'k':k, 'delta':delta}
analytical_solution_params_final={'A':A,'delta':delta,'x0':x0}

max_error_list=[]
mean_error_list=[]
t_list=list(np.arange(0,2,0.1))
for t in t_list:
    M=int(np.ceil(t/k))
    num_solution_params_final = {'h':h, 'k':k, 'delta':delta,'steps':M}
    u_prev = initial_cond(x, **initial_cond_param)
    u_curr = U1(u_prev, **solution_params_first_step)
    u_final_num=solver(u_prev,u_curr,**num_solution_params_final)
    u_final_anal=analytical_solution(x,t,**analytical_solution_params_final)
    error=abs(u_final_num-u_final_anal)
    max_error_list.append(max(error))
    mean_error_list.append(np.mean(error))


# In[89]:


plt.plot(t_list,max_error_list)
plt.title("Mean Error")
plt.xlabel("t")
plt.xlim(0, 0.5)
plt.ylabel("Mean error")


# In[83]:


plt.plot(t_list,mean_error_list)
plt.title("Mean Error")
plt.xlabel("t")
plt.xlim(0, 0.5)
plt.ylabel("Mean error")


# In[47]:


A=2
x0=0.25
h1=0.01
x=np.arange(-8,8,h1)
N=int(np.ceil(16/h1))
delta=0.03
k1=0.5*calc_k(h1,delta,A)

initial_cond_param={'A':A,'x0':x0,'delta':delta,'init_type':1}
solution_params_first_step = {'h':h1, 'k':k1, 'delta':delta}
analytical_solution_params_final={'A':A,'delta':delta,'x0':x0}

max_error_list_2=[]
mean_error_list_2=[]
t_list=list(np.arange(0,2,0.1))
for t in t_list:
    M=int(np.ceil(t/k1))
    num_solution_params_final = {'h':h1, 'k':k1, 'delta':delta,'steps':M}
    u_prev = initial_cond(x, **initial_cond_param)
    u_curr = U1(u_prev, **solution_params_first_step)
    u_final_num=solver(u_prev,u_curr,**num_solution_params_final)
    u_final_anal=analytical_solution(x,t,**analytical_solution_params_final)
    error=abs(u_final_num-u_final_anal)
    max_error_list_2.append(max(error))
    mean_error_list_2.append(np.mean(error))


# In[48]:


A=2
x0=0.25
h2=0.007
x=np.arange(-8,8,h2)
N=int(np.ceil(16/h2))
delta=0.03
k2=0.5*calc_k(h2,delta,A)

initial_cond_param={'A':A,'x0':x0,'delta':delta,'init_type':1}
solution_params_first_step = {'h':h2, 'k':k2, 'delta':delta}
analytical_solution_params_final={'A':A,'delta':delta,'x0':x0}

max_error_list_3=[]
mean_error_list_3=[]
t_list=list(np.arange(0,2,0.1))
for t in t_list:
    M=int(np.ceil(t/k2))
    num_solution_params_final = {'h':h2, 'k':k2, 'delta':delta,'steps':M}
    u_prev = initial_cond(x, **initial_cond_param)
    u_curr = U1(u_prev, **solution_params_first_step)
    u_final_num=solver(u_prev,u_curr,**num_solution_params_final)
    u_final_anal=analytical_solution(x,t,**analytical_solution_params_final)
    error=abs(u_final_num-u_final_anal)
    max_error_list_3.append(max(error))
    mean_error_list_3.append(np.mean(error))


# In[109]:


plt.plot(t_list,max_error_list,'b',label=str(h))
plt.plot(t_list,max_error_list_2,'r',label=str(h1))
plt.plot(t_list,max_error_list_3,'g',label=str(h2))
plt.title("Maximum Error")
plt.xlabel("t")
plt.xlim(0, 0.5)
plt.ylim(0, 0.035)
plt.ylabel("Max error")
plt.legend()
plt.savefig('fig4.eps')
plt.show()


# In[112]:


print(np.interp(0.5,t_list,mean_error_list))
print(np.interp(0.5,t_list,max_error_list))


# In[107]:


plt.plot(t_list,mean_error_list,'b',label=str(h))
plt.plot(t_list,mean_error_list_2,'r',label=str(h1))
plt.plot(t_list,mean_error_list_3,'g',label=str(h2))
plt.title("Mean Error")
plt.xlabel("t")
plt.xlim(0, 0.5)
plt.ylim(0, 0.0005)
plt.ylabel("Mean error")
plt.legend()
plt.savefig('fig3.eps')
plt.show()


# Quite clear that: $$\text{Numerical solution's propagation speed} \geq \text{Exact solution's propagation speed}$$

# ### Question 3

# In[180]:


A1=2
x01=0.25
A2=1
x02=0.75
h=0.005
x=np.arange(-8,8,h)
N=int(np.ceil(16/h))
delta=0.03
k=min(0.5*calc_k(h,delta,A1), 0.5*calc_k(h,delta,A2))
initial_cond_param_1={'A':A1,'x0':x01,'delta':delta,'init_type':1}
initial_cond_param_2={'A':A2,'x0':x02,'delta':delta,'init_type':1}
solution_params_first_step = {'h':h, 'k':k, 'delta':delta}

print("h="+str(h)+"; k="+str(k))

t_list=[6]

for t in t_list:
    M=int(np.ceil(t/k))
    num_solution_params_final = {'h':h, 'k':k, 'delta':delta,'steps':M}

    u_prev = initial_cond(x, **initial_cond_param_1)+initial_cond(x, **initial_cond_param_2)
    u_curr = U1(u_prev, **solution_params_first_step)
    u_final_num=solver(u_prev,u_curr,**num_solution_params_final)
    
    u_prev_1 = initial_cond(x, **initial_cond_param_1)
    u_curr_1 = U1(u_prev_1, **solution_params_first_step)
    u_final_num_1=solver(u_prev_1,u_curr_1,**num_solution_params_final)
    
    u_prev_2 = initial_cond(x, **initial_cond_param_2)
    u_curr_2 = U1(u_prev_2, **solution_params_first_step)
    u_final_num_2=solver(u_prev_2,u_curr_2,**num_solution_params_final)

    plt.plot(x,u_final_num,'r',label='Combined Initial Condition')
    plt.plot(x,u_final_num_1,'b--',label='Initial Condition with $A=2$, $x_0=0.25$')
    plt.plot(x,u_final_num_2,'g--',label='Initial Condition with $A=1$, $x_0=0.75$')
    plt.title("Time, t="+str(t))
    plt.ylabel('$u(x,t)$')
    plt.xlabel('$x$')
    plt.legend()
    plt.savefig('figq4big.eps')
    plt.show()

    Mass=np.trapz(u_final_num, x)
    E=np.trapz(0.5*np.multiply(u_final_num,u_final_num),x)

    print("Mass, M = "+str(Mass))
    print("Energy, E = "+str(E))


# In[170]:


A1=2
x01=0.25
A2=1
x02=0.75
h=0.005
x=np.arange(-8,8,h)
N=int(np.ceil(16/h))
delta=0.03
k=min(0.5*calc_k(h,delta,A1), 0.5*calc_k(h,delta,A2))
initial_cond_param_1={'A':A1,'x0':x01,'delta':delta,'init_type':1}
initial_cond_param_2={'A':A2,'x0':x02,'delta':delta,'init_type':1}
solution_params_first_step = {'h':h, 'k':k, 'delta':delta}

print("h="+str(h)+"; k="+str(k))

t_list=[0, 0.1, 0.5, 1, 2, 3, 6, 10]

for t in t_list:
    M=int(np.ceil(t/k))
    num_solution_params_final = {'h':h, 'k':k, 'delta':delta,'steps':M}

    u_prev = initial_cond(x, **initial_cond_param_1)+initial_cond(x, **initial_cond_param_2)
    u_curr = U1(u_prev, **solution_params_first_step)
    u_final_num=solver(u_prev,u_curr,**num_solution_params_final)

    plt.plot(x,u_final_num,'r',label='Numerical Solution')
    plt.title("Time, t="+str(t))
    plt.ylabel('$u(x,t)$')
    plt.xlabel('$x$')
    plt.savefig('figq4t'+str(t)+'.eps')
    plt.show()

    Mass=np.trapz(u_final_num, x)
    E=np.trapz(0.5*np.multiply(u_final_num,u_final_num),x)

    print("Mass, M = "+str(Mass))
    print("Energy, E = "+str(E))


# In[181]:


A1=2
x01=0.25
A2=1
x02=0.75
h=0.005
x=np.arange(-8,8,h)
N=int(np.ceil(16/h))
delta=0.03
k=min(0.5*calc_k(h,delta,A1), 0.5*calc_k(h,delta,A2))
initial_cond_param_1={'A':A1,'x0':x01,'delta':delta,'init_type':1}
initial_cond_param_2={'A':A2,'x0':x02,'delta':delta,'init_type':1}
solution_params_first_step = {'h':h, 'k':k, 'delta':delta}

print("h="+str(h)+"; k="+str(k))

t_list=list(np.arange(0,10,0.5))

M_list_1=[]
E_list_1=[]
for t in t_list:
    M=int(np.ceil(t/k))
    num_solution_params_final = {'h':h, 'k':k, 'delta':delta,'steps':M}


    u_prev = initial_cond(x, **initial_cond_param_1)
    u_curr = U1(u_prev, **solution_params_first_step)
    u_final_num=solver(u_prev,u_curr,**num_solution_params_final)

    Mass=np.trapz(u_final_num, x)
    E=np.trapz(0.5*np.multiply(u_final_num,u_final_num),x)
    E_list_1.append(E)
    M_list_1.append(Mass)

for i in range(len(t_list)):
    print(t_list[i], M_list_1[i], E_list_1[i])


# In[182]:


A1=2
x01=0.25
A2=1
x02=0.75
h=0.005
x=np.arange(-8,8,h)
N=int(np.ceil(16/h))
delta=0.03
k=min(0.5*calc_k(h,delta,A1), 0.5*calc_k(h,delta,A2))
initial_cond_param_1={'A':A1,'x0':x01,'delta':delta,'init_type':1}
initial_cond_param_2={'A':A2,'x0':x02,'delta':delta,'init_type':1}
solution_params_first_step = {'h':h, 'k':k, 'delta':delta}

print("h="+str(h)+"; k="+str(k))

t_list=list(np.arange(0,10,0.5))

M_list_2=[]
E_list_2=[]
for t in t_list:
    M=int(np.ceil(t/k))
    num_solution_params_final = {'h':h, 'k':k, 'delta':delta,'steps':M}


    u_prev = initial_cond(x, **initial_cond_param_2)
    u_curr = U1(u_prev, **solution_params_first_step)
    u_final_num=solver(u_prev,u_curr,**num_solution_params_final)

    Mass=np.trapz(u_final_num, x)
    E=np.trapz(0.5*np.multiply(u_final_num,u_final_num),x)
    E_list_2.append(E)
    M_list_2.append(Mass)

for i in range(len(t_list)):
    print(t_list[i], M_list_2[i], E_list_2[i])


# In[185]:


for i in range(len(t_list)): 
    print('$'+str(t_list[i])+'$ & $'+str(M_list_1[i]+M_list_2[i]-M_list[i]) +'$ & $'+str( E_list_1[i]+E_list_2[i]-E_list[i])+'$ \\ \hline')


# In[169]:


A1=2
x01=0.25
A2=1
x02=0.75
h=0.005
x=np.arange(-8,8,h)
N=int(np.ceil(16/h))
delta=0.03
k=min(0.5*calc_k(h,delta,A1), 0.5*calc_k(h,delta,A2))
initial_cond_param_1={'A':A1,'x0':x01,'delta':delta,'init_type':1}
initial_cond_param_2={'A':A2,'x0':x02,'delta':delta,'init_type':1}
solution_params_first_step = {'h':h, 'k':k, 'delta':delta}

print("h="+str(h)+"; k="+str(k))

t_list=list(np.arange(0,10,0.5))

M_list=[]
E_list=[]
for t in t_list:
    M=int(np.ceil(t/k))
    num_solution_params_final = {'h':h, 'k':k, 'delta':delta,'steps':M}


    u_prev = initial_cond(x, **initial_cond_param_1)+initial_cond(x, **initial_cond_param_2)
    u_curr = U1(u_prev, **solution_params_first_step)
    u_final_num=solver(u_prev,u_curr,**num_solution_params_final)

    Mass=np.trapz(u_final_num, x)
    E=np.trapz(0.5*np.multiply(u_final_num,u_final_num),x)
    E_list.append(E)
    M_list.append(Mass)

for i in range(len(t_list)):
    print(t_list[i], M_list[i], E_list[i])


# ### Question 4

# In[198]:


A=2
x0=0.25
h=0.005
t=0.7*1/(2*np.pi)
x=np.arange(0,1,h)
N=int(np.ceil(16/h))
delta=0
k=0.5*calc_k(h,delta,A)
M=int(np.ceil(t/k))

initial_cond_param={'A':A,'x0':x0,'delta':delta,'init_type':2}
solution_params_first_step = {'h':h, 'k':k, 'delta':delta}
num_solution_params_final = {'h':h, 'k':k, 'delta':delta,'steps':M}
analytical_solution_params_final={'A':A,'delta':delta,'x0':x0}

u_prev = initial_cond(x, **initial_cond_param)
u_curr = U1(u_prev, **solution_params_first_step)
u_final_num=solver(u_prev,u_curr,**num_solution_params_final)

print("h="+str(h)+"; k="+str(k))

plt.plot(x,u_final_num,'r')
plt.title("Time, t="+str('%.5f'%(k*2+k*(M-2))))
plt.ylabel('$u(x,t)$')
plt.xlabel('$x$')
plt.savefig('figq5i.eps')
plt.show()


# In[212]:


A=2
x0=0.25
h=0.005
t=1/(2*np.pi)
x=np.arange(0,1,h)
N=int(np.ceil(16/h))
delta=0
k=0.5*calc_k(h,delta,A)
M=int(np.ceil(t/k))

initial_cond_param={'A':A,'x0':x0,'delta':delta,'init_type':2}
solution_params_first_step = {'h':h, 'k':k, 'delta':delta}
num_solution_params_final = {'h':h, 'k':k, 'delta':delta,'steps':M}
analytical_solution_params_final={'A':A,'delta':delta,'x0':x0}

u_prev = initial_cond(x, **initial_cond_param)
u_curr = U1(u_prev, **solution_params_first_step)
u_final_num=solver(u_prev,u_curr,**num_solution_params_final)

print("h="+str(h)+"; k="+str(k))

plt.plot(x,u_final_num,'r')
plt.title("Time, $t=1/2\pi$")
plt.ylabel('$u(x,t)$')
plt.xlabel('$x$')
plt.savefig('figq5ii.eps')
plt.show()


# In[245]:


A=2
x0=0.25
h=0.005
t=0
x=np.arange(0,1,h)
N=int(np.ceil(16/h))
delta=0.03
k=0.5*calc_k(h,delta,A)
M=int(np.ceil(t/k))

initial_cond_param={'A':A,'x0':x0,'delta':delta,'init_type':2}
solution_params_first_step = {'h':h, 'k':k, 'delta':delta}
num_solution_params_final = {'h':h, 'k':k, 'delta':delta,'steps':M}
analytical_solution_params_final={'A':A,'delta':delta,'x0':x0}

u_prev = initial_cond(x, **initial_cond_param)
u_curr = U1(u_prev, **solution_params_first_step)
u_final_num=solver(u_prev,u_curr,**num_solution_params_final)

print("h="+str(h)+"; k="+str(k))

plt.plot(x,u_final_num,'r')
plt.title("Time, t="+str('%.5f'%(k*2+k*(M-2))))
plt.ylabel('$u(x,t)$')
plt.xlabel('$x$')
plt.savefig('figq5ai.eps')
plt.show()


# In[358]:


A=2
x0=0.25
h=0.005
t=0.101
x=np.arange(0,1,h)
N=int(np.ceil(16/h))
delta=0.03
k=0.5*calc_k(h,delta,A)
M=int(np.ceil(t/k))

initial_cond_param={'A':A,'x0':x0,'delta':delta,'init_type':2}
solution_params_first_step = {'h':h, 'k':k, 'delta':delta}
num_solution_params_final = {'h':h, 'k':k, 'delta':delta,'steps':M}
analytical_solution_params_final={'A':A,'delta':delta,'x0':x0}

u_prev = initial_cond(x, **initial_cond_param)
u_curr = U1(u_prev, **solution_params_first_step)
u_final_num=solver(u_prev,u_curr,**num_solution_params_final)

print("h="+str(h)+"; k="+str(k))

plt.plot(x,u_final_num,'r')
plt.title("Time, t="+str('%.5f'%(k*2+k*(M-2))))
plt.ylabel('$u(x,t)$')
plt.xlabel('$x$')
plt.savefig('figq5aii.eps')
plt.show()


# In[247]:


A=2
x0=0.25
h=0.005
t=0.16
x=np.arange(0,1,h)
N=int(np.ceil(16/h))
delta=0.03
k=0.5*calc_k(h,delta,A)
M=int(np.ceil(t/k))

initial_cond_param={'A':A,'x0':x0,'delta':delta,'init_type':2}
solution_params_first_step = {'h':h, 'k':k, 'delta':delta}
num_solution_params_final = {'h':h, 'k':k, 'delta':delta,'steps':M}
analytical_solution_params_final={'A':A,'delta':delta,'x0':x0}

u_prev = initial_cond(x, **initial_cond_param)
u_curr = U1(u_prev, **solution_params_first_step)
u_final_num=solver(u_prev,u_curr,**num_solution_params_final)

print("h="+str(h)+"; k="+str(k))

plt.plot(x,u_final_num,'r')
plt.title("Time, t="+str('%.5f'%(k*2+k*(M-2))))
plt.ylabel('$u(x,t)$')
plt.xlabel('$x$')
plt.savefig('figq5aiii.eps')
plt.show()


# In[248]:


A=2
x0=0.25
h=0.005
t=0.25
x=np.arange(0,1,h)
N=int(np.ceil(16/h))
delta=0.03
k=0.5*calc_k(h,delta,A)
M=int(np.ceil(t/k))

initial_cond_param={'A':A,'x0':x0,'delta':delta,'init_type':2}
solution_params_first_step = {'h':h, 'k':k, 'delta':delta}
num_solution_params_final = {'h':h, 'k':k, 'delta':delta,'steps':M}
analytical_solution_params_final={'A':A,'delta':delta,'x0':x0}

u_prev = initial_cond(x, **initial_cond_param)
u_curr = U1(u_prev, **solution_params_first_step)
u_final_num=solver(u_prev,u_curr,**num_solution_params_final)

print("h="+str(h)+"; k="+str(k))

plt.plot(x,u_final_num,'r')
plt.title("Time, t="+str('%.5f'%(k*2+k*(M-2))))
plt.ylabel('$u(x,t)$')
plt.xlabel('$x$')
plt.savefig('figq5aiv.eps')
plt.show()


# In[389]:


A=2
x0=0.25
h=0.005
t=0.335
x=np.arange(0,1,h)
N=int(np.ceil(16/h))
delta=0.03
k=0.5*calc_k(h,delta,A)
M=int(np.ceil(t/k))

initial_cond_param={'A':A,'x0':x0,'delta':delta,'init_type':2}
solution_params_first_step = {'h':h, 'k':k, 'delta':delta}
num_solution_params_final = {'h':h, 'k':k, 'delta':delta,'steps':M}
analytical_solution_params_final={'A':A,'delta':delta,'x0':x0}

u_prev = initial_cond(x, **initial_cond_param)
u_curr = U1(u_prev, **solution_params_first_step)
u_final_num=solver(u_prev,u_curr,**num_solution_params_final)

print("h="+str(h)+"; k="+str(k))
print(max(u_final_num))
plt.plot(x,u_final_num,'r')
plt.title("Time, t="+str('%.5f'%(k*2+k*(M-2))))
plt.ylabel('$u(x,t)$')
plt.xlabel('$x$')
plt.savefig('figq5av.eps')
plt.show()


# In[392]:


A=2
x0=0.25
h=0.005
t=0.70
x=np.arange(0,1,h)
N=int(np.ceil(16/h))
delta=0.03
k=0.5*calc_k(h,delta,A)
M=int(np.ceil(t/k))

initial_cond_param={'A':A,'x0':x0,'delta':delta,'init_type':2}
solution_params_first_step = {'h':h, 'k':k, 'delta':delta}
num_solution_params_final = {'h':h, 'k':k, 'delta':delta,'steps':M}
analytical_solution_params_final={'A':A,'delta':delta,'x0':x0}

u_prev = initial_cond(x, **initial_cond_param)
u_curr = U1(u_prev, **solution_params_first_step)
u_final_num=solver(u_prev,u_curr,**num_solution_params_final)

print("h="+str(h)+"; k="+str(k))

plt.plot(x,u_final_num,'r')
plt.title("Time, t="+str('%.5f'%(k*2+k*(M-2))))
plt.ylabel('$u(x,t)$')
plt.xlabel('$x$')
plt.savefig('figq5avi.eps')
plt.show()


# In[307]:


A=2
x0=0.25
h=0.005
t=0.1
x=np.arange(0,1,h)
N=int(np.ceil(16/h))
delta=0.01
k=0.5*calc_k(h,delta,A)
M=int(np.ceil(t/k))

initial_cond_param={'A':A,'x0':x0,'delta':delta,'init_type':2}
solution_params_first_step = {'h':h, 'k':k, 'delta':delta}
num_solution_params_final = {'h':h, 'k':k, 'delta':delta,'steps':M}
analytical_solution_params_final={'A':A,'delta':delta,'x0':x0}

u_prev = initial_cond(x, **initial_cond_param)
u_curr = U1(u_prev, **solution_params_first_step)
u_final_num=solver(u_prev,u_curr,**num_solution_params_final)

print("h="+str(h)+"; k="+str(k))
print(max(u_final_num))
plt.plot(x,u_final_num,'r')
plt.title("Time, t="+str('%.5f'%(k*2+k*(M-2))))
plt.ylabel('$u(x,t)$')
plt.xlabel('$x$')
plt.savefig('figq5bi.eps')
plt.show()


# In[364]:


A=2
x0=0.25
h=0.005
t=0.1235
x=np.arange(0,1,h)
N=int(np.ceil(16/h))
delta=0.01
k=0.5*calc_k(h,delta,A)
M=int(np.ceil(t/k))

initial_cond_param={'A':A,'x0':x0,'delta':delta,'init_type':2}
solution_params_first_step = {'h':h, 'k':k, 'delta':delta}
num_solution_params_final = {'h':h, 'k':k, 'delta':delta,'steps':M}
analytical_solution_params_final={'A':A,'delta':delta,'x0':x0}

u_prev = initial_cond(x, **initial_cond_param)
u_curr = U1(u_prev, **solution_params_first_step)
u_final_num=solver(u_prev,u_curr,**num_solution_params_final)

print("h="+str(h)+"; k="+str(k))
print(max(u_final_num))
plt.plot(x,u_final_num,'r')
plt.title("Time, t="+str('%.5f'%(k*2+k*(M-2))))
plt.ylabel('$u(x,t)$')
plt.xlabel('$x$')
plt.savefig('figq5bii.eps')
plt.show()


# In[309]:


A=2
x0=0.25
h=0.005
t=0.2
x=np.arange(0,1,h)
N=int(np.ceil(16/h))
delta=0.01
k=0.5*calc_k(h,delta,A)
M=int(np.ceil(t/k))

initial_cond_param={'A':A,'x0':x0,'delta':delta,'init_type':2}
solution_params_first_step = {'h':h, 'k':k, 'delta':delta}
num_solution_params_final = {'h':h, 'k':k, 'delta':delta,'steps':M}
analytical_solution_params_final={'A':A,'delta':delta,'x0':x0}

u_prev = initial_cond(x, **initial_cond_param)
u_curr = U1(u_prev, **solution_params_first_step)
u_final_num=solver(u_prev,u_curr,**num_solution_params_final)

print("h="+str(h)+"; k="+str(k))
print(max(u_final_num))
plt.plot(x,u_final_num,'r')
plt.title("Time, t="+str('%.5f'%(k*2+k*(M-2))))
plt.ylabel('$u(x,t)$')
plt.xlabel('$x$')
plt.savefig('figq5biii.eps')
plt.show()


# In[310]:


A=2
x0=0.25
h=0.005
t=0.2995
x=np.arange(0,1,h)
N=int(np.ceil(16/h))
delta=0.01
k=0.5*calc_k(h,delta,A)
M=int(np.ceil(t/k))

initial_cond_param={'A':A,'x0':x0,'delta':delta,'init_type':2}
solution_params_first_step = {'h':h, 'k':k, 'delta':delta}
num_solution_params_final = {'h':h, 'k':k, 'delta':delta,'steps':M}
analytical_solution_params_final={'A':A,'delta':delta,'x0':x0}

u_prev = initial_cond(x, **initial_cond_param)
u_curr = U1(u_prev, **solution_params_first_step)
u_final_num=solver(u_prev,u_curr,**num_solution_params_final)

print("h="+str(h)+"; k="+str(k))
print(max(u_final_num))
plt.plot(x,u_final_num,'r')
plt.title("Time, t="+str('%.5f'%(k*2+k*(M-2))))
plt.ylabel('$u(x,t)$')
plt.xlabel('$x$')
plt.savefig('figq5biv.eps')
plt.show()


# In[376]:


A=2
x0=0.25
h=0.005
t=0.5
x=np.arange(0,1,h)
N=int(np.ceil(16/h))
delta=0.01
k=0.5*calc_k(h,delta,A)
M=int(np.ceil(t/k))

initial_cond_param={'A':A,'x0':x0,'delta':delta,'init_type':2}
solution_params_first_step = {'h':h, 'k':k, 'delta':delta}
num_solution_params_final = {'h':h, 'k':k, 'delta':delta,'steps':M}
analytical_solution_params_final={'A':A,'delta':delta,'x0':x0}

u_prev = initial_cond(x, **initial_cond_param)
u_curr = U1(u_prev, **solution_params_first_step)
u_final_num=solver(u_prev,u_curr,**num_solution_params_final)

print("h="+str(h)+"; k="+str(k))
print(max(u_final_num))
plt.plot(x,u_final_num,'r')
plt.title("Time, t="+str('%.5f'%(k*2+k*(M-2))))
plt.ylabel('$u(x,t)$')
plt.xlabel('$x$')
plt.savefig('figq5bv.eps')
plt.show()


# In[383]:


A=2
x0=0.25
h=0.005
t=0.72
x=np.arange(0,1,h)
N=int(np.ceil(16/h))
delta=0.01
k=0.5*calc_k(h,delta,A)
M=int(np.ceil(t/k))

initial_cond_param={'A':A,'x0':x0,'delta':delta,'init_type':2}
solution_params_first_step = {'h':h, 'k':k, 'delta':delta}
num_solution_params_final = {'h':h, 'k':k, 'delta':delta,'steps':M}
analytical_solution_params_final={'A':A,'delta':delta,'x0':x0}

u_prev = initial_cond(x, **initial_cond_param)
u_curr = U1(u_prev, **solution_params_first_step)
u_final_num=solver(u_prev,u_curr,**num_solution_params_final)

print("h="+str(h)+"; k="+str(k))
print(max(u_final_num))
plt.plot(x,u_final_num,'r')
plt.title("Time, t="+str('%.5f'%(k*2+k*(M-2))))
plt.ylabel('$u(x,t)$')
plt.xlabel('$x$')
plt.savefig('figq5bvi.eps')
plt.show()


# In[322]:


A=2
x0=0.25
h=0.005
t=0.05
x=np.arange(0,1,h)
N=int(np.ceil(16/h))
delta=0.05
k=0.5*calc_k(h,delta,A)
M=int(np.ceil(t/k))

initial_cond_param={'A':A,'x0':x0,'delta':delta,'init_type':2}
solution_params_first_step = {'h':h, 'k':k, 'delta':delta}
num_solution_params_final = {'h':h, 'k':k, 'delta':delta,'steps':M}
analytical_solution_params_final={'A':A,'delta':delta,'x0':x0}

u_prev = initial_cond(x, **initial_cond_param)
u_curr = U1(u_prev, **solution_params_first_step)
u_final_num=solver(u_prev,u_curr,**num_solution_params_final)

print("h="+str(h)+"; k="+str(k))
print(max(u_final_num))
plt.plot(x,u_final_num,'r')
plt.title("Time, t="+str('%.5f'%(k*2+k*(M-2))))
plt.ylabel('$u(x,t)$')
plt.xlabel('$x$')
plt.savefig('figq5ci.eps')
plt.show()


# In[398]:


A=2
x0=0.25
h=0.005
t=0.0905
x=np.arange(0,1,h)
N=int(np.ceil(16/h))
delta=0.05
k=0.5*calc_k(h,delta,A)
M=int(np.ceil(t/k))

initial_cond_param={'A':A,'x0':x0,'delta':delta,'init_type':2}
solution_params_first_step = {'h':h, 'k':k, 'delta':delta}
num_solution_params_final = {'h':h, 'k':k, 'delta':delta,'steps':M}
analytical_solution_params_final={'A':A,'delta':delta,'x0':x0}

u_prev = initial_cond(x, **initial_cond_param)
u_curr = U1(u_prev, **solution_params_first_step)
u_final_num=solver(u_prev,u_curr,**num_solution_params_final)

print("h="+str(h)+"; k="+str(k))
print(max(u_final_num))
plt.plot(x,u_final_num,'r')
plt.title("Time, t="+str('%.5f'%(k*2+k*(M-2))))
plt.ylabel('$u(x,t)$')
plt.xlabel('$x$')
plt.savefig('figq5cii.eps')
plt.show()


# In[340]:


A=2
x0=0.25
h=0.005
t=0.2
x=np.arange(0,1,h)
N=int(np.ceil(16/h))
delta=0.05
k=0.5*calc_k(h,delta,A)
M=int(np.ceil(t/k))

initial_cond_param={'A':A,'x0':x0,'delta':delta,'init_type':2}
solution_params_first_step = {'h':h, 'k':k, 'delta':delta}
num_solution_params_final = {'h':h, 'k':k, 'delta':delta,'steps':M}
analytical_solution_params_final={'A':A,'delta':delta,'x0':x0}

u_prev = initial_cond(x, **initial_cond_param)
u_curr = U1(u_prev, **solution_params_first_step)
u_final_num=solver(u_prev,u_curr,**num_solution_params_final)

print("h="+str(h)+"; k="+str(k))
print(max(u_final_num))
plt.plot(x,u_final_num,'r')
plt.title("Time, t="+str('%.5f'%(k*2+k*(M-2))))
plt.ylabel('$u(x,t)$')
plt.xlabel('$x$')
plt.savefig('figq5ciii.eps')
plt.show()


# In[344]:


A=2
x0=0.25
h=0.005
t=0.351
x=np.arange(0,1,h)
N=int(np.ceil(16/h))
delta=0.05
k=0.5*calc_k(h,delta,A)
M=int(np.ceil(t/k))

initial_cond_param={'A':A,'x0':x0,'delta':delta,'init_type':2}
solution_params_first_step = {'h':h, 'k':k, 'delta':delta}
num_solution_params_final = {'h':h, 'k':k, 'delta':delta,'steps':M}
analytical_solution_params_final={'A':A,'delta':delta,'x0':x0}

u_prev = initial_cond(x, **initial_cond_param)
u_curr = U1(u_prev, **solution_params_first_step)
u_final_num=solver(u_prev,u_curr,**num_solution_params_final)

print("h="+str(h)+"; k="+str(k))
print(max(u_final_num))
plt.plot(x,u_final_num,'r')
plt.title("Time, t="+str('%.5f'%(k*2+k*(M-2))))
plt.ylabel('$u(x,t)$')
plt.xlabel('$x$')
plt.savefig('figq5civ.eps')
plt.show()


# In[345]:


A=2
x0=0.25
h=0.005
t=0.5
x=np.arange(0,1,h)
N=int(np.ceil(16/h))
delta=0.05
k=0.5*calc_k(h,delta,A)
M=int(np.ceil(t/k))

initial_cond_param={'A':A,'x0':x0,'delta':delta,'init_type':2}
solution_params_first_step = {'h':h, 'k':k, 'delta':delta}
num_solution_params_final = {'h':h, 'k':k, 'delta':delta,'steps':M}
analytical_solution_params_final={'A':A,'delta':delta,'x0':x0}

u_prev = initial_cond(x, **initial_cond_param)
u_curr = U1(u_prev, **solution_params_first_step)
u_final_num=solver(u_prev,u_curr,**num_solution_params_final)

print("h="+str(h)+"; k="+str(k))
print(max(u_final_num))
plt.plot(x,u_final_num,'r')
plt.title("Time, t="+str('%.5f'%(k*2+k*(M-2))))
plt.ylabel('$u(x,t)$')
plt.xlabel('$x$')
plt.savefig('figq5cv.eps')
plt.show()


# In[404]:


A=2
x0=0.25
h=0.005
t=1.3
x=np.arange(0,1,h)
N=int(np.ceil(16/h))
delta=0.05
k=0.5*calc_k(h,delta,A)
M=int(np.ceil(t/k))

initial_cond_param={'A':A,'x0':x0,'delta':delta,'init_type':2}
solution_params_first_step = {'h':h, 'k':k, 'delta':delta}
num_solution_params_final = {'h':h, 'k':k, 'delta':delta,'steps':M}
analytical_solution_params_final={'A':A,'delta':delta,'x0':x0}

u_prev = initial_cond(x, **initial_cond_param)
u_curr = U1(u_prev, **solution_params_first_step)
u_final_num=solver(u_prev,u_curr,**num_solution_params_final)

print("h="+str(h)+"; k="+str(k))
print(max(u_final_num))
plt.plot(x,u_final_num,'r')
plt.title("Time, t="+str('%.5f'%(k*2+k*(M-2))))
plt.ylabel('$u(x,t)$')
plt.xlabel('$x$')
plt.savefig('figq5cvi.eps')
plt.show()


# In[ ]:




