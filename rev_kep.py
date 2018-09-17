# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 17:49:21 2013

@author: Amit Vishwas
"""
import numpy as np
from matplotlib import pyplot as plt

#from mpl_toolkits.mplot3d import Axes3D
#import mpl_toolkits.mplot3d.art3d as art3d

from matplotlib.patches import Ellipse
from matplotlib import animation

import matplotlib.gridspec as gridspec

# Font to be used for labels on the plot

font = {'size'   : 9}
plt.rc('font', **font)

# Setup figure and subplots

# Size, dpi and Title for Plot window
#fig = plt.figure(num = 'Orbit Simulator', figsize = (12,8.5),dpi = 100) 
fig = plt.figure(num = 'Orbit Simulator', figsize = (9.5,6.75),dpi = 100) 

# Divide in 3x3 grid, set area to be used on the plot
gs = gridspec.GridSpec(3, 3)
gs.update(left=0.07, right=0.95, wspace=0.15)

#ax = fig.add_subplot(gs[0,:-1], aspect ='equal', projection = '3d') # Maybe use to implement 3D view

# Define the main subplot where orbits are shown
ax = fig.add_subplot(gs[0:,:-1], aspect = 'equal')
ax.set_ylabel('Distance (in AU)')
plt.setp(ax.get_xticklabels(), visible=False) # Set xaxis tick labels to be invisible
ax.text(0.01, 0.01, 'As seen by Observer',
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax.transAxes,
        color='Black', fontsize=12)

# Define the subplot where the velocity profile is shown
ax2 = fig.add_subplot(gs[:-1,-1], aspect = 'auto')
ax2.set_xlabel('Time (in Years)')
ax2.yaxis.tick_right()
ax2.set_ylabel('Velocity (in km/s)')
ax2.locator_params(nbins=6) # limit number of x-ticks

# Define subplot where the Top view of the orbit is shown
ax3 = fig.add_subplot(gs[0,0], aspect = 'equal')
ax3.xaxis.set_visible(False)
ax3.yaxis.set_visible(False)
ax3.text(0.1, 0.99, 'Orbit Top view',
        verticalalignment='top', horizontalalignment='left',
        transform=ax.transAxes,
        color='Black', fontsize=12)

pause = True # Click to pause functionality
change = False

# Initialize global variables - orbital elements

phase = 0.0 # Angle in the orbit, measured with respect to periastron
timer = 0.0 # Time Counter

 
comx = 0. # Center of Mass co-ordinates
comy = 0.

m1 = 3.0;                       # Mass of Obj 1, in Solar mass units
m2 = 1.0;                       # Mass of Obj 2, in Solar mass units

semi_a = 1.0                    # Semi major axis for the orbit, in AU

ecc = 0.3                       # Eccentricity

alpha = semi_a*(1-ecc**2)

nodeangle = 0.                  # Node angle for the orbit

inclination = np.pi/2           # Inclination of the orbit

mu = m1*m2/(m1+m2);             # Reduced Mass

semi_b = semi_a*(1-ecc**2)**0.5 # Semi-minor Axis

L = np.sqrt(mu*semi_a*(1-ecc**2)) # Orbital angula rmomentum : constant for a given orbit

P = ((1/(m1+m2))*semi_a**3)**0.5    #  Period of the orbit, in years

tarray = np.zeros(721) # Placeholder to store conversion  between time step "i" to phase in orbit
xt = np.zeros(721) # Placeholder to store conversion between time step "i" to actual time units in years

xt[:]= [(2*P/720)*x for x in range(721)]  

for i in range(721):
    tht = np.radians(phase)    
    tarray[i] = tht
    phase += np.absolute((1 + ecc*np.cos(tht))**2 / (1 - ecc**2)**1.5)
    phase %= 360

phase = 0.

##################### Show Orbiting Bodies & corresponding orbits

M1 = plt.Circle((0, 0), 0.03, fc='r', clip_on=True, lw = 0); # Draw a circle to represent Body 1
M2 = plt.Circle((0, 0), 0.03, fc='b', clip_on=True, lw = 0); # Draw a circle to represent Body 2

# Try to draw the orbit that the objects will follow
orb1, = ax.plot(0,0,'r-', alpha = 0.33, visible = False) # empty place holder graphs for orbits
orb2, = ax.plot(0,0,'b-', alpha = 0.33, visible = False)

############ Previous attempts for orbits ####
#Ellipse(xy=(-semi_a*(ecc)*(mu/m1), 0), width=2*semi_a*(mu/m1), height=2*semi_b*(mu/m1)*np.cos(inclination), 
#                        edgecolor='r', fc='None', alpha = 0.33, lw=1)
#Ellipse(xy=(semi_a*(ecc)*(mu/m2), 0), width=2*semi_a*(mu/m2), height=2*semi_b*(mu/m2)*np.cos(inclination), 
#                        edgecolor='b', fc='None', alpha = 0.33, lw=1)
##############################################
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.grid(True)

############ Show Velocity of corrresponding bodies with respect to time

# Draw circles for showing the instantaneous velocity of the body on the velocity - time graph
Mv1 = plt.Circle((0, 0), 0.05*P, fc = 'r', ec='r', clip_on=True, lw = 1); 
Mv2 = plt.Circle((0, 0), 0.05*P, fc = 'b', ec='b', clip_on=True, lw = 1);

# 29.87 is velocity of Earth around the Sun, scaled using a^3/(M_sol) = P^2
d3 = 29.87*np.sqrt((m1+m2/alpha))*(1+ecc) 
d4 = 29.87*np.sqrt((m1+m2/alpha))* ecc

d6 = np.sin(inclination)*np.sqrt((d4**2 + d3**2)) # used to define the axis limit on velocity plot, some number larger than either

v1, = ax2.plot(0,0,'r-', visible = False) # empty place holder graphs for velocity curves
v2, = ax2.plot(0,0,'b-', visible = False)
ax2.set_xlim(0, 2*P) # Plot velocity for two prbits
ax2.set_ylim(-d6-0.1, d6+0.1)
ax2.grid(True)
#ax.get_xaxis().set_animated(True)  # enabling it takes away the labels

############### Preffered view of the orbits - from the top, no effect of inclination ####

Mi1 = plt.Circle((0, 0), 0.05, fc='r', clip_on=True, lw = 0);
Mi2 = plt.Circle((0, 0), 0.05, fc='b', clip_on=True, lw = 0); 

# Draw orbits as elipses
orbi1 = Ellipse(xy=(-semi_a*(ecc)*(mu/m1), 0), width=2*semi_a*(mu/m1), height=2*semi_b*(mu/m1), 
                        edgecolor='r', fc='None', lw=0.5)
orbi2 = Ellipse(xy=(semi_a*(ecc)*(mu/m2), 0), width=2*semi_a*(mu/m2), height=2*semi_b*(mu/m2), 
                        edgecolor='b', fc='None', lw=0.5)

ax3.set_xlim(-2, 2)
ax3.set_ylim(-2, 2)
ax3.grid(True)

###############################################################################

# pause animation on click
def onClick(event):
    global pause
    pause ^= True

###############################################################################

def init():
    global M1, M2, orb1, orb2, Mv1, Mv2, Mi1, Mi2, orbi1, orbi2, phase, v1, v2

    M1.center = (-100, -100)        # initialize the patches at a far location
    M2.center = (-100, -100)
    
    ax.add_patch(M1)
#    art3d.pathpatch_2d_to_3d(M1, z=0, zdir="x")
    ax.add_patch(M2)
#    art3d.pathpatch_2d_to_3d(M2, z=0, zdir="x")

#    orb1.center = (-100, -100)
#    ax.add_patch(orb1)
#    orb2.center = (-100, -100)
#    ax.add_patch(orb2)
#####################################################    

    Mv1.center = (-100, -100)
    Mv2.center = (-100, -100)
     
    ax2.add_patch(Mv1)
    ax2.add_patch(Mv2)
#####################################################

    Mi1.center = (-100, -100)
    Mi2.center = (-100, -100)

    ax3.add_patch(Mi1)
    ax3.add_patch(Mi2)

    orbi1.center = (-100, -100)
    ax3.add_patch(orbi1)
    orbi2.center = (-100, -100)
    ax3.add_patch(orbi2)
######################################################    
    
## return everything that you want to remain visible as the animation runs    
    return M1,M2, orb1, orb2, Mv1, Mv2, Mi1, Mi2, orbi1, orbi2, v1, v2

###############################################################################

def update(val):
    global comx, comy, m1, m2, d6
    global semi_a, semi_b, ecc, alpha, nodeangle, inclination 
    global mu, L, P, r , r1, r2
    global M1, M2, orb1, orb2, Mi1, Mi2, orbi1, orbi2, v1, v2, pause
    global phase, timer, xt, tarray, change

    phase = 0.
    timer = 0.
    
    v1.set_visible(False)
    v2.set_visible(False)
    
    orb2.set_visible(False)
    orb2.set_visible(False)    
    
    m1 = round(s_m1.val,1)
    m2 = round(s_m2.val,1)

    semi_a = round(s_a.val,1)

    if round(s_ecc.val,1) != ecc :
        ecc = round(s_ecc.val,1)
        change = True

    alpha = semi_a*(1-ecc**2)

    nodeangle = np.radians(int(s_node.val))

    inclination = np.radians(int(s_inc.val))
    
    mu = ((m1*m2)/(m1+m2));

    semi_b = semi_a*(1-ecc**2)**0.5

    L = np.sqrt(mu*alpha)

    P = ((1/(m1+m2))*semi_a**3)**0.5

    if change == True:

        for i in range(721):
            tht = np.radians(phase)    
            tarray[i] = tht
            phase += np.absolute((1 + ecc*np.cos(tht))**2 / (1 - ecc**2)**1.5)
            phase %= 360

        phase = 0.
        change = False

    xt[:]= [(2*P/720)*x for x in range(721)]    

    r = alpha/(1+ecc);
    r1 = r*(mu/m1);
    r2 = -r*(mu/m2);
    
    M1.set_radius(0.03*(semi_a))
    M2.set_radius(0.03*(semi_a)) 

    orb1.set_xdata(comx + (mu/m1)*(alpha/(1+(ecc*np.cos(tarray[0:361])))) * np.cos(tarray[0:361] + nodeangle));
    orb1.set_ydata(comy + (mu/m1)*(alpha/(1+(ecc*np.cos(tarray[0:361])))) * np.cos(inclination) * np.sin(tarray[0:361] + nodeangle));
    orb1.set_visible(True)
    ax.draw_artist(orb1)        
        
    orb2.set_xdata(comx - (mu/m2)*(alpha/(1+(ecc*np.cos(tarray[0:361])))) * np.cos(tarray[0:361] + nodeangle));
    orb2.set_ydata(comy - (mu/m2)*(alpha/(1+(ecc*np.cos(tarray[0:361])))) * np.cos(inclination) * np.sin(tarray[0:361] + nodeangle));
    orb2.set_visible(True)
    ax.draw_artist(orb2)
    
########### Old orbit plot attempt ####
#    orb1.center = (comx + semi_a*(ecc)*(mu/m1)*np.cos(nodeangle+np.pi), comy + np.cos(inclination)*semi_a*(ecc)*(mu/m1)*np.sin(nodeangle+np.pi))
#
#    orb1.width = 2*semi_a*(mu/m1)*(np.cos(nodeangle))**2 + 2*semi_b*(mu/m1)*(np.sin(nodeangle))**2
#    orb1.height = np.cos(inclination)*(2*semi_a*(mu/m1)*(np.sin(nodeangle))**2 + 2*semi_b*(mu/m1)*(np.cos(nodeangle))**2)
#    #orb1.angle = np.rad2deg(nodeangle)   
#    
#    orb2.center = (comx + semi_a*(ecc)*(mu/m2)*np.cos(nodeangle), comy + np.cos(inclination)*semi_a*(ecc)*(mu/m2)*np.sin(nodeangle))
#
#    orb2.width = 2*semi_a*(mu/m2)*(np.cos(nodeangle))**2 + 2*semi_b*(mu/m2)*(np.sin(nodeangle))**2
#    orb2.height = np.cos(inclination)*(2*semi_a*(mu/m2)*(np.sin(nodeangle))**2 + 2*semi_b*(mu/m2)*(np.cos(nodeangle))**2)
    #orb2.angle = np.rad2deg(nodeangle)

    ax.set_xlim(-2*semi_a, 2*semi_a)
    ax.set_ylim(-2*semi_a, 2*semi_a)
###############################################################    

    d3 = 29.87*np.sqrt((m1+m2/alpha))*(1+ecc)
    d4 = 29.87*np.sqrt((m1+m2/alpha))* ecc
     
    d6 = np.sin(inclination)*np.sqrt((d4**2 + d3**2))
    
    
    v1.set_ydata((mu/m1)*np.sin(inclination)*(d4*np.sin(tarray+nodeangle)*np.sin(tarray) + (1/(1+ecc))*d3*np.cos(tarray+nodeangle)*(1+ecc*np.cos(tarray))))
    v1.set_xdata(xt)
    v1.set_visible(True)    
    ax2.draw_artist(v1)
    
    v2.set_ydata((-mu/m2)*np.sin(inclination)*(d4*np.sin(tarray+nodeangle)*np.sin(tarray) + (1/(1+ecc))*d3*np.cos(tarray+nodeangle)*(1+ecc*np.cos(tarray)))) 
    v2.set_xdata(xt)
    v2.set_visible(True)
    ax2.draw_artist(v2)
 
    ax2.set_xlim(0, 2*P)
    ax2.set_ylim(-d6-0.1, d6+0.1)   
    
    Mv1.set_radius(0.05*(P))
    Mv2.set_radius(0.05*(P)) 
###############################################################

    Mi1.set_radius(0.05*(semi_a))
    Mi2.set_radius(0.05*(semi_a)) 
    
    orbi1.width = 2*semi_a*(mu/m1)
    orbi1.height = 2*semi_b*(mu/m1)

    orbi1.angle = np.rad2deg(nodeangle)   
    orbi1.center = (comx + semi_a*(ecc)*(mu/m1)*np.cos(nodeangle+np.pi), comy + semi_a*(ecc)*(mu/m1)*np.sin(nodeangle+np.pi))
    
    orbi2.width = 2*semi_a*(mu/m2)
    orbi2.height = 2*semi_b*(mu/m2)
    
    orbi2.angle = np.rad2deg(nodeangle)
    orbi2.center = (comx + semi_a*(ecc)*(mu/m2)*np.cos(nodeangle), comy + semi_a*(ecc)*(mu/m2)*np.sin(nodeangle))
    
    ax3.set_xlim(-2*semi_a, 2*semi_a)
    ax3.set_ylim(-2*semi_a, 2*semi_a)
##################################################################    
    pause = False

###############################################################################

def animate(i):
    
    global semi_a, alpha, ecc, inclination, nodeangle
    global r, r1, r2, mu, m1, m2, P
    global M1, M2, orb1, orb2, Mi1, Mi2, orbi1, orbi2, comx, comy
    global phase, tarray, timer, xt

    if not pause:
        
        tht = phase
        
        r = alpha/(1+(ecc*np.cos(tht)));
        r1 = r*(mu/m1);
        r2 = -r*(mu/m2);
#############################################################    
        #x1, y1 = M1.center
        x1 = comx + r1 * np.cos(tht + nodeangle);
        y1 = (comy + r1 * np.cos(inclination) * np.sin(tht + nodeangle));
        
        #x2, y2 = M2.center
        x2 = comx + r2 * np.cos(tht + nodeangle);
        y2 = (comy + r2 * np.cos(inclination) * np.sin(tht + nodeangle));
        
        M1.center = (x1, y1)
        M2.center = (x2, y2)
    
#        orb1.center = (comx + semi_a*(ecc)*(mu/m1)*np.cos(nodeangle+np.pi), comy + np.cos(inclination)*semi_a*(ecc)*(mu/m1)*np.sin(nodeangle+np.pi))
#        orb2.center = (comx + semi_a*(ecc)*(mu/m2)*np.cos(nodeangle), comy + np.cos(inclination)*semi_a*(ecc)*(mu/m2)*np.sin(nodeangle))
############################################################ 

        d3 = 29.87*np.sqrt((m1+m2/alpha))*(1+ecc*np.cos(tht))
        d4 = 29.87*np.sqrt((m1+m2/alpha))* ecc*np.sin(tht)
     
        d6 = np.sin(inclination)*(d4*np.sin(tht+nodeangle) + d3*np.cos(tht+nodeangle))
         
        
        vm1 = (mu/m1)*d6
        vm2 = -(mu/m2)*d6
     
        xv1, yv1 = Mv1.center
        xv1 = 2*P*(timer/720)
        yv1 = vm1
         
        xv2, yv2 = Mv2.center
        xv2 = 2*P*(timer/720)
        yv2 = vm2
        
        Mv1.center = (xv1, yv1)
        Mv2.center = (xv2, yv2)
#############################################################

        xi1 = comx + r1 * np.cos(tht + nodeangle);
        yi1 = comy + r1 * np.sin(tht + nodeangle);
        
        xi2 = comx + r2 * np.cos(tht + nodeangle);
        yi2 = comy + r2 * np.sin(tht + nodeangle);

        Mi1.center = (xi1, yi1)
        Mi2.center = (xi2, yi2)
    
        orbi1.center = (comx + semi_a*(ecc)*(mu/m1)*np.cos(nodeangle+np.pi), comy + semi_a*(ecc)*(mu/m1)*np.sin(nodeangle+np.pi))
        orbi2.center = (comx + semi_a*(ecc)*(mu/m2)*np.cos(nodeangle), comy + semi_a*(ecc)*(mu/m2)*np.sin(nodeangle))
############################################################ 

        phase = tarray[timer]        
        timer += 1
        timer %= 720
    
    return M1,M2, orb1, orb2, Mv1, Mv2, v1, v2, Mi1, Mi2, orbi1, orbi2

###############################################################################

fig.canvas.mpl_connect('button_press_event', onClick)

anim = animation.FuncAnimation(fig, animate, 
                               init_func=init, 
                               frames=360, 
                               interval=20,
                               blit=True, repeat = True)
                               
###############################################################################

############################## Put sliders

axslider_inc = plt.axes([0.1, 0.92, 0.25, 0.03])
s_inc = plt.Slider(axslider_inc, 'Inc ', 0, 90, valfmt='%0d', valinit=90)
s_inc.on_changed(update)

axslider_node = plt.axes([0.65, 0.92, 0.25, 0.03])
s_node = plt.Slider(axslider_node, 'Node Angle', -90, 90, valfmt='%0d', valinit=0)
s_node.on_changed(update)

axslider_a = plt.axes([0.1, 0.06, 0.5, 0.03])
s_a = plt.Slider(axslider_a, 'a ', 0.1, 10.0, valfmt='%0.1f', valinit=1.0)
s_a.on_changed(update)

axslider_ecc = plt.axes([0.1, 0.01, 0.5, 0.03])
s_ecc = plt.Slider(axslider_ecc, 'Ecc ', 0, 0.9, valfmt='%0.1f', valinit=0.3)
s_ecc.on_changed(update)

axslider_m1 = plt.axes([0.67, 0.06, 0.25, 0.03])
s_m1 = plt.Slider(axslider_m1, 'm1 ', 0.1, 10.0, valfmt='%0.1f', valinit=3.0)
s_m1.on_changed(update)

axslider_m2 = plt.axes([0.67, 0.01, 0.25, 0.03])
s_m2 = plt.Slider(axslider_m2, 'm2 ', 0.1, 10.0, valfmt='%0.1f', valinit=1.0)
s_m2.on_changed(update)

###############################################################################

plt.show()