# MD simulation in the NVEkin-ensemble

#-------------non-Graphics-libraries-------------------------------------------
import numpy as np
import pandas as pd
import random
import math
#--------------graphics-libraries----------------------------------------------
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
#---------------simulation settings--------------------------------------------
# critical point for force-shifted LJ-fluid:    Errington2003 [DOI: 10.1063/1.1532344]
#pc*=0.199; rhoc*=0.320; Tc*= 0.937.
#rho* = rho * (sigma^3/1 particle)
#T*= T * (kB/ epsilon)
#p* = p * (sigma^3/epsilon)




#number of particles
N=100
#Lennard-Jones particle properties - could be configured for each particle individually as well
sigma=1
epsilon=1
mass=1
#box length
length=pow(1/0.320*N,1.0/3.0)
boxlengths=[length,length,length]
#number of steps
Nit=300000
#time tep length
dt=0.008
#temperature
temperature=0.937     #this is technically kB T
#output file settings
filenameprefix='data/system'
filenamesuffix='.csv'
#------------------graphics settings--------------------------------------
pygame.init()
display = (800,600)
pygame.display.set_mode(display, DOUBLEBUF|OPENGL|HWSURFACE,32)
pygame.display.set_caption('MD simulation by Thomas Heinemann')
pygame.display.gl_set_attribute(GL_DEPTH_SIZE, 160)
glEnable(GL_DEPTH_TEST)
glDepthFunc(GL_LESS)
glDepthRange(0,1)
gluPerspective(450, (display[0]/display[1]), 0.1,boxlengths[2]*1.9)
glTranslatef(-boxlengths[0]*0.5,-boxlengths[1]*0.5, -boxlengths[2]*1.6)
#-------------------------------------------------------------------------



#-----------simple math---------------------------------------------------
def vlength_simple(a):
    return np.sqrt((a*a).sum())
#-------------------------------------------------------------------------


#-------------Lennard-Jones particle definition---------------------------
class LJ_Particle:
  def __init__(particleobject, pos, velo, acc, lj_epsilon, lj_sigma, mass):
    particleobject.pos = pos
    particleobject.velo = velo
    particleobject.acc = acc
    particleobject.lj_epsilon = lj_epsilon #LJ-parameter epsilon
    particleobject.lj_sigma = lj_sigma     #LJ-parameter sigma
    particleobject.mass = mass

def LJ_force(particle1, particle2, boxlengths):   #force shifted version
    #Lorentz-Berthelot rules
    epsilon=np.sqrt(particle1.lj_epsilon*particle2.lj_epsilon)
    sigma=0.5*(particle1.lj_sigma+particle2.lj_sigma)
    #acceleration update
    distv=dist_vect_periodic_boundary(particle1.pos,particle2.pos,boxlengths)
    r=np.sqrt((distv*distv).sum())
    #cutoff radius
    cutoff=2.5*sigma
    F_r=((-12.0*sigma/pow(r,13))-(-6.0*sigma/pow(r,7)))*4*epsilon*((1.0/r)*distv)
    F_cutoff=((-12.0*sigma/pow(cutoff,13))-(-6.0*sigma/pow(cutoff,7)))*4*epsilon*((1.0/r)*distv)
    if r<cutoff:
        return F_r-F_cutoff
    else:
        return np.array([0,0,0])
#-------------------------------------------------------------------------


#----------------------save routine for particles' positions and velocities (and also accelerations for numerical purpose)---------------------
def save_system(particleset,filename):
    print("saving "+filename)
    columns = ('index','posx', 'posy', 'posz', 'velox', 'veloy', 'veloz', 'accx', 'accy', 'accz')
    for i in range(0,N):
        data = {
            str(i): {'index': i, 'posx': particleset[i].pos[0], 'posy': particleset[i].pos[1], 'posz': particleset[i].pos[2], 'velox': particleset[i].velo[0], 'veloy': particleset[i].velo[1], 'veloz': particleset[i].velo[2], 'accx': particleset[i].acc[0], 'accy': particleset[i].acc[1], 'accz': particleset[i].acc[2]}
        }
        if i==0:
            df = pd.DataFrame(data=data, index=columns).T
        else:
            df=df.append(pd.DataFrame(data=data, index=columns).T)
    df.to_csv(filename)
#-------------------------------------------------------------------------

#---------handling periodic boundaries------------------------------------
def pos_periodic_boundary(a,boxlengths):
    return a % boxlengths

def radial_dist_periodic_boundary(a,b,boxlengths):  #minimum image convention
    distv=dist_vect_periodic_boundary(a,b,boxlengths)
    return vlength_simple(distv)

def dist_periodic_boundary(a,b,length):  #minimum image convention
    x=np.argmin([(b-a)%length,(a-b)%length])
    if x==0:
        return (b-a)%length
    else:
        return -((a-b)%length)

def dist_vect_periodic_boundary(a,b,boxlengths):  #minimum image convention
    newv=0*a
    for i in range(0,len(a)):
        newv[i]=dist_periodic_boundary(a[i],b[i],boxlengths[i])
    return newv
#-------------------------------------------------------------------------


#----------reading/setting physical quantities---------------------------------------------------------------
#----------kinetic energy----------------
def calc_kinetic_energy(particleset):
    Ekin=0
    for i in range(len(particleset)):
        Ekin=Ekin+particleset[i].mass*0.5*(particleset[i].velo*particleset[i].velo).sum()
    return Ekin

def calc_microscopic_temperature(particleset):
    return calc_kinetic_energy(particleset)/(1.5*len(particleset))

def set_microscopic_temperature(particleset,temperature):
    temperature0=calc_microscopic_temperature(particleset)
    factor=np.sqrt(temperature/temperature0)
    for i in range(len(particleset)):
        particleset[i].velo=factor*particleset[i].velo
    return particleset
#-----------------------------------------
#---------------pressure------------------
def calc_microscopic_pressure(particleset,microscopic_temperature, boxlengths):
    #p=N kB T / V  +  1 / V / d* Sum_(i<j) (Fij*(-)rij), rij=rj-ri, see de Miguel [DOI: 10.1063/1.2363381] - uses different definition for rij
    Fij_rij=0
    for i in range(0,len(particleset)):
        for j in range(0,i):
            Fij_rij=Fij_rij-(LJ_force(particleset[i], particleset[j], boxlengths)*dist_vect_periodic_boundary(particleset[i].pos, particleset[j].pos,boxlengths)).sum()
    V=boxlengths[0]*boxlengths[1]*boxlengths[2]
    p=len(particleset)*microscopic_temperature/V+1.0/(3.0*V)*Fij_rij
    return p
#-----------------------------------------
#-------------------------------------------------------------------------


#--------------graphics settings for simulation box and particles--------------------
def Cube():
    verticies = (
        (boxlengths[0], 0, 0),
        (boxlengths[0], boxlengths[1], 0),
        (0, boxlengths[1], 0),
        (0, 0, 0),
        (boxlengths[0], 0, boxlengths[2]),
        (boxlengths[0], boxlengths[1], boxlengths[2]),
        (0, 0, boxlengths[2]),
        (0, boxlengths[1], boxlengths[2])
        )
    edges = (
        (0,1),
        (0,3),
        (0,4),
        (2,1),
        (2,3),
        (2,7),
        (6,3),
        (6,4),
        (6,7),
        (5,1),
        (5,4),
        (5,7)
        )
    glColor(1, 1, 1)
    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(verticies[vertex])
    glEnd()

def particle_display(pos,radius):
    th_step=1.0*math.pi/10.
    phi_step=2.0*math.pi/10.
    for th_i in range(0,10):
        th=math.pi/10.0*th_i
        for phi_i in range(0,10):
            phi=2.0*math.pi/10.0*phi_i
            glColor(0.5*(math.cos(th)+1), 0, 0.5)
            glBegin(GL_TRIANGLES)
            glVertex((radius*math.cos(phi)*math.sin(th+th_step)+pos[0],radius*math.sin(phi)*math.sin(th+th_step)+pos[1],radius*math.cos(th+th_step)+pos[2]))
            glVertex((radius*math.cos(phi)*math.sin(th)+pos[0],radius*math.sin(phi)*math.sin(th)+pos[1],radius*math.cos(th)+pos[2]))
            glVertex((radius*math.cos(phi-phi_step)*math.sin(th+th_step)+pos[0],radius*math.sin(phi-phi_step)*math.sin(th+th_step)+pos[1],radius*math.cos(th+th_step)+pos[2]))
            glEnd()
            glBegin(GL_TRIANGLES)
            glVertex((radius*math.cos(phi-phi_step)*math.sin(th+th_step)+pos[0],radius*math.sin(phi-phi_step)*math.sin(th+th_step)+pos[1],radius*math.cos(th+th_step)+pos[2]))
            glVertex((radius*math.cos(phi)*math.sin(th)+pos[0],radius*math.sin(phi)*math.sin(th)+pos[1],radius*math.cos(th)+pos[2]))
            glVertex((radius*math.cos(phi-phi_step)*math.sin(th)+pos[0],radius*math.sin(phi-phi_step)*math.sin(th)+pos[1],radius*math.cos(th)+pos[2]))
            glEnd()
#-------------------------------------------------------------------------


#-------------------------------------------------------------------------
print("Creating initial configuration")
particleset=[]  # empty particle list
pos=np.array([0,0,0]) #initialize vector
for i in range(N):
    while True:
        pos=np.array([(random.random())*boxlengths[0],(random.random())*boxlengths[1],(random.random())*boxlengths[2]])
        overlap=False
        #checking for overlap
        for j in range(0,i):
            if radial_dist_periodic_boundary(pos,particleset[j].pos,boxlengths)<sigma:
                overlap=True
                break
        if overlap==False:
            break
        #
    velo=np.array([random.random()*1-0.5,random.random()*1-0.5,random.random()*1-0.5])   #simple initialization - not accounting for Boltzmann distribution
    acc=np.array([0,0,0])
    particleset.append(LJ_Particle(pos,velo, acc,sigma,epsilon,mass))
set_microscopic_temperature(particleset,temperature)
print("Saving initial configuration")
filename=filenameprefix+str(0)+filenamesuffix
save_system(particleset,filename)
#-------------------------------------------------------------------------


#-------------------------------------------------------------------------
print("Solving the Newtonian equations of motion using the V-Verlet method")


for it in range(1,Nit+1):

    paintcondition=(it % 20 ==0)
    savecondition=(it % 20 ==0)

    #quit program if window  has been closed
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
    #

    #calculating and printing physical quantities
    T=calc_microscopic_temperature(particleset)
    P=calc_microscopic_pressure(particleset,T, boxlengths)
    print(it, T, P)
    #


    #clear window
    if paintcondition:
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        Cube()
    #

    #Velocity-Verlet algorithm
    for i in range(N):
        #velocity update - half step
        particleset[i].velo=particleset[i].velo+0.5*dt*(particleset[i].acc)
        #delete acceleration
        particleset[i].acc=np.array([0,0,0])
        #position update step with periodic boundaries
        particleset[i].pos=pos_periodic_boundary(particleset[i].pos+dt*particleset[i].velo, boxlengths)
        if paintcondition:
            particle_display(particleset[i].pos,particleset[i].lj_sigma*0.5)
    #acceleration and full step velocity update
    for i in range(N):
        for j in range(i+1,N):
            #acceleration update
            accadd=LJ_force(particleset[i], particleset[j], boxlengths)/(particleset[i].mass)
            particleset[i].acc=particleset[i].acc+1*accadd
            particleset[j].acc=particleset[j].acc-1*accadd
        #velocity update part 2
        particleset[i].velo=particleset[i].velo+0.5*dt*(particleset[i].acc)
    particleset=set_microscopic_temperature(particleset,temperature)
    #saving system
    if savecondition:
        filename=filenameprefix+str(it)+filenamesuffix
        save_system(particleset,filename)
    #

    #update window and wait
    if paintcondition:
        #glRotatef(1, 3, 1, 1)
        pygame.display.flip()
        pygame.time.wait(2)
    #
#-------------------------------------------------------------------------

exit()
