'''
Created on Aug 19, 2012

@author: mattias

This is about mobility, here type (t) doesn't not stand for the wafer type but rather the carrier type
'''

#import tkFileDialog, sys, re, numpy,scipy
from numpy import *
import time
#import glob
#import os
#import scipy.optimize


###
# This is for testing new things, like the mobility model im about to right
###


##
# global vairables needed, concentration, doping
#
def uc(t):
    i=1
    if(t=='n'):
        i = 0

    return umin[i]*umax[i]/(umax[i]-umin[i])

def un(t):
    i=1
    if(t=='n'):
        i = 0
    #print i

    return umax[i]*umax[i] / (umax[i]-umin[i])




def F(t):
    i=1
    j=0
    if(t=='n'):
        i = 0
        j = 1

    #mr = [1./1.258,1.258]  #value taken from users.cecs.anu.edu.au/~u5096045/QSSModel52.xls is m1/m2

    return (r1*P(t)**r6+r2+r3*mr[i]/mr[j])/(P(t)**(r6)+r4+r5*mr[i]/mr[j])

def P(t):

    return 1/(fCW/PCW(t) + fBH/PBH(t))

def PCW(t):
    i=1
    carrier = n ##meant to be other way around, so its right as it is
    if(t=='n'):
        i = 0
        carrier = p

    return 3.97e13*(1/(Nsc(t))*((T/300.)**(3.)))**(2./3.)


def PBH(t):
    i=1
    carrier = p+n
    if(t=='n'):
        i = 0


    return 1.36e20/carrier*mr[i]*(T/300.0)**2.0

def Z(t): #high doping effects - clustering
    i=1
    if(t=='n'):
        i = 0

    return 1. + 1./(c[i] + (Nref2[i]/Doping[i])**2.)

def G(t):
    i=1
    if(t=='n'):
        i = 0
    return 1. - s1/(s2+(T/300/mr[i])**s4*P(t))**s3+s5/((300/T/mr[i])**s7*P(t))**s6

def Nsc(t):

    carrier = n ## these are meant to be reversed
    if(t=='n'):
        carrier = p

    return Doping[0]*Z('n')+Doping[1]*Z('p')+carrier

def Nsceff(t):
    i=1
    j=0
    carrier = n ## these are meant to be reversed

    if(t=='n'):
        i = 0
        carrier = p
        j=1

    return Doping[i]*Z(i)+Doping[j]*Z(j)*G(t)+carrier/F(t)


#Dopant and carrier-carrier scattering
def uDCS(t):

    if(t=='n'):
        i = 0
    else:
        i= 1
    return  un(t) * Nsc(t)/Nsceff(t)   *  (Nref[i]/Nsc(t))**(alpha[i])   +uc(t)*((n+p)/Nsceff(t))

#Dopant and carrier-carrier scattering
def uLS(t):
    if(t=='n'):
        i = 0
    else:
        i= 1
    return  umax[i]*(300/T)**theta[i]


def ntype(Dopingn,Dopingp,deltan):
    global T,n,p,Doping
    constants()
    Doping = array([Dopingn,Dopingp])
    n,p = Doping[0] + deltan,Doping[1] + deltan

    return 1/(1/uDCS('n')+1/uLS('n'))



def ptype(Dopingn,Dopingp,deltan):
    global n,p,Doping
    constants()
    Doping = array([Dopingn,Dopingp])
    n,p = Doping[0] + deltan,Doping[1] + deltan

    return 1/(1/uDCS('p')+1/uLS('p'))


def Sum(Dopingn,Dopingp,deltan):
    global n,p,Doping
    t0=time.clock()
    constants()
    
    Doping = array([Dopingn,Dopingp])
    
    n,p = Doping[0] + deltan,Doping[1] + deltan
    mun =1/(1/uDCS('n')+1/uLS('n'))
    mup= 1/(1/uDCS('p')+1/uLS('p'))

    return mun + mup

def constants():
    global ni,umax,umin,theta,Nref,alpha,T,mr,s1,s2,s3,s4,s5,s6,s7,c,Nref2,r1,r2,r3,r4,r5,r6,fCW,fBH
    umax        = array([1414, 470.5])
    umin        = array([68.5, 44.9])
    theta       = array([2.285, 2.247])

    ni = 9.66e9

    #Nref        = array([9.68e16, 2.23e17]) #sentarous - arsnic
    Nref        = array([9.2e16, 2.23e17])
    alpha       = array([.711, .719])

    c           = array([0.21,0.5])
    Nref2       = array([4e20,7.2e20])

    fCW,fBH = 2.459,3.828

    T = 300
    mr = array([1.,1.258])

    s1,s2,s3,s4,s5,s6,s7 = .89233,.41372,.19778, .28227, .005978,1.80618,0.72169
    r1,r2,r3,r4,r5,r6    = .7643,2.2999,6.5502,2.367,-0.8552,.6478


if __name__ == '__main__':






    deltan  = logspace(10,20,11)
    #deltan = 1e13
    Dopingn,Dopingp = 1e16,1
    print 'n-type'
    print 'Delta n \t N Mobility \t P Mobility'
    print column_stack((deltan,ntype(Dopingn,Dopingp,deltan),ptype(Dopingn,Dopingp,deltan)))
    #plot(deltan,Sum(Dopingn,Dopingp,deltan),label='n-type')

    Dopingn,Dopingp = 1,1e16
    print 'p-type'
    print 'Delta n \t N Mobility \t P Mobility'
    print column_stack((deltan,ntype(Dopingn,Dopingp,deltan),ptype(Dopingn,Dopingp,deltan)))
    print 'Sum'
    print 'Delta n \t Sum Mobility '
    print column_stack((deltan,Sum(Dopingn,Dopingp,deltan)))
    #plot(deltan,Sum(Dopingn,Dopingp,deltan),label='p-type')
    #plot(deltan,mup,label='p-type')
    #semilogx()
    #legend(loc=0)
    #grid(True)


    #show()
    Conductivity = 1.2e-0
    Deltan = 1e4
    print Deltan
    dop = 1e17
    for i in range(6):

        Deltai=Deltan
        Deltan  = Conductivity / 1.6e-19/Sum(0,dop,Deltan)/0.018
        print (Deltan-Deltai)/Deltai*100 ,i+1,Deltan













