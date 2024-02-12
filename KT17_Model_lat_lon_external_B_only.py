# -*- coding: utf-8 -*-
"""
Created on Wed May 25 16:12:02 2022

@author: smz16
"""
import numpy as np
import math
import pandas as pd

'''
 *** Parameters v1.0 ***

 Model parameters from Korth et al., Modular model for Mercury’s magnetospheric
 magnetic field confined within the average observed magnetopause, J. Geophys.
 Res. Space Physics, 120, doi: 10.1002/2015JA021022, 2015.
'''

mu= 190.00 # dipole moment [nT Rp^3]
tilt= 0.00*3.141592650/180.00 # dipole tilt angle [radians]
pdip= 0.00*3.141592650/180.00 # dipole longitude [radians]
rdx= 0.00 # dipole x offset
rdy= 0.00 # dipole y offset
rdz= 0.1960 # dipole z offset
#rss= 1.410 # distance from Mercury center to sub-solar magnetopause [Rp]
#rss is replaced by P_ram calculation in Variables section

r0= 1.420 # distance from Mercury center to fitted sub-solar magnetopause [Rp]

alfa= 0.50 # magnetopause flaring factor
tamp1= 7.640 # tail disk current magnitude
tamp2= 2.060 # harris sheet current magntidue
d0 = 0.090 # half-width of current sheet in Z at inner edge of tail current [Rp]

deltadx= 1.00 # expansion magnitudes of tail current sheet in x direction
deltady= 0.10 # expansion magnitudes of tail current sheet in y direction

scalex= 1.50 # e-folding distance for the sunward expansion of the harris sheet
scaley= 9.00 # scale distance for the flankward expansion of the harris sheet

zshift= 3.50 # location of image sheets for harris sheet

mptol= 1.0e-3 # Tolerance for magnetopause encounter

r_taildisk=[59048.35734,-135664.4246,-913.4507339, 209989.1008, -213142.9370, \
            19.69235037, -18.16704312, 12.69175932, -14.13692134, 14.13449724, \
            7.682813704, 9.663177797, 0.6465427021, 1.274059603, 1.280231032]

r_dipshld=[7.792407683, 74.37654983, 4.119647072, -131.3308600, 546.6006311, \
          -1077.694401, 52.46268495, 1057.273707, -74.91550119, -141.8047123, \
          3.876004886, 156.2250932, -506.6470185, 1439.804381, -64.55225925, \
          -1443.754088, 0.1412297078, 0.7439847555, 1.042798338, 0.7057116022]

r_diskshld= [-398.4670279, -1143.001682, -1836.300383, -73.92180417, -326.3986853, \
             -29.96868107, -1157.035602, -604.1846034, -52.04876183, -2030.691236, \
             -1529.120337, -6.382209946, 2587.666032, 213.8979183, -28.30225993, \
             630.1309859, 2968.552238, 888.6328623, 497.3863092, 2304.254471, \
             858.4176875, 1226.958595, 850.1684953, -20.90110940, -203.9184239, \
             -792.6099018, 1115.955690, 527.3226825, 22.47634041, -0.0704405637, \
             -1405.093137, -97.20408343, 5.656730182, -138.7129102, -1979.755673, \
             5.407603749, 1.091088905, 0.6733299808, 0.3266747827, 0.9533161464, \
             1.362763038, 0.0014515208]

r_slabshld=[-91.67686636, -87.31240824, 251.8848107, 95.65629983, -80.96810700, \
            198.1447476, -283.1968987, -269.1514899, 504.6322310, 166.0272150, \
            -214.9025413, 623.7920115, -35.99544615, -322.8644690, 345.7105790, \
            928.8553184, 810.1295090, 19.62627762, -12.70326428, 490.4662048, \
            -814.0985363, -1781.184984, -1371.261326, 60.31364790, 116.6305510, \
            -178.3347065, 604.0308838, 1155.151174, 770.3896601, -202.8545948, \
            298.6337705, 304.7964641, 33.70850254, 393.6080147, 308.1194271, \
            -660.1691658, 1.677629714, 1.292226584, 0.3116253398, -0.4392669057, \
            0.7578074817, 1.497779521] 
 
n_taildisk=15
n_dipshld=20
n_diskshld=42
n_slabshld=42



    
'''
-----------------------------------------------------------------------
 Subroutine KT17_TAILSLAB
-----------------------------------------------------------------------

 calculates msm components of the field from an equatorial harris-type current
 sheet, slowly expanding sunward

------------input parameters:

 d0 - basic (minimal) half-thickness
 deltadx - sunward expansion factor for the current sheet thickness
 deltady - flankward expansion factor for the current sheet thickness
 scalex - e-folding distance for the sunward expansion of the current sheet
 scaley - scale distance for the flankward expansion of the current sheet
 zshift - z shift of image sheets
 x,y,z - msm coordinates

------------output parameters:
 bx,by,bz - field components in msm system, in nanotesla.
'''
def kt17_tailslab(xmsm,ymsm,zmsm):
    
    d=d0+deltadx*np.exp(xmsm/scalex)+deltady*(ymsm/scaley)**2
    dddx=deltadx/scalex*np.exp(xmsm/scalex)
    zpzi=zmsm+zshift
    zmzi=zmsm-zshift
    bx=(np.tanh(zmsm/d)-0.50*(np.tanh(zmzi/d)+np.tanh(zpzi/d)))/d
    by=0.00
    bz=(zmsm*np.tanh(zmsm/d)-0.50*(zmzi*np.tanh(zmzi/d)+zpzi*np.tanh(zpzi/d)))*dddx/d**2

    return bx,by,bz
    
    
    
'''
-----------------------------------------------------------------------
 Subroutine KT17_TAILDISK
-----------------------------------------------------------------------
 calculates msm components of the field from a t01-like 'long-module' equatorial
 current disk with a 'hole' in the center and a smooth inner edge
 (see tsyganenko, jgra, v107, no a8, doi 10.1029/2001ja000219, 2002, fig.1, right
 panel).

------------input parameters:

 d0 - basic (minimal) half-thickness
 deltadx - sunward expansion factor for the current sheet thickness
 deltady - flankward expansion factor for the current sheet thickness
 x,y,z - msm coordinates

------------output parameters:
 bx,by,bz - field components in msm system, in nanotesla.
'''

def kt17_taildisk(xmsm,ymsm,zmsm):
    nr3=int(n_taildisk/3)
    f=r_taildisk[0:nr3]
    b=r_taildisk[nr3:2*nr3] 
    c=r_taildisk[2*nr3:3*nr3]
    
    xshift=0.30 # shift the center of the disk to the dayside by xshift
    sc=7.00 # renormalize length scales
    
    x=(xmsm-xshift)*sc
    y=ymsm*sc
    z=zmsm*sc
    d0_sc=d0*sc
    deltadx_sc=deltadx*sc
    deltady_sc=deltady*sc
    
    rho=np.sqrt(x**2+y**2)
    drhodx=x/rho
    drhody=y/rho
    
    dex=np.exp(x/7.00)
    d=d0_sc+deltady_sc*(y/20.00)**2+deltadx_sc*dex 
    # the last term makes the sheet thicken sunward, 
    # to avoid problems in the subsolar region
    dddy=deltady_sc*y*0.0050 
    dddx=deltadx_sc/7.00*dex
    
    dzeta=np.sqrt(z**2+d**2) 
    # this is to spread out the sheet in z direction over
    # finite thickness 2d
    ddzetadx=d*dddx/dzeta
    ddzetady=d*dddy/dzeta
    ddzetadz=z/dzeta
    
    bx=0.00
    by=0.00
    bz=0.00 
    
    for i in range(0,5):
        bi=b[i]
        ci=c[i]
        
        s1=np.sqrt((rho+bi)**2+(dzeta+ci)**2)
        s2=np.sqrt((rho-bi)**2+(dzeta+ci)**2)
        
        ds1drho=(rho+bi)/s1
        ds2drho=(rho-bi)/s2
        ds1ddz=(dzeta+ci)/s1
        ds2ddz=(dzeta+ci)/s2
        
        ds1dx=ds1drho*drhodx+ds1ddz*ddzetadx
        ds1dy=ds1drho*drhody+ds1ddz*ddzetady
        ds1dz=ds1ddz*ddzetadz
        
        ds2dx=ds2drho*drhodx+ds2ddz*ddzetadx
        ds2dy=ds2drho*drhody+ds2ddz*ddzetady
        ds2dz=ds2ddz*ddzetadz
        
        s1ts2=s1*s2
        s1ps2=s1+s2
        s1ps2sq=s1ps2**2 
        
        fac1=np.sqrt(s1ps2sq-(2.00*bi)**2)
        as_var=fac1/(s1ts2*s1ps2sq) 
        # NOTE have changed as to as_var for python compatibility
        dasds1=(1.00/(fac1*s2)-as_var/s1ps2*(s2*s2+s1*(3.00*s1+4.00*s2)))/(s1*s1ps2)
        dasds2=(1.00/(fac1*s1)-as_var/s1ps2*(s1*s1+s2*(3.00*s2+4.00*s1)))/(s2*s1ps2)
        
        dasdx=dasds1*ds1dx+dasds2*ds2dx
        dasdy=dasds1*ds1dy+dasds2*ds2dy
        dasdz=dasds1*ds1dz+dasds2*ds2dz
        
        bx=bx-f[i]*x*dasdz
        by=by-f[i]*y*dasdz
        bz=bz+f[i]*(2.00*as_var+x*dasdx+y*dasdy)

    return bx, by, bz

'''        
-----------------------------------------------------------------------
 Subroutine KT17_SHIELD
-----------------------------------------------------------------------    
'''    

def kt17_shield(n,r,x,y,z):
    
    o=int(round(-0.5+np.sqrt(n+0.25)))
    c=r[0:o*o]
    p=r[o*o:o*o+o]

    jmax=o
    kmax=o

    bx=0.00
    by=0.00
    bz=0.00
    
    for j in range(0,jmax):
        for k in range(0,kmax):
            cypj=np.cos(y*p[j])
            sypj=np.sin(y*p[j])
            szpk=np.sin(z*p[k])
            czpk=np.cos(z*p[k])
            sqpp=np.sqrt(p[j]**2+p[k]**2)

            epp=np.exp(x*sqpp)

            hx=-sqpp*epp*cypj*szpk
            hy=+epp*sypj*szpk*p[j]
            hz=-epp*cypj*czpk*p[k]
            bx=bx+hx*c[(j)*kmax+k]
            by=by+hy*c[(j)*kmax+k]
            bz=bz+hz*c[(j)*kmax+k]

    return bx,by,bz    
    
'''
-----------------------------------------------------------------------
 Subroutine KT17_MPDIST
-----------------------------------------------------------------------
'''

def kt17_mpdist(mode,x,y,z):
    rho2=y**2+z**2
    r=np.sqrt(x**2+rho2)
    rho=np.sqrt(rho2)

    id_var=1 

    if rho > 1.0e-8:
        # not on the x-axis - no singularities to worry about
        ct=x/r
        st=rho/r
        t=math.atan2(st,ct)
        sp=z/rho
        cp=y/rho 
    else:           # on the x-axis
        if x > 0.0: # on the dayside
            ct=x/r
            st=1.0/r  #set rho=10**-8, to avoid singularity of grad_fi (if mode=1, see gradfip=... below)
            t=math.atan2(st,ct)
            sp=0.00
            cp=1.00  
        else: # on the tail axis to avoid singularity:
            fi = -1000.00 # assign rm=1000 (a conventional substitute value)
            # Adding these to avoid the singularity
            ct = 0 
            t = 0
            
    rm=r0/np.sqrt(alfa*(1.00+ct)) # standard form of shue et al.,1997, magnetopause model
    if rm < r:
        id_var = -1 # NOTE conversion from id to id_var
    fi = r - rm
    
    if mode != 0:
        drm_dt=0.250*rm**3/r0**2*st
        gradfir=1.00
        gradfit=-drm_dt/r
        gradfip=0.00 # axial symmetry
        gradfix=gradfir*ct-gradfit*st
        gradfiy=(gradfir*st+gradfit*ct)*cp-gradfip*sp
        gradfiz=(gradfir*st+gradfit*ct)*sp+gradfip*cp 
    else:
        gradfix=np.nan
        gradfiy=np.nan
        gradfiz=np.nan
    return fi,gradfix,gradfiy,gradfiz,id_var,t 
    
'''
-----------------------------------------------------------------------
 Subroutine KT17_DIPOLE
-----------------------------------------------------------------------
'''
            
def kt17_dipole(xmsm,ymsm,zmsm):
    #calculates components of dipole field
    #input parameters: x,y,z - msm coordinates in rm (1 rm = 2440 km)
    #output parameters: bx,by,bz - field components in msm system, in nanotesla.
    
    # dipole tilt
    psi=0.00
    sps=np.sin(psi/57.295779510)
    
    cps=np.sqrt(1.00-sps**2)
    # compute field components
    p=xmsm**2
    u=zmsm**2
    v=3.00*zmsm*xmsm
    t=ymsm**2
    q=mu/np.sqrt(p+t+u)**5
    bx=q*((t+u-2.00*p)*sps-v*cps)
    by=-3.00*ymsm*q*(xmsm*sps+zmsm*cps)
    bz=q*((p+t-2.00*u)*cps-v*sps) 
 
    return bx,by,bz 


'''
-----------------------------------------------------------------------
 Subroutine KT17_BFIELD
-----------------------------------------------------------------------
'''

def kt17_bfield(n,x_a,y_a,z_a):

    # initialize variables
    
    kappa=r0/rss
    kappa3=kappa**3
    
    bx_a = []; by_a = []; bz_a = []
    # magnetic field computation
     
    for i in range(0,n):
        if n == 1: # NOTE added this for python compatibility
            x_a = [x_a]
            y_a = [y_a]
            z_a = [z_a]

        x=x_a[i]
        y=y_a[i]
        z=z_a[i]
        
        x=x*kappa
        y=y*kappa
        z=z*kappa
        
        fi,gradfix,gradfiy,gradfiz,id_var,t = kt17_mpdist(0,x,y,z)
        noshield = 0
        if fi < mptol:
            id_var = 1
        if noshield == 1:
            id_var = 1
            
        if id_var == 1:
            bx_dcf=0.00
            by_dcf=0.00
            bz_dcf=0.00
            bx_dsk=0.00
            by_dsk=0.00
            bz_dsk=0.00
            bx_slb=0.00
            by_slb=0.00
            bz_slb=0.00

            bx_dip, by_dip, bz_dip = kt17_dipole(x,y,z)
            bx_shld, by_shld, bz_shld = kt17_shield(n_dipshld,r_dipshld,x,y,z)
            '''
            bx_dcf=kappa3*(bx_dip+bx_shld)
            by_dcf=kappa3*(by_dip+by_shld)
            bz_dcf=kappa3*(bz_dip+bz_shld)
            '''
            bx_dcf=kappa3*(bx_shld)
            by_dcf=kappa3*(by_shld)
            bz_dcf=kappa3*(bz_shld)
            '''
            bx_dcf=kappa3*(bx_dip)
            by_dcf=kappa3*(by_dip)
            bz_dcf=kappa3*(bz_dip)
            '''
            
            bx_tldsk, by_tldsk, bz_tldsk = kt17_taildisk(x,y,z)
            bx_dkshld, by_dkshld, bz_dkshld = kt17_shield(n_diskshld,r_diskshld,x,y,z)
            '''
            bx_dsk=tamp1*(bx_tldsk+bx_dkshld)
            by_dsk=tamp1*(by_tldsk+by_dkshld)
            bz_dsk=tamp1*(bz_tldsk+bz_dkshld)
            '''
            bx_dsk=tamp1*(bx_dkshld)
            by_dsk=tamp1*(by_dkshld)
            bz_dsk=tamp1*(bz_dkshld)
            '''
            bx_dsk=tamp1*(bx_tldsk)
            by_dsk=tamp1*(by_tldsk)
            bz_dsk=tamp1*(bz_tldsk)
            '''
            
            bx_tlslb, by_tlslb, bz_tlslb = kt17_tailslab(x,y,z)
            bx_slbshld, by_slbshld, bz_slbshld = kt17_shield(n_slabshld,r_slabshld,x,y,z)
            '''
            bx_slb=tamp2*(bx_tlslb+bx_slbshld)
            by_slb=tamp2*(by_tlslb+by_slbshld)
            bz_slb=tamp2*(bz_tlslb+bz_slbshld)
            '''
            bx_slb=tamp2*(bx_slbshld)
            by_slb=tamp2*(by_slbshld)
            bz_slb=tamp2*(bz_slbshld)
            '''
            bx_slb=tamp2*(bx_tlslb)
            by_slb=tamp2*(by_tlslb)
            bz_slb=tamp2*(bz_tlslb)
            '''
            
            bx_msm=bx_dcf+bx_dsk+bx_slb
            by_msm=by_dcf+by_dsk+by_slb
            bz_msm=bz_dcf+bz_dsk+bz_slb
            bx_a.append(bx_msm)
            by_a.append(by_msm)
            bz_a.append(bz_msm)
            
        else:
            bx_a.append(1.0e-8)
            by_a.append(1.0e-8)
            bz_a.append(1.0e-8)

    return bx_a,by_a,bz_a


''' 
*** Variables ***

Model variables that can change on each iteration of the program
'''


# X, Y, Z are the coordinates at which the field will be calculated
# X = pd.read_csv('X_4_lat_long.csv', index_col=None, header=None).to_numpy()
# Y = pd.read_csv('Y_4_lat_long.csv', index_col=None, header=None).to_numpy()
# Z = pd.read_csv('Z_4_lat_long.csv', index_col=None, header=None).to_numpy()


#rhel = 0.39 # Heliocentric distance
act = 50.0 # Disturbance Index

    
f=2.06873-0.00279*act
B_eq=mu*10**(-9)
mu0=4*np.pi*10**(-7)

# determining standoff distance scale factor from 
# Winslow et al. (2013) average conditions R_SS = 1.45 R_M, P_ram = 14.3 nPa
scale_fac = 1.45/(((2*B_eq**2)/(mu0*14.3*10**(-9)))**(1/6))

P_ram = 15 # solar wind ram pressure [nPa]
rss=(((2*B_eq**2)/(mu0*P_ram*10**(-9)))**(1/6))*scale_fac
tamp1=6.4950+0.0229*act
tamp2=1.6245+0.0088*act










 


 
        
