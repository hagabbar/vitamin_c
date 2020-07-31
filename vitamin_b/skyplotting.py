import numpy as np
from ligo.skymap import kde
import matplotlib
matplotlib.use('Agg')
from matplotlib.colors import to_rgb
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap

def greedy(density):
    i,j = np.shape(density)
    idx = np.argsort(density.flatten())[::-1]
    c = np.cumsum(density.flatten()[idx])
    c = c/c[-1]
    np.append(c,1.0)
    p = np.zeros(i*j)
    p[idx] = c[:]
    return p.reshape(i,j)

def plot_sky(pts,contour=True,filled=False,ax=None,trueloc=None,cmap='Reds',col='red'):

    cls = kde.Clustered2DSkyKDE
    pts[:,0] = pts[:,0] - np.pi
    skypost = cls(pts, trials=5, jobs=8)

    # make up some data on a regular lat/lon grid.
    nlats = 145; nlons = 291; delta = 2.*np.pi/(nlons-1)
    lats = (0.5*np.pi-delta*np.indices((nlats,nlons))[0,:,:])
    lons = (delta*np.indices((nlats,nlons))[1,:,:]-np.pi)
    locs = np.column_stack((lons.flatten(),lats.flatten()))
    prob = skypost(locs).reshape(nlats,nlons)
    p1 = greedy(prob)
 
    # compute mean location of samples
    nx = np.cos(pts[:,1])*np.cos(pts[:,0])
    ny = np.cos(pts[:,1])*np.sin(pts[:,0])
    nz = np.sin(pts[:,1])
    mean_n = [np.mean(nx),np.mean(ny),np.mean(nz)]
    bestloc = [trueloc[0],trueloc[1]]

    if ax is None:
        map = Basemap(projection='moll',lon_0=0,resolution=None,celestial=True)
        map.drawmapboundary(fill_color='white')
        # draw lat/lon grid lines every 30 degrees.
        meridian = ["-180","-150","-120","-90","-60","-30","0","30","+60","+90","+120","+150"]
        map.drawmeridians(np.arange(-180,180,30),labels=[1,1,1,1])
        for i in np.arange(len(meridian)):
            plt.annotate(r"$\textrm{%s}$" % meridian[i] + u"\u00b0",xy=map(np.arange(-180,180,30)[i],0),xycoords='data')
        map.drawparallels(np.arange(-90,90,30),labels=[1,0,0,0])
    else:
        map = ax    

    # compute native map projection coordinates of lat/lon grid.
    x, y = map(lons*180./np.pi, lats*180./np.pi)
 
    # contour data over the map.
    if filled:
        base_color = np.array(to_rgb(col))
        opp_color = 1.0 - base_color
        cs1 = map.contourf(x,y,1.0-p1,levels=[0.0,0.1,0.5,1.0],colors=[base_color+opp_color,base_color+0.8*opp_color,base_color+0.6*opp_color,base_color])
    cs2 = map.contour(x,y,p1,levels=[0.5,0.9],linewidths=2.0,colors=col)
    if trueloc is not None:
        xx, yy = map((trueloc[0]*180./np.pi)-180.0, trueloc[1]*180./np.pi)
        map.plot(xx,yy,marker='+',markersize=20,linewidth=5,color='black')
    return map




