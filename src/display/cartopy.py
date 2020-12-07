"""
Creates plots of SEVIR events using cartopy library
"""

import numpy as np
from matplotlib import animation, rc
import matplotlib.pyplot as plt
import cartopy.crs as crs
from cartopy.crs import Globe
from .display import get_cmap
import cartopy.feature as cfeature

def make_animation(frames,meta,img_type='vil',fig=None,
                   interval=100,title=None,**kwargs):
    """
    frames: numpy array
       [1,L,W,T] tensor, where T represents time steps
    meta pandas series
       meta data for frames
    ax 
         cartopy axis
    img_type string
        SEVIR img_type
    kwargs
        other inputs to ax.imshow
        
    Returns     matplotlib.animation object
    
    """
    proj,img_extent = make_ccrs(meta)
    if fig is None:
        fig=plt.gcf()
    ax=fig.add_subplot(1,1,1,projection=proj)
    xll,xur=img_extent[0],img_extent[1]
    yll,yur=img_extent[2],img_extent[3]
    ax.set_xlim((xll,xur))
    ax.set_ylim((yll,yur))
    cmap,norm,vmin,vmax=get_cmap(img_type)
    im=ax.imshow(frames[:,:,0],
          interpolation='nearest',
          origin='lower',
          extent=[xll,xur,yll,yur],
          transform=proj,
          cmap=cmap,norm=norm,vmin=vmin,vmax=vmax);
    
    ax.add_feature(cfeature.STATES)
    #ax.add_feature(cfeature.LAND)
    #ax.add_feature(cfeature.OCEAN)
    #ax.add_feature(cfeature.COASTLINE)
    #ax.add_feature(cfeature.BORDERS )
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.RIVERS)
    if title:
        ax.set_title(title)

    def init():
        return (im,)

    def animate(i):
        im.set_data(frames[:,:,i]);
        return (im,)

    return animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=range(frames.shape[2]), 
                                   interval=interval, blit=True);


def make_ccrs(info):
    """
    Gets cartopy coordinate reference system and image extent for SEVIR events
    
    Paramters
    ---------
    info pandas.Series object
        Any row from the SEVIR CATALOG, or metadata returned from SEVIRGenerator
    
    Returns
    -------
    ccrs - catropy.crs object containing coordinate ref system
    img_extent - image_extent used for imshow
      
    
    """
    # parse the info.proj string
    # e.g. +proj=laea +lat_0=38 +lon_0=-98 +units=m +a=6370997.0 +ellps=sphere
    pjd = {}
    proj_list = info.proj.split()
    for p in proj_list:
        grps = p.strip().split('=')
        key=grps[0].strip('+')
        val=str(grps[1]) if len(grps)==2 else ''
        if _check_num(val):
            val=float(val)
        pjd.update({key:val})
    
    # Create appropriate cartopy object based on proj_dict
    a=pjd.get('a',None)
    b=pjd.get('b',None)
    ellps=pjd.get('ellps','WGS84')
    datum=pjd.get('datum',None)
    globe=Globe(datum=datum,ellipse=ellps,semimajor_axis=a,semiminor_axis=b)
    if ('proj' in pjd) and (pjd['proj']=='laea'):
        ccrs = crs.LambertAzimuthalEqualArea(central_longitude=pjd.get('lon_0',0.0),
                                             central_latitude=pjd.get('lat_0',0.0),
                                             globe=globe)
    else:
        raise NotImplementedError('Projection %s not implemented, please add it' % info.proj)
    
    # use ccrs to compute image extent
    x1,y1=ccrs.transform_point(info.llcrnrlon,info.llcrnrlat,crs.Geodetic())
    x2,y2=ccrs.transform_point(info.urcrnrlon,info.urcrnrlat,crs.Geodetic())
    img_extent=(x1,x2,y1,y2)
    
    return ccrs,img_extent

def _check_num(s):
    """
    Checks if string is a number
    """
    try:
        float(s)
        return True
    except ValueError:
        return False


    
    
    
    