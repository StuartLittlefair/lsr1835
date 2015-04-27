import  os
import pylab
from scipy.interpolate import interp1d
import re
from scipy.optimize import leastsq as lsq
import itertools
import numpy
import pyfits
import emcee
import scipy.optimize as opt
from mcmc_utils import *
import argparse

# import trm.subs
light = 299792458
planck = 6.626068e-34
boltzmann = 1.3806503e-23

def getY(x,y,xgoal):
    func = interp1d(x,y,kind='linear')
    return func(xgoal)

def bb(wave,temp):
    # surface flux of BB in mJy
    wave_m = wave*1.0e-10
    nu = light/wave_m
    fac   = 2*planck*numpy.pi*nu**3 / light / light
    denom = numpy.exp(planck*nu/boltzmann/temp) - 1
    fnu_wm2 = fac/denom
    #convert to mJy
    return fnu_wm2*1e29
    
def chisq(y,yf,e):
    return numpy.sum( ((y-yf)/e)**2 )

def rebin(a,factor):
    out=[]
    for i in range(0,len(a),factor):
        out.append(a[i:i+factor].mean())
    return numpy.array(out)
    
def loadData():
    x,y,_ = numpy.loadtxt('./keck/differenceSpectrum.txt').T
    mask = x<9000
    return x[mask], y[mask]

def modelName(teff,g,cloud):
        str = "./models/lte%2d-%.1f-0.0.AMES-" % (teff/100,g)
        if not cloud:
                str += "cond.7_out"
        else:
                str += "dusty.7_out"
        if os.path.exists(str):
                return str      
        else:
                raise Exception("file " + str + " does not exist")

def loadModel(teff,g,cloud):
    file = modelName(teff,g,cloud)
    mx,my = numpy.loadtxt(file).T
    #model flux in ergs/sec/cm2/AA
    
    #convert to W/m2/m
    my *= 1.0e7
    
    #convert to Fnu (W/m2/Hz)
    #mx in 
    my = my*mx*mx*1e-20/light
    
    #convert to mJy
    my = my*1e29
    return mx, my
                  
def fluxCalibrateTemplate(tx,ty,teff,logg,cloud):
    mx, my = loadModel(teff,logg,cloud)
    model = numpy.array([getY(mx,my,xval) for xval in tx])
    scaleFac = model.mean()/ty.mean()
    return scaleFac
    
def loadTemplate(file):
    hdu = pyfits.open(file)
    hdr = hdu[0].header
    flux = hdu[0].data[1,:]
    wav0 = hdr['CRVAL1']
    dWav = hdr['CD1_1']
    npix = flux.copy().size
    wave = wav0 + dWav*numpy.arange(npix)
    lam  = wave*1e-10
    #convert flux to fnu
    return wave, flux*lam*lam/3e8
    
def norm(wav,flux):
    #scale spectra to match at top of TiO bandhead at 7580AA
    mask=(wav<7590)&(wav>7560)
    mean = flux[mask].mean()
    return mean
    
def scale(data,model):
    guess = numpy.abs(data.mean()) / numpy.abs(model.mean()) 
    errfunc = lambda scale, data, model: data-scale*model
    out = lsq(errfunc,guess,args=(data,model),full_output=1)
    return out[0]
    
def model(p,x):
    # add a small amount of BB and block corresponding photosphere
    # high state = (1-cover)*I + cover*S
    # low state  = I
    # difference Spectrum = cover*(S-I)
    # scale by (R/d)^2 to match spectrum of BD
    bbtemp  = p[0]
    # fac combines covering fraction of spot with (R/d)^2 factor
    fac  = p[1]
    spot    = numpy.array([bb(xval,bbtemp) for xval in x])
    return  fac*(spot-immac) 

def model2(p,x):
    teff = p[0]
    fac  = p[1]
    mx, my = loadModel(2600,5.5,False)
    spot = numpy.array([getY(mx,my,xval) for xval in x])
    return fac*(spot-immac)
    
def errorbars(p,x):
    #errorbars are part of model. Equal to p[3] everywhere except emission line regions
    err = numpy.array([p[2] for xval in x])
    m1 = (x>4085) & (x<4115)
    m2 = (x>4290) & (x<4320)
    m3 = (x>4845) & (x<4875)
    m4 = (x>5560) & (x<5590)
    m5 = (x>6545) & (x<6595)
    m6 = (x>7575) & (x<7680)
    m7 = (x>8165) & (x<8220)
    m8 = (x>8200) 
    mask = (m1) | (m2) | (m3) | (m4) | (m5) | (m6) | (m7) | (m8) 
    err[mask] = 1e72
    return err, mask 

def ln_likelihood(p):
    yfit = model(p,x)
    res = y-yfit
    err,_ = errorbars(p,x)
    return -0.5*(np.sum(np.log(2.0*np.pi*err**2)) + chisq(y,yfit,err))
    
def ln_prior(p):
    ln_p = 0.0
    prior = Prior('uniform',500,20000)
    ln_p += prior.ln_prob(p[0])

    # scale factor p[1] (uniform in log implies P(v) ~ 1/v)
    prior = Prior('log_uniform',1.0e-28,1.0e-2)
    # for a 1RJup BD at 10pc, this allows covering fractions from ? to ?
    # obviously, if you have radius constraints it's a good idea to use them as a check!
    ln_p += prior.ln_prob(p[1])

    # average error on focus (uniform in log implies P(v) ~ 1/v)
    prior = Prior('log_uniform',0.001,20.0)
    ln_p += prior.ln_prob(p[2])
    return ln_p

def ln_posterior(p):
    lnp = ln_prior(p)
    if numpy.isfinite(lnp):
        return ln_likelihood(p)+ln_prior(p)
    else:
        return lnp
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Fit difference spectrum')
    parser.add_argument('--fit','-f',action='store_true',dest='fit')
    args = parser.parse_args()

    x,y=loadData()
    y = rebin(y,3)
    x = rebin(x,3)
        
    # load bochanski template in, scaling so flux levels match PHOENIX models
    # teff, spectral type conversion from Stephens et al (2009) (M8 gives good fit)
    baseX, baseY = loadTemplate('templates/m8.all.na.k.fits')  
    scaleFac     = fluxCalibrateTemplate(baseX,baseY,2600,5.5,True)
    immac = scaleFac*numpy.array([getY(baseX,baseY,xval) for xval in x]) 
    
    p0 = [2180,1.64e-21,0.03]
    # model params: BB temp, scaling factor (combines (R/d)^2 and surface area, so 
    # divide by (R/d)^2 to get covering fraction), uncertainty
    # errors included in model. Equal to p[3] everywhere except in emission line regions
    p0 = numpy.array(p0)

    ndim = 3
    nwalkers = 24
    nthreads = 4
    
    if args.fit:
        sampler = emcee.EnsembleSampler(nwalkers,ndim,ln_posterior,threads=nthreads)
        p0 = emcee.utils.sample_ball(p0,0.01*p0,size=nwalkers)

        #burn-in
        nburn = 1000
        pos, lnprob, state = sampler.run_mcmc(p0,nburn)
        print 'burn-in done'

        #production chain
        sampler.reset()
        nprod = 1000
        sampler = run_mcmc_save(sampler,pos,nprod,state,"chain.txt")
        chain  = flatchain(sampler.chain,ndim,thin=4)
        nameList = [r'Tb',r'$\epsilon / 10^{-21}$',r'$\sigma$']
        pars = []
        for i in range(ndim):
            par = chain[:,i]
            lolim,best,uplim = np.percentile(par,[16,50,84])
            print "%s = %e +%e -%e" % (nameList[i],best,uplim-best,best-lolim)
            pars.append(best)
        chain[:,1] /= 1e-21
        fig = thumbPlot(chain,nameList)
        pylab.show()
    
    else:
        pars = p0
    
    diffSpec = model(pars,x)
    err,mask = errorbars(pars,x)
    print scale(y,diffSpec), chisq(y,diffSpec,err), len(y)-3
    if args.fit:
        print 'Mean acceptance fraction = %f' % np.mean(sampler.acceptance_fraction)
        print 'Auto-correlation times: ', sampler.get_autocorr_time()
    
    pylab.subplot(211)
    pylab.plot(x[x<6000],y[x<6000],label='data')
    pylab.plot(x[x<6000],diffSpec[x<6000],label='aurora',lw=2)
    pylab.subplot(212)
    pylab.plot(x[x>6000],y[x>6000],label='data')
    pylab.plot(x[x>6000],diffSpec[x>6000],label='aurora',lw=2)    
    pylab.show()

 