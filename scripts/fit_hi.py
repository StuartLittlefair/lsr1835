from fitDiffSpec import *
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
sns.set_style('whitegrid', {"axes.edgecolor": "0.05"})

def bb(wave,temp):
    # surface flux of BB in mJy
    wave_m = wave*1.0e-10
    nu = light/wave_m
    fac   = 2*planck*numpy.pi*nu**3 / light / light
    denom = numpy.exp(planck*nu/boltzmann/temp) - 1
    fnu_wm2 = fac/denom
    #convert to mJy
    return fnu_wm2*1e29

def calc_hi(fh, N):
    return N*(fh*spot + (1-fh)*immac)
    
def ln_likelihood(p,x,yhi):
    fh, N, _ = p
    yfit = calc_hi(fh,N)
    err,_ = errorbars(p,x)
    return -0.5*(np.sum(np.log(2.0*np.pi*err**2)) + chisq(yhi,yfit,err)) 

def ln_prior(p):
    ln_p = 0.0
    # fh - unconstrained
    # defined from Halpha variability - assume that lo state has no halpha
    prior = Prior('uniform',0.002,0.5)
    ln_p += prior.ln_prob(p[0])
    # N
    prior = Prior('log_uniform',1.0e-21,1.0e-15)
    ln_p += prior.ln_prob(p[1])
    # error bars
    prior = Prior('log_uniform',0.001,20.0)
    ln_p += prior.ln_prob(p[2])
    return ln_p    

def ln_posterior(p,x,yhi):
    lnp = ln_prior(p)
    if numpy.isfinite(lnp):
        return ln_likelihood(p,x,yhi)+ln_prior(p)
    else:
        return lnp
 
# read in high state spectra
x,yhi = np.loadtxt('keck/hiState.txt').T
        
# load bochanski template in, scaling so flux levels match PHOENIX models
# teff, spectral type conversion from Stephens et al (2009) (M8 gives good fit)
baseX, baseY = loadTemplate('templates/m8.all.na.k.fits')  
scaleFac     = fluxCalibrateTemplate(baseX,baseY,2600,5.5,True)
immac = scaleFac*numpy.array([getY(baseX,baseY,xval) for xval in x])
print 'template computed'

# model params: BB temp, scaling factor (combines (R/d)^2 and surface area, so 
# divide by (R/d)^2 to get covering fraction), errorbars
p0 = np.array([2180,1.64e-21,0.03])
bbtemp = p0[0]
spot    = numpy.array([bb(xval,bbtemp) for xval in x])

# find factors needed
from astropy import constants as c, units as u
R = 1.1 * c.R_jup # from models
d = 5.7 * u.pc # distance from H08
N = ((R/d).decompose())**2
# this value of N is a guideline. It can also be constrained from fitting the high and low state spectra to the template. 
# see later

ndim = 3
nwalkers = 24
nthreads = 4

sampler = emcee.EnsembleSampler(nwalkers,ndim,ln_posterior,args=[x,yhi],threads=nthreads)

# starting guess 
p0 = numpy.array([0.03,N,0.03])
p0 = emcee.utils.sample_ball(p0,0.01*p0,size=nwalkers)


# burn-in
nburn = 1000
pos, lnprob, state = run_burnin(sampler,p0,nburn)

# production
sampler.reset()
nprod = 5000
sampler = run_mcmc_save(sampler,pos,nprod,state,'chain_hi.txt')

chain  = flatchain(sampler.chain,ndim,thin=4)

#chain = readchain('chain_hi.txt')
#chain = flatchain(chain,ndim+1)[:,:-1]
#print chain.shape

nameList = [r'$f_h$',r'$N/10^{20}$',r'$\sigma$']
chain[:,1] *= 1e20
fig = thumbPlot(chain,nameList)
plt.show()

chain[:,1] /= 1e20
fh,N,e= chain.mean(axis=0)
efh, eN, ee = chain.std(axis=0)
print fh, efh
print N, eN
print e, ee
histate = calc_hi(fh,N)

# normalise hi state at 6000-6500 AA (correction for (R/d)**2)
m_mask = (x>6000) & (x<6500)
hiScale = histate[m_mask].mean()
hiScale = yhi[m_mask].mean()/hiScale
print hiScale
histate *= hiScale

#plot
gs  = gridspec.GridSpec(2,1,height_ratios=[2,1])
gs.update(hspace=0.0)
ax_main = plt.subplot(gs[0,0])
ax_res  = plt.subplot(gs[1,0],sharex=ax_main)

# main plot
ax_main.step(x,histate,color='r',alpha=0.5,label='model',lw=0.8)
ax_main.step(x,yhi,color='k',alpha=0.5,label='data',lw=0.8)
ax_main.semilogy()
ax_main.set_ylim((0.01,100))
#ax_main.legend()

# residuals
residuals = yhi - histate
ax_res.step(x,residuals,lw=0.8,alpha=0.7,color='k')
#  IMPORTANT: set Y-axis to same scale as features in difference spectrum
ax_res.set_ylim((-0.6,0.6))

# tidy up axes
plt.setp(ax_main.get_xticklabels(),visible=False)
ax_res.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4,prune='both'))

#labels
ax_res.set_xlabel(r'$\mathrm{Wavelength} \, (\AA)$')
ax_main.set_ylabel(r'$\log_{10} \, \mathrm{Flux\ Density\, (mJy)}$')
ax_res.set_ylabel(r'$\mathrm{Residuals \, (mJy)}$')
plt.savefig('histate_fit.pdf')
plt.show()