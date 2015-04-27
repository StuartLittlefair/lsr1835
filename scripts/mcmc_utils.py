import numpy as np
import scipy.signal as signal
import scipy.optimize as opt
import scipy.stats as stats
import triangle
from progress import ProgressBar
import scipy.integrate as intg
import warnings
from matplotlib import pyplot as plt
import os

TINY = -np.inf

class Prior(object):
    '''a class to represent a prior on a parameter, which makes calculating 
    prior log-probability easier.

    Priors can be of five types: gauss, gaussPos, uniform, log_uniform and mod_jeff

    gauss is a Gaussian distribution, and is useful for parameters with
    existing constraints in the literature
    gaussPos is like gauss but enforces positivity
    Gaussian priors are initialised as Prior('gauss',mean,stdDev)

    uniform is a uniform prior, initialised like Prior('uniform',low_limit,high_limit)
    uniform priors are useful because they are 'uninformative'

    log_uniform priors have constant probability in log-space. They are the uninformative prior
    for 'scale-factors', such as error bars (look up Jeffreys prior for more info)
    
    mod_jeff is a modified jeffries prior - see Gregory et al 2007
    they are useful when you have a large uncertainty in the parameter value, so
    a jeffreys prior is appropriate, but the range of allowed values starts at 0
    
    they have two parameters, p0 and pmax.
    they act as a jeffrey's prior about p0, and uniform below p0. typically
    set p0=noise level
    '''
    def __init__(self,type,p1,p2):
        assert type in ['gauss','gaussPos','uniform','log_uniform','mod_jeff']
        self.type = type
        self.p1   = p1
        self.p2   = p2
        if type == 'log_uniform' and self.p1 < 1.0e-30:
            warnings.warn('lower limit on log_uniform prior rescaled from %f to 1.0e-30' % self.p1)
            self.p1 = 1.0e-30
        if type == 'log_uniform':
            self.normalise = 1.0
            self.normalise = np.fabs(intg.quad(self.ln_prob,self.p1,self.p2)[0])
        if type == 'mod_jeff':
            self.normalise = np.log((self.p1+self.p2)/self.p1)

    def ln_prob(self,val):
        if self.type == 'gauss':	
            return np.log( stats.norm(scale=self.p2,loc=self.p1).pdf(val) )
        elif self.type == 'gaussPos':
            if val <= 0.0:
                return TINY
            else:
                return np.log( stats.norm(scale=self.p2,loc=self.p1).pdf(val) )
        elif self.type == 'uniform':
            if (val > self.p1) and (val < self.p2):
                return np.log(1.0/np.abs(self.p1-self.p2))
            else:	
                return TINY
        elif self.type == 'log_uniform':
            if (val > self.p1) and (val < self.p2):
                return np.log(1.0 / self.normalise / val)
            else:	
                return TINY
        elif self.type == 'mod_jeff':
            if (val > 0) and (val < self.p2):
                return np.log(1.0/ self.normalise / (val+self.p1))
            else:
                return TINY
		 	
class Param(object):
	'''A Param needs a starting value, a current value, and a prior
	and a flag to state whether is should vary'''
	def __init__(self,startVal,prior,isVar=True):
		self.startVal = startVal
		self.prior    = prior
		self.currVal  = startVal
		self.isVar    = isVar
		
	@classmethod
	def fromString(cls,parString):
	    fields = parString.split()
	    val = float(fields[0])
	    priorType = fields[1].strip()
	    priorP1   = float(fields[2])
	    priorP2   = float(fields[3])
	    if len(fields) == 5:
	        isVar = bool(fields[4])
	    else:
	        isVar = True
	    return 	cls(val, Prior(priorType,priorP1,priorP2), isVar)
	
def fracWithin(pdf,val):
	return pdf[pdf>=val].sum()

def thumbPlot(chain,labels,**kwargs):
    fig = triangle.corner(chain,labels=labels,**kwargs)
    return fig

def scatterWalkers(pos0,percentScatter):
    warnings.warn('scatterWalkers decprecated: use emcee.utils.sample_ball instead')
    nwalkers = pos0.shape[0]
    npars    = pos0.shape[1]
    scatter = np.array([np.random.normal(size=npars) for i in xrange(nwalkers)])
    return pos0 + percentScatter*pos0*scatter/100.0

def run_burnin(sampler,startPos,nSteps,storechain=False):
    iStep = 0
    bar = ProgressBar()
    for pos, prob, state in sampler.sample(startPos,iterations=nSteps,storechain=storechain):
        bar.render(int(100*iStep/nSteps),'running Burn In')
        iStep += 1
    return pos, prob, state
    
def run_mcmc_save(sampler,startPos,nSteps,rState,file,**kwargs):
    '''runs and MCMC chain with emcee, and saves steps to a file'''
    #open chain save file
    if file:
        f = open(file,"w")
        f.close()
    iStep = 0
    bar = ProgressBar()
    for pos, prob, state in sampler.sample(startPos,iterations=nSteps,rstate0=rState,storechain=True,**kwargs):
        if file:
            f = open(file,"a")
        bar.render(int(100*iStep/nSteps),'running MCMC')
        iStep += 1
        for k in range(pos.shape[0]):
            # loop over all walkers and append to file
            thisPos = pos[k]
            thisProb = prob[k]
            if file:
                f.write("{0:4d} {1:s} {2:f}\n".format(k," ".join(map(str,thisPos)),thisProb ))
        if file:        
            f.close()
    return sampler
    
def run_ptmcmc_save(sampler,startPos,nSteps,file,**kwargs):
    '''runs PT MCMC and saves zero temperature chain to file'''
    if not os.path.exists(file):
        f = open(file,"w")
        f.close()

    iStep = 0    
    bar = ProgressBar()
    for pos, prob, like in sampler.sample(startPos,iterations=nSteps,storechain=True,**kwargs):
        bar.render(int(100*iStep/nSteps),'running MCMC')
        iStep += 1
        f = open(file,"a")
        # pos is shape (ntemps, nwalkers, npars)
        # prob is shape (ntemps, nwalkers)
        # loop over all walkers for zero temp and append to file
        zpos = pos[0,...]
        zprob = prob[0,...]
        for k in range(zpos.shape[0]):
            thisPos = zpos[k]
            thisProb = zprob[k]
            f.write("{0:4d} {1:s} {2:f}\n".format(k," ".join(map(str,thisPos)),thisProb ))
    f.close()
    return sampler    
    
def flatchain(chain,npars,nskip=0,thin=1):
    '''flattens a chain (i.e collects results from all walkers), 
    with options to skip the first nskip parameters, and thin the chain
    by only retrieving a point every thin steps - thinning can be useful when
    the steps of the chain are highly correlated'''
    return chain[:,nskip::thin,:].reshape((-1,npars))
    
def readchain(file,nskip=0,thin=1):
    data = np.loadtxt(file)
    nwalkers=int(data[:,0].max()+1)
    nprod = int(data.shape[0]/nwalkers)
    npars = data.shape[1]-1 # first is walker ID, last is ln_prob
    chain = np.reshape(data[:,1:],(nwalkers,nprod,npars))
    return chain

def plotchains(chain,npar,alpha=0.2):
    nwalkers, nsteps, npars = chain.shape
    fig = plt.figure()
    for i in range(nwalkers):
        plt.plot(chain[i,:,npar],alpha=alpha,color='k')
    return fig

def ln_marginal_likelihood(params, lnp):
    '''given a flattened chain which consists of a series
    of samples from the parameter posterior distributions,
    and another array which is ln_prob (posterior) for these
    parameters, estimate the marginal likelihood of this model, 
    allowing for model selection.
    
    Such a chain is created by reading in the output file of 
    an MCMC run, and running flatchain on it.
    
    Uses the method of Chib & Jeliazkov (2001) as outlined
    by Haywood et al 2014
    
    '''
    raise Exception('This routine is incorrect and should not be used until fixed. See the emcee docs for the Parallel Tempering sampler instead')
    # maximum likelihood estimate
    loc_best = lnp.argmin()
    log_max_likelihood = lnp[loc_best]
    best = params[loc_best]
    # standard deviations
    sigmas = params.std(axis=0)
    
    # now for the magic
    # at each step in the chain, add up 0.5*((val-best)/sigma)**2 for all params
    term = 0.5*((params-best)/sigmas)**2
    term = term.sum(axis=1)
    
    # top term in posterior_ordinate
    numerator = np.sum(np.exp(term))
    denominator = np.sum(lnp/log_max_likelihood)
    posterior_ordinate = numerator/denominator
    
    log_marginal_likelihood = log_max_likelihood - np.log(posterior_ordinate)
    return log_marginal_likelihood
    
def rebin(xbins,x,y,e=None,weighted=True,errors_from_rms=False):
    digitized = np.digitize(x,xbins)
    xbin = []
    ybin = []
    ebin = []
    for i in range(0,len(xbins)):
            bin_y_vals = y[digitized == i]
            bin_x_vals = x[digitized == i]
            if e is not None:
                bin_e_vals = e[digitized == i]
            if weighted:
                if e is None:
                    raise Exception('Cannot compute weighted mean without errors')
                weights = 1.0/bin_e_vals**2
                xbin.append( np.sum(weights*bin_x_vals) / np.sum(weights) )
                ybin.append( np.sum(weights*bin_y_vals) / np.sum(weights) )
                if errors_from_rms:
                    ebin.append(np.std(bin_y_vals))
                else:
                    ebin.append( np.sqrt(1.0/np.sum(weights) ) )
            else:
                xbin.append(bin_x_vals.mean())
                ybin.append(bin_y_vals.mean())
                if errors_from_rms:
                    ebin.append(np.std(bin_y_vals))
                else:
                    ebin.append(np.sqrt(np.sum(bin_e_vals**2)) / len(bin_e_vals))
    xbin = np.array(xbin)
    ybin = np.array(ybin)
    ebin = np.array(ebin)
    return (xbin,ybin,ebin)