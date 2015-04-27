import numpy
import pylab
import argparse
import scipy.optimize as optimize
import re
def fitfunc(p,x):
	return p[0] + p[1]*numpy.sin(2.0*numpy.pi*(x+0.1414))

def fitSin(x,y,e):
	ampGuess = 0.5*(y.max()-y.min())
	pinit = [y.min()+ampGuess/2.,ampGuess]
	errfunc = lambda p, x, y, e: (y-fitfunc(p,x))/e
	out     = optimize.leastsq(errfunc,pinit,args=(x,y,e))
	return out

def parse_args():	
	parser = argparse.ArgumentParser()
	parser.add_argument('files', metavar='N', nargs='+', \
		help='files to plot')
	args = parser.parse_args()	
	return args.files

def plotFit(file):
	 data = numpy.loadtxt(file)
	 x = data[:,0]
	 y = data[:,1]
	 e = data[:,2]
	 p,success=fitSin(x,y,e)
	 pylab.errorbar(x%1,y,yerr=e,fmt='.')
	 pylab.errorbar(1+x%1,y,yerr=e,fmt='.')
	 xt = numpy.linspace(0.0,2.0,300)
	 pylab.plot(xt,fitfunc(p,xt))
	
if __name__ == "__main__":
	
	wav = []
	ampl = []
	for file in parse_args():
		data = numpy.loadtxt(file)
		ofile = re.sub('.dat','_fit.pdf',file)
		m = re.search('(\d*)_(\d*).dat',file)
		assert m
		wstart = int(m.group(1))
		wend   = int(m.group(1))
		w      = 0.5*(wstart+wend)
		x = data[:,0]
		y = data[:,1]
		e = data[:,2]
		#remove 2nd order fit
		poly=numpy.poly1d(numpy.polyfit(x,y,2))
		y = y-poly(x)
		p,success=fitSin(x,y,e)
		wav.append(w)
		ampl.append(p[1])
		
	

	pylab.plot(wav,ampl)

	numpy.savetxt('amplSpectrum.txt',numpy.column_stack( (wav,ampl) ))
	pylab.show()
# 		pylab.errorbar(x%1,y,yerr=e,fmt='.')
# 		pylab.errorbar(1+x%1,y,yerr=e,fmt='.')
# 		xt = numpy.linspace(0.0,2.0,300)
# 		pylab.plot(xt,fitfunc(p,xt))
# 		pylab.savefig(ofile)
