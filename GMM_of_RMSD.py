import MDAnalysis as mda
import numpy as np
from MDAnalysis.analysis import align
from MDAnalysis.analysis.rms import rmsd
import time
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
#from sklearn import mixture

#perform an rmsd calculation at each frame of the trajectory
#bin/append the frame to a new .xtc depending on the rmsd value.

#for the first part of this script you should bin the rmsd's and plot a histogram from them.
#This way you'll be able to see if you can reproduce the rmsd-dist.xvg distributions
#obtained in gmx cluster.

rmsd_hist=[]

for i in range(1,3):
    print(i)
    number = str(i)
    u=mda.Universe('../../data/r1_prot_lig_fit.gro', f'../../data/r{number}_prot_lig_fit.xtc')
    #print(u)
    prot=u.select_atoms("resname LIG")

    ref=mda.Universe('../../data/r1_prot_lig_fit.gro')
    for ts in u.trajectory:
        #The below 'align.alignto' feature is commented out and should only be used if you haven't
        #processed your trajectory beforehand to align frames by CA. WARNING - if you use the below
        #alignto feature, the speed of the rmsd calculation will decrease by ~70 fold...
        #it is therefore HIGHLY recommended to preprocess your trajectories before running this script...
        #align.alignto(u, ref, select="protein and name CA", weights="mass")
        reflig=ref.select_atoms('resname LIG')
        moblig=u.select_atoms('resname LIG')
        ligand_rmsd=rmsd(moblig.positions,reflig.positions)
        rmsd_hist.append(ligand_rmsd)
        #print(ts.time/1000)

#reshape the histogram so that it is amenable to GMM
x = np.array(rmsd_hist).reshape(-1,1)

#the following chunk of code is to determine how many gaussians (n_components) are neccessary
n_estimators = np.arange(1,10)
clfs = [GaussianMixture(n,max_iter=500).fit(x) for n in n_estimators]
bics= [clf.bic(x) for clf in clfs]
print(bics)

#find the lowest bayesian information criterion (bics) of the 10 estimates and its corrosponding n_components:
j=1
for i in bics:
    if i==min(bics):
        optimal_component_n=j
    else:
        j=j+1

print('the optimal number of n_components is: ' + str(optimal_component_n))

###This is the actual GMM procedure itself. In it, you have automoatically input the value of the number of
###gaussians required to minimise the BICS.

n_components=optimal_component_n
clf= GaussianMixture(n_components=n_components, max_iter=1000, random_state = 10).fit(x)
xpdf = np.linspace(-10,20,1000).reshape((-1,1))
density = np.exp(clf.score_samples(xpdf))
plt.hist(x, bins = 80, density = True, alpha=0.5)

#where our means are for each the n_components
print('The means of each n_component are : '+ str(clf.means_))

#the width of each respective gaussian
print('The covariances of each gaussian are: '+ str(clf.covariances_))

#how much each of these gaussian components contribute to the full density
print('The weights of each gaussian are: '+ str(clf.weights_))

####plotting the probability density functions of individual gaussian components####
#for i in range(clf.n_components):
#    pdf=clf.weights_[i] * stats.norm(clf.means_[i, 0], np.sqrt(clf.covariances_[i, 0])).pdf(xpdf)
#    plt.fill(xpdf, pdf, facecolor='gray',edgecolor='none', alpha=0.3)

####plotting####
#plt.plot(xpdf, density, '-r')
#plt.xlim(0, 7)
#plt.savefig('new_GMM.png')

### Extract and rank ketamine poses that corrospond to the peaks obtained from the new_GMM ###
### Rank each pose according to it's associated GMM weight ###

##the next block of code is very similar to the code you used for obtaining the histograms
#only that this time you will write out to seperate trajectories the ligand coordinates
#that are within +/- the covariance of it's mean

for i in clf.means_:
    covar_number=0
    covar=clf.covariances_[int(covar_number)]
    for j in range(1,3):
        print(j)
        number = str(j)
        u=mda.Universe('../../data/r1_prot_lig_fit.gro', f'../../data/r{number}_prot_lig_fit.xtc')
        ref=mda.Universe('../../data/r1_prot_lig_fit.gro')
        with mda.Writer(f"mean_{i}.xtc") as w:
            for ts in u.trajectory:
                #The below 'align.alignto' feature is commented out and should only be used if you haven't
                #processed your trajectory beforehand to align frames by CA. WARNING - if you use the below
                #alignto feature, the speed of the rmsd calculation will decrease by ~70 fold...
                #it is therefore HIGHLY recommended to preprocess your trajectories before running this script...
                #align.alignto(u, ref, select="protein and name CA", weights="mass")
                reflig=ref.select_atoms('resname LIG')
                moblig=u.select_atoms('resname LIG')
                ligand_rmsd=rmsd(moblig.positions,reflig.positions)

                #if ligand_rmsd is within the covariance of a particular mean, write out those coorinates to a specific
                #trajectory file. The question is however, with this particular python script format,
                #will looping over individual trajectories overwrite this file?
                k=0
                if ligand_rmsd > (i - covar) and ligand_rmsd < (i + covar):
                    w.write(u.select_atoms("resname LIG"))
    covar_number=covar_number+1
