#lighting the hearth
import matplotlib.pyplot
import numpy
import pandas
import seaborn 
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS
from mpl_toolkits import mplot3d
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import Isomap
from matplotlib import offsetbox


#functions
def drawVector(v0,v1,ax=None):
	ax = ax or matplotlib.pyplot.gca()
	arrowprops=dict(arrowstyle='->',linewidth=2,shrinkA=0,shrinkB=0, color='black')
	ax.annotate('',v1,v0,arrowprops=arrowprops)
	
def plotDigits(data):
    fig, axes = matplotlib.pyplot.subplots(4, 10, figsize=(10, 4),
                             subplot_kw={'xticks':[], 'yticks':[]},
                             gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(data[i].reshape(8, 8),cmap='binary', interpolation='nearest',clim=(0, 16))
      
def plotComponents(data, model, images=None, ax=None,
                    thumb_frac=0.05, cmap='gray'):
    ax = ax or matplotlib.pyplot.gca()
    
    Proj = model.fit_transform(data)
    ax.plot(Proj[:, 0], Proj[:, 1], '.k')
    
    if images is not None:
        min_dist_2 = (thumb_frac * max(Proj.max(0) - Proj.min(0))) ** 2
        shown_images = numpy.array([2 * Proj.max(0)])
        for i in range(data.shape[0]):
            dist = numpy.sum((Proj[i] - shown_images) ** 2, 1)
            if numpy.min(dist) < min_dist_2:
                # don't show points that are too close
                continue
            shown_images = numpy.vstack([shown_images, Proj[i]])
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(images[i], cmap=cmap),
                                      Proj[i])
            ax.add_artist(imagebox)
            

######			PCA			#####
'''
Digits = load_digits()
seaborn.set()

R = numpy.random.RandomState(1)
X = numpy.dot(R.rand(2,2),R.randn(2,200)).T

pca = PCA(n_components=2)
pca.fit(X)

print(pca.components_)
print(pca.explained_variance_)

matplotlib.pyplot.figure(1)
matplotlib.pyplot.scatter(X[:,0],X[:,1])

for length, vector in zip(pca.explained_variance_,pca.components_):
	v = vector*3*numpy.sqrt(length)
	drawVector(pca.mean_,pca.mean_+v)

matplotlib.pyplot.axis('equal');

pca = PCA(n_components=1)
pca.fit(X)
X_pca = pca.transform(X)
print("original shape:   ", X.shape)
print("transformed shape:", X_pca.shape)


matplotlib.pyplot.figure(2)
X_new = pca.inverse_transform(X_pca)
matplotlib.pyplot.scatter(X[:, 0], X[:, 1], alpha=0.2)
matplotlib.pyplot.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8)
matplotlib.pyplot.axis('equal');

pca = PCA(2)  # project from 64 to 2 dimensions
ProjectedDigits = pca.fit_transform(Digits.data)
print(Digits.data.shape)
print(ProjectedDigits.shape)


matplotlib.pyplot.figure(3)
matplotlib.pyplot.scatter(ProjectedDigits[:, 0], ProjectedDigits[:, 1],
            c=Digits.target, edgecolor='none', alpha=0.5,
            cmap=matplotlib.pyplot.cm.get_cmap('rainbow', 10))
matplotlib.pyplot.xlabel('component 1')
matplotlib.pyplot.ylabel('component 2')
matplotlib.pyplot.colorbar();


pca = PCA().fit(Digits.data)

matplotlib.pyplot.figure(4)
matplotlib.pyplot.plot(numpy.cumsum(pca.explained_variance_ratio_))
matplotlib.pyplot.xlabel('number of components')
matplotlib.pyplot.ylabel('cumulative explained variance');


matplotlib.pyplot.figure(5)
plotDigits(Digits.data)


numpy.random.seed(42)
NoisyDigits = numpy.random.normal(Digits.data, 4)
plotDigits(NoisyDigits.data)

pca = PCA(0.50).fit(NoisyDigits)
pca.n_components_

Components = pca.transform(NoisyDigits)
Filtered = pca.inverse_transform(Components)
plotDigits(Filtered)
'''
######   		  Manifold				#####
Digits = load_digits()

fig, axes = matplotlib.pyplot.subplots(4, 10, figsize=(10, 4),subplot_kw={'xticks':[], 'yticks':[]},gridspec_kw=dict(hspace=0.1, wspace=0.1))

model = Isomap(n_components=2)
Proj = model.fit_transform(Digits.data)

for i, ax in enumerate(axes.flat):
        ax.imshow(Digits.data[i].reshape(8, 8),cmap='binary', interpolation='nearest',clim=(0, 16))
'''
matplotlib.pyplot.figure(1)
fig, ax = matplotlib.pyplot.subplots(figsize=(10, 10))
plotComponents(Digits.data,model=Isomap(n_components=2),images=Digits.images[:, ::2, ::2])
'''
data = Digits.data[:]
target = Digits.target[:]
model = Isomap(n_components=2)
Proj = model.fit_transform(data)

matplotlib.pyplot.figure(2)
matplotlib.pyplot.scatter(Proj[:, 0], Proj[:, 1], c=target, cmap=matplotlib.pyplot.cm.get_cmap('jet', 10))
matplotlib.pyplot.colorbar(ticks=range(10))
matplotlib.pyplot.clim(-0.5, 9.5);
           
data = Digits.data[Digits.target == 1][:]

fig, ax = matplotlib.pyplot.subplots(figsize=(10, 10))
model = Isomap(n_neighbors=5, n_components=2, eigen_solver='dense')
plotComponents(data, model, images=data.reshape((-1, 8, 8)),
                ax=ax, thumb_frac=0.05, cmap='gray_r')
                
                  
#show plots
matplotlib.pyplot.show()
