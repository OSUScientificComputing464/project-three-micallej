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
       
def makeHello(N=1000, rseed=42):
    # Make a plot with "HELLO" text; save as PNG
    fig, ax = matplotlib.pyplot.subplots(figsize=(4, 1))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis('off')
    ax.text(0.5, 0.4, 'HELLO', va='center', ha='center', weight='bold', size=85)
    fig.savefig('hello.png')
    matplotlib.pyplot.close(fig)
    
    # Open this PNG and draw random points from it
    from matplotlib.image import imread
    data = imread('hello.png')[::-1, :, 0].T
    rng = numpy.random.RandomState(rseed)
    X = rng.rand(4 * N, 2)
    i, j = (X * data.shape).astype(int).T
    mask = (data[i, j] < 1)
    X = X[mask]
    X[:, 0] *= (data.shape[0] / data.shape[1])
    X = X[:N]
    return X[numpy.argsort(X[:, 0])]
    
def rotate(x, angle):
	theta = numpy.deg2rad(angle)
	r = [[numpy.cos(theta), numpy.sin(theta)],[-numpy.sin(theta), numpy.cos(theta)]]
	return numpy.dot(x, r)
	
def randomProjection(X, dimension=3, rseed=42):
    assert dimension >= X.shape[1]
    rng = numpy.random.RandomState(rseed)
    C = rng.randn(dimension, dimension)
    e, V = numpy.linalg.eigh(numpy.dot(C, C.T))
    return numpy.dot(X, V[:X.shape[1]])

def makeHelloSCurve(X):
    t = (X[:, 0] - 2) * 0.75 * numpy.pi
    x = numpy.sin(t)
    y = X[:, 1]
    z = numpy.sign(t) * (numpy.cos(t) - 1)
    return numpy.vstack((x, y, z)).T

   
    
'''
#PCA
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
#Manifold
seaborn.set()
X = makeHello(1000)
colorize = dict(c=X[:,0],cmap=matplotlib.pyplot.get_cmap('rainbow',5))
'''
matplotlib.pyplot.figure(0)
matplotlib.pyplot.scatter(X[:,0],X[:,1], **colorize)
matplotlib.pyplot.axis('equal');
'''
X2 = rotate(X, 20) + 5
'''
matplotlib.pyplot.figure(1)
matplotlib.pyplot.scatter(X2[:, 0], X2[:, 1], **colorize)
matplotlib.pyplot.axis('equal');
'''
D = pairwise_distances(X)
print(D.shape)
'''
matplotlib.pyplot.figure(2)
matplotlib.pyplot.imshow(D, zorder=2, cmap='Blues', interpolation='nearest')
matplotlib.pyplot.colorbar();
'''
D2 = pairwise_distances(X2)
numpy.allclose(D, D2)

model = MDS(n_components=2, dissimilarity='precomputed', random_state=1)
Out = model.fit_transform(D)
'''
matplotlib.pyplot.figure(3)
matplotlib.pyplot.scatter(Out[:, 0], Out[:, 1], **colorize)
matplotlib.pyplot.axis('equal');
'''
X3 = randomProjection(X,3)
print(X3.shape)


ax = matplotlib.pyplot.axes(projection='3d')
ax.scatter3D(X3[:,0], X3[:,1], X3[:,2], **colorize);
ax.view_init(azim=70,elev=50)

model = MDS(n_components=2, random_state=1)

Out3 = model.fit_transform(X3)
'''
matplotlib.pyplot.figure(4)
matplotlib.pyplot.scatter(Out3[:, 0], Out3[:, 1], **colorize)
matplotlib.pyplot.axis('equal');
'''
XS = makeHelloSCurve(X)
'''
ax = matplotlib.pyplot.axes(projection='3d')
ax.scatter3D(XS[:,0], XS[:,1], XS[:,2], **colorize);
model = MDS(n_components=2, random_state=2)
OutS = model.fit_transform(XS)
matplotlib.pyplot.scatter(OutS[:,0], OutS[:,1], **colorize);
matplotlib.pyplot.axis('equal');
'''
model = LocallyLinearEmbedding(n_neighbors=100, n_components=2, method='modified', eigen_solver='dense')

Out = model.fit_transform(XS)
fig, ax = matplotlib.pyplot.subplots()

ax.scatter(Out[:, 0], Out[:, 1], **colorize)
ax.set_ylim(0.15, -0.15);



#show plots
matplotlib.pyplot.show()
