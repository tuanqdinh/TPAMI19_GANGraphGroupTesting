name0 = 'diffwgancom010_1000_10_10.npy'
name1 = 'diffwgancom011_1000_10_10.npy'
d1 = np.load('../../data/gaussians/' + name1)
d0 = np.load('../../data/gaussians/' + nplotme0)
x = np.arange(1, len(d0) + 1)
plt.plot(x, d1, label='L-GAN')
plt.plot(x, d0, label='Vanilla-GAN')
plt.legend()
plt.xlabel('#Modes')
plt.ylabel("Norm2 of W")
plt.title('Compare Norm on signals: 10 coms, 10 modes, 1000 dims')
plt.show()


name_ad0 = 'sync/mesh41_010_2.npy'
sad0 = np.load(name_ad0)
name_cn0 = 'sync/mesh42_010_2.npy'
scn0 = np.load(name_cn0)
name_ad1 = 'sync/mesh41_011_2.npy'
sad1 = np.load(name_ad1)
name_cn1 = 'sync/mesh42_011_2.npy'
scn1 = np.load(name_cn1)
ord = np.arange(len(sad0) + len(scn0))
np.random.shuffle(ord)
# for vannilla
X0 = np.concatenate((sad0, scn0))
y0 = np.zeros(len(sad0) + len(scn0))
y0[:len(sad0)] = 1
y0 = y0[ord]
X0 = X0[ord, :]
X0 = normalize(X0, axis=1)
clf = SVM.SVC()
clf.fit(X0, y0)
pred_ad0 = clf.predict(n_ad)
pred_cn0 = clf.predict(n_cn)
# for lap
X1 = np.concatenate((sad1, scn1))
y1 = np.zeros(len(sad1) + len(scn1))
y1[:len(sad1)] = 1
y1 = y1[ord]
X1 = X1[ord, :]
X1 = normalize(X1, axis=1)
clf = SVM.SVC()
clf.fit(X1, y1)
pred_ad1 = clf.predict(n_ad)
pred_cn1 = clf.predict(n_cn)
