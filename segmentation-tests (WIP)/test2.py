'''
Requeriments:
- Python (We use 3.5).
- Packages: matplotlib, skimage, numpy, mahotas.
'''

import numpy as np
from matplotlib.image import imread
from skimage.color import rgb2gray, label2rgb, gray2rgb
from skimage.filters import threshold_otsu
from skimage.segmentation import slic, mark_boundaries
from matplotlib.pyplot import show, imshow, subplot, figure, title, imsave, suptitle
from skimage.future import graph
from matplotlib import colors
from skimage.draw import ellipse
from skimage.measure import label, compare_mse, compare_ssim, compare_psnr
from mahotas.features import haralick
#conda install -c https://conda.anaconda.org/conda-forge mahotas
from balu.FeatureAnalysis import Bfa_jfisher
from skimage.filters import gaussian
from skimage.morphology import opening, disk, closing
import os
import fnmatch
from scipy.ndimage.morphology import binary_fill_holes
from balu.FeatureExtraction import Bfx_haralick, Bfx_geo, Bfx_basicgeo
from skimage.exposure import histogram
from scipy.special import entr
from scipy.io import savemat, loadmat
from balu.FeatureSelection import Bfs_clean
from balu.Classification import Bcl_structure
from balu.PerformanceEvaluation import Bev_performance, Bev_confusion

def magic(segmentationProcess=True, saveSegmentation=True, featuresProcess=True, trainAndTest=True):
    global path, pathSegmentation, dph2

    if saveSegmentation:
        counter = 0
        if not os.path.exists(pathSegmentation):
            os.makedirs(pathSegmentation)

    if featuresProcess:
        X = []
        Xn = []
        d = []
        imagesNames = []

    def compare_jona(img1, img2):
        c = 0
        for i in range(len(GT)):
            for j in range(len(GT[i])):
                if GT[i][j] == Isegmented[i][j]:
                    c += 1
        return c / (len(GT) * len(GT[0])), 0.58

    def _weight_haralick(graph, src, dst, n):
        """Callback to handle merging nodes by haralick method.
        The method expects that the mean color of `dst` is already computed.
        Parameters
        ----------
        graph : RAG
            The graph under consideration.
        src, dst : int
            The vertices in `graph` to be merged.
        n : int
            A neighbor of `src` or `dst` or both.
        Returns
        -------
        data : dict
            A dictionary with the `"weight"` attribute set as the absolute
            difference of the mean color between node `dst` and `n`.
        """
        s = np.zeros((IOriginal.shape[0:2]), dtype=np.uint8)
        for i in graph.node[dst]['labels']:
            s += L == i

        hdst = haralick(gray2rgb(s) * I)

        s = np.ones((IOriginal.shape[0:2]), dtype=np.uint8)
        for i in graph.node[n]['labels']:
            s += L == i
        hn = haralick(gray2rgb(s) * I)

        #print(hdst)
        #print(hn)

        J = Bfa_jfisher(hdst + hn, np.zeros(len(hdst)) + np.ones(len(hn)))

        #print(J)

        #diff = graph.node[dst]['mean color'] - graph.node[n]['mean color']
        #diff = np.linalg.norm(diff)
        #return {'weight': diff}
        return J

    def _weight_mean_color(graph, src, dst, n):
        """Callback to handle merging nodes by recomputing mean color.
        The method expects that the mean color of `dst` is already computed.
        Parameters
        ----------
        graph : RAG
            The graph under consideration.
        src, dst : int
            The vertices in `graph` to be merged.
        n : int
            A neighbor of `src` or `dst` or both.
        Returns
        -------
        data : dict
            A dictionary with the `"weight"` attribute set as the absolute
            difference of the mean color between node `dst` and `n`.
        """

        diff = graph.node[dst]['mean color'] - graph.node[n]['mean color']
        diff = np.linalg.norm(diff)
        #return {'weight': diff}
        return diff

    def merge_mean_color(graph, src, dst):
        """Callback called before merging two nodes of a mean color distance graph.
        This method computes the mean color of `dst`.
        Parameters
        ----------
        graph : RAG
            The graph under consideration.
        src, dst : int
            The vertices in `graph` to be merged.
        """
        graph.node[dst]['total color'] += graph.node[src]['total color']
        graph.node[dst]['pixel count'] += graph.node[src]['pixel count']
        graph.node[dst]['mean color'] = (graph.node[dst]['total color'] /
                                         graph.node[dst]['pixel count'])

    if segmentationProcess or saveSegmentation or featuresProcess:
        for image in fnmatch.filter(os.listdir('imgs'), '*.bmp'):

            if segmentationProcess:
                IOriginal = imread(path + '/' + image)

                mask = np.zeros((IOriginal.shape[0:2]), dtype=np.uint8)
                rr, cc = ellipse(round(IOriginal.shape[0]/2), round(IOriginal.shape[1]/2), round(IOriginal.shape[0]/2)-1, round(IOriginal.shape[1]/2)-1)
                mask[rr, cc] = 1

                maskrgb = gray2rgb(mask)
                #imshow(mask)
                #show()

                #IOriginalGray = rgb2gray(IOriginal)

                I = maskrgb * IOriginal

                #imshow(I)
                #show()

                GT = rgb2gray(imread(path + 'GT/' + image[:-4] + '_lesion.bmp').astype(float)) > 120

                # n_segments sujeto a cambios para optimización de la segmentación
                L = slic(I, n_segments=400)

                #subplot(1, 2, 1)
                Islic = mark_boundaries(I, L)
                #imshow(Islic)
                #show()
                #subplot(1, 2, 2)
                #imshow(mark_boundaries(GT, L, color=(1, 0, 0)))
                #show()

                g = graph.rag_mean_color(I, L)
                lc = graph.draw_rag(L, g, Islic)

                #imshow(lc)
                #show()

                L2 = graph.merge_hierarchical(L, g, thresh=50, rag_copy=False,
                                              in_place_merge=True,
                                              merge_func=merge_mean_color,
                                              weight_func=_weight_mean_color)

                Islic2 = mark_boundaries(I, L2)
                #figure()
                g2 = graph.rag_mean_color(I, L2)
                lc2 = graph.draw_rag(L2, g2, Islic2)

                '''
                out = label2rgb(L2, I, kind='avg')
                out = mark_boundaries(out, L2, (0, 0, 0))
                '''

                #subplot(1, 2, 1)
                #imshow(GT, cmap='gray')

                #subplot(1, 2, 2)
                #imshow(lc2)
                #show()

                s = np.zeros((IOriginal.shape[0:2]), dtype=np.uint8)
                L2label = label(L2)

                IGray = rgb2gray(IOriginal)# * mask
                IGaussian = gaussian(IGray, sigma=0.5)
                thresh = threshold_otsu(IGray)
                IOtsu = IGray <= thresh

                #Islic3 = mark_boundaries(IOtsu, L2)
                #g3 = graph.rag_mean_color(I, L2)
                #lc2 = graph.draw_rag(L2, g3, Islic3, border_color='#ff6600')
                #imshow(lc2, cmap='gray')
                #show()

                for i in range(1, L2label.max()):
                    lbl = (L2label == i)
                    countc = 0
                    count = 0
                    for j in range(len(lbl)):
                        for k in range(len(lbl[i])):
                            if not lbl[j][k] == 0:
                                if IOtsu[j][k]:
                                    countc += 1
                                count += 1
                    if (countc/count) >= 0.4:
                        s += lbl

                sMask = s * mask
                sMaskClose = closing(sMask, selem=disk(3))
                sMaskOpen = opening(sMaskClose, selem=disk(3))  # ??

                slabel = label(sMaskOpen)

                Isegmented = np.zeros((IOriginal.shape[0:2]), dtype=np.uint8)
                max = 0
                iMax = 0
                for i in range(1, slabel.max()):
                    if np.sum(slabel == i) > max:
                        max = np.sum(slabel == i)
                        iMax = i

                Isegmented = slabel == iMax

                if Isegmented[0][0] and Isegmented[0][len(Isegmented[0])-1] and Isegmented[len(Isegmented)-1][0] and Isegmented[len(Isegmented)-1][len(Isegmented[0])-1]:
                    Isegmented = np.invert(Isegmented)

                Isegmented = binary_fill_holes(Isegmented)

                #subplot(1, 2, 1)
                #imshow(GT, cmap='gray')

                #subplot(1, 2, 2)
                #imshow(Isegmented, cmap='gray')
                #show()

                if saveSegmentation:
                    imsave(pathSegmentation + '/' + image[:-4] + '_our.png', Isegmented, cmap='gray')
                    counter += 1
                    print(counter, '/', len(fnmatch.filter(os.listdir('imgs'), '*.bmp')))

                '''
                s = np.ones((IOriginal.shape[0:2]), dtype=np.uint8)
                L2label = label(L2)

                for i in range(1, L2label.max()):
                    R = 0
                    G = 0
                    B = 0
                    count = 0
                    It = (gray2rgb(L2label == i) * I)
                    for j in range(len(I)):
                        for k in range(len(I[j])):
                            if It[j][k][0] > 0 and It[j][k][1] > 0 and It[j][k][2] > 0 and:              #??
                                print('R +=', It[j][k][0])
                                R += It[j][k][0]
                                print('G +=', It[j][k][1])
                                G += It[j][k][1]
                                print('B +=', It[j][k][2])
                                B += It[j][k][2]
                                count += 1
                    print(R/count)
                    print(G/count)
                    print(B/count)
                    imshow(gray2rgb(L2label == i) * I)
                    show()
                '''
            if not segmentationProcess:
                IOriginal = imread(path + '/' + image)

                mask = np.zeros((IOriginal.shape[0:2]), dtype=np.uint8)
                rr, cc = ellipse(round(IOriginal.shape[0] / 2), round(IOriginal.shape[1] / 2),
                                 round(IOriginal.shape[0] / 2) - 1, round(IOriginal.shape[1] / 2) - 1)
                mask[rr, cc] = 1

                maskrgb = gray2rgb(mask)

                I = maskrgb * IOriginal

                GT = (rgb2gray(imread(path + 'GT/' + image[:-4] + '_lesion.bmp').astype(float)) * mask) > 120

                Isegmented255 = rgb2gray(imread(pathSegmentation + '/' + image[:-4] + '_our.png').astype(float))
                Isegmented = Isegmented255 > 120

            if featuresProcess:
                """
                    - Eliminar cositas constantes
                    - Feature selection para sacar las mejores
                """

                #print('mse ', image[:-4], compare_mse(GT, Isegmented))
                #mean_mse += compare_mse(GT, Isegmented)
                #print('ssim', image[:-4], compare_ssim(GT, Isegmented))
                #mean_ssim += compare_ssim(GT, Isegmented)
                #print('pnsr', image[:-4], compare_psnr(GT, Isegmented))
                #mean_pnsr += compare_psnr(GT, Isegmented)
                #print('jona', image[:-4], compare_jona(GT, Isegmented))
                #mean_jona += compare_jona(GT, Isegmented)
                #print('')

                jona, mean_jona = compare_jona(GT, Isegmented)

                if (jona >= mean_jona) and (np.sum(Isegmented255) > 0):

                    Xstack = []
                    Xnstack = []
                    print(image[: -4])

                    options = {'b': [
                        {'name': 'basicgeo', 'options': {'show': False}},                       # basic geometric features
                        {'name': 'hugeo', 'options': {'show': False}},                          # Hu moments
                        {'name': 'flusser', 'options': {'show': False}},                        # Flusser moments
                        {'name': 'fourierdes', 'options': {'show': False, 'Nfourierdes': 12}},  # Fourier descriptors
                    ]}

                    Xtmp, Xntmp = Bfx_geo(Isegmented255, options)
                    #print(Xtmp)
                    #print(Xntmp)

                    Xstack.extend(Xtmp[0])
                    Xnstack.extend(Xntmp)

                    options = {'dharalick': 3}  # 3 pixels distance for coocurrence

                    J = I[:, :, 0]  # red channel
                    Xtmp, Xntmp = Bfx_haralick(J, Isegmented255, options)  # Haralick features
                    # print(Xtmp)
                    Xntmp = [name + '_red' for name in Xntmp]
                    # print(Xntmp)

                    Xstack.extend(Xtmp[0])
                    Xnstack.extend(Xntmp)

                    J = I[:, :, 1]  # green channel
                    Xtmp, Xntmp = Bfx_haralick(J, Isegmented255, options)  # Haralick features
                    #print(Xtmp)
                    Xntmp = [name + '_green' for name in Xntmp]
                    #print(Xntmp)

                    Xstack.extend(Xtmp[0])
                    Xnstack.extend(Xntmp)

                    J = I[:, :, 2]  # blue channel
                    Xtmp, Xntmp = Bfx_haralick(J, Isegmented255, options)  # Haralick features
                    #print(Xtmp)
                    Xntmp = [name + '_blue' for name in Xntmp]
                    #print(Xntmp)

                    Xstack.extend(Xtmp[0])
                    Xnstack.extend(Xntmp)

                    a, _ = histogram(rgb2gray(I))
                    a = [element / sum(a) for element in a]

                    Xtmp = [entr(a).sum(axis=0)]
                    #print(Xtmp)
                    Xntmp = ['Entropy']
                    #print(Xntmp)

                    Xstack.extend(Xtmp)
                    Xnstack.extend(Xntmp)
                    #print(Xstack)
                    #print(Xnstack)

                    mean_red = 0
                    mean_green = 0
                    mean_blue = 0
                    count = 0

                    for i in range(len(I)):
                        for j in range(len(I[i])):
                            if mask[i][j]:
                                mean_red += I[i][j][0]
                                mean_green += I[i][j][1]
                                mean_blue += I[i][j][2]
                                count += 1

                    Xtmp = [mean_red / count, mean_green / count, mean_blue / count]
                    Xntmp = ['mean_red', 'mean_green', 'mean_blue']

                    Xstack.extend(Xtmp)
                    Xnstack.extend(Xntmp)
                    # print(Xstack)
                    # print(Xnstack)


                    X.append(Xstack)
                    if len(Xn) == 0:
                        Xn = Xnstack
                    d.extend([dph2[image[:-4]]])
                    imagesNames.extend([image[: -4]])

                    #print(X)
                    #print(Xn)
                    #print(d)
                    #print(imagesNames)

        if featuresProcess:
            print(X)
            print(Xn)
            print(d)
            print(imagesNames)
            savemat('X-Xn-d-names.mat', {'X': X, 'Xn': Xn, 'd': d, 'imagesNames': imagesNames})

    if trainAndTest:
        data = loadmat('X-Xn-d-names.mat')
        X = data['X']
        Xn = data['Xn']
        d = data['d'][0]
        imagesNames = data['imagesNames']

        #print(X)
        #print(len(Xn))
        #print(d)
        #print(imagesNames)

        # training
        print('training')
        sclean = Bfs_clean(X, 1)
        Xclean = X[:, sclean]
        Xnclean = Xn[sclean]

        #print(sclean)
        #print(Xclean)
        #print(Xnclean)

        Xtrain = Xclean[: 85]
        Xntrain = Xnclean[: 85]
        dtrain = d[: 85]

        b = [
            {'name': 'lda', 'options': {'p': []}},
            {'name': 'maha', 'options': {}},
            {'name': 'qda', 'options': {'p': []}},
            {'name': 'svm', 'options': {'kernel': 1}},
            {'name': 'svm', 'options': {'kernel': 2}},
            #{'name': 'knn', 'options': {'k': 5}},
            {'name': 'nn', 'options': {'method': 1, 'iter': 15}}
        ]
        op = b
        struct = Bcl_structure(Xtrain, dtrain, op)

        Xtest = Xclean[85:]
        Xntest = Xnclean[85:]
        dtest = d[85:]

        #print('testing')

        ds, _ = Bcl_structure(Xtest, struct)
        for i in range(len(op)):
            T, p = Bev_confusion(dtest, ds[:, i])
            print(b[i]['name'])
            print(p)
            print(T)

'''
Clinical Diagnosis:
    0 - Common Nevus;
    1 - Atypical Nevus;
    2 - Melanoma.
'''

dph2 = {
    'IMD003': 0,
    'IMD009': 0,
    'IMD016': 0,
    'IMD022': 0,
    'IMD024': 0,
    'IMD025': 0,
    'IMD035': 0,
    'IMD038': 0,
    'IMD042': 0,
    'IMD044': 0,
    'IMD045': 0,
    'IMD050': 0,
    'IMD092': 0,
    'IMD101': 0,
    'IMD103': 0,
    'IMD112': 0,
    'IMD118': 0,
    'IMD125': 0,
    'IMD132': 0,
    'IMD134': 0,
    'IMD135': 0,
    'IMD144': 0,
    'IMD146': 0,
    'IMD147': 0,
    'IMD150': 0,
    'IMD152': 0,
    'IMD156': 0,
    'IMD159': 0,
    'IMD161': 0,
    'IMD162': 0,
    'IMD175': 0,
    'IMD177': 0,
    'IMD182': 0,
    'IMD198': 0,
    'IMD200': 0,
    'IMD010': 0,
    'IMD017': 0,
    'IMD020': 0,
    'IMD039': 0,
    'IMD041': 0,
    'IMD105': 0,
    'IMD107': 0,
    'IMD108': 0,
    'IMD133': 0,
    'IMD142': 0,
    'IMD143': 0,
    'IMD160': 0,
    'IMD173': 0,
    'IMD176': 0,
    'IMD196': 0,
    'IMD197': 0,
    'IMD199': 0,
    'IMD203': 0,
    'IMD204': 0,
    'IMD206': 0,
    'IMD207': 0,
    'IMD208': 0,
    'IMD364': 0,
    'IMD365': 0,
    'IMD367': 0,
    'IMD371': 0,
    'IMD372': 0,
    'IMD374': 0,
    'IMD375': 0,
    'IMD378': 0,
    'IMD379': 0,
    'IMD380': 0,
    'IMD381': 0,
    'IMD383': 0,
    'IMD384': 0,
    'IMD385': 0,
    'IMD389': 0,
    'IMD390': 0,
    'IMD392': 0,
    'IMD394': 0,
    'IMD395': 0,
    'IMD397': 0,
    'IMD399': 0,
    'IMD400': 0,
    'IMD402': 0,
    'IMD002': 1,
    'IMD004': 1,
    'IMD013': 1,
    'IMD015': 1,
    'IMD019': 1,
    'IMD021': 1,
    'IMD027': 1,
    'IMD030': 1,
    'IMD032': 1,
    'IMD033': 1,
    'IMD037': 1,
    'IMD040': 1,
    'IMD043': 1,
    'IMD047': 1,
    'IMD048': 1,
    'IMD049': 1,
    'IMD057': 1,
    'IMD075': 1,
    'IMD076': 1,
    'IMD078': 1,
    'IMD120': 1,
    'IMD126': 1,
    'IMD137': 1,
    'IMD138': 1,
    'IMD139': 1,
    'IMD140': 1,
    'IMD149': 1,
    'IMD153': 1,
    'IMD157': 1,
    'IMD164': 1,
    'IMD166': 1,
    'IMD169': 1,
    'IMD171': 1,
    'IMD210': 1,
    'IMD347': 1,
    'IMD155': 1,
    'IMD376': 1,
    'IMD006': 1,
    'IMD008': 1,
    'IMD014': 1,
    'IMD018': 1,
    'IMD023': 1,
    'IMD031': 1,
    'IMD036': 1,
    'IMD154': 1,
    'IMD170': 1,
    'IMD226': 1,
    'IMD243': 1,
    'IMD251': 1,
    'IMD254': 1,
    'IMD256': 1,
    'IMD278': 1,
    'IMD279': 1,
    'IMD280': 1,
    'IMD304': 1,
    'IMD305': 1,
    'IMD306': 1,
    'IMD312': 1,
    'IMD328': 1,
    'IMD331': 1,
    'IMD339': 1,
    'IMD356': 1,
    'IMD360': 1,
    'IMD368': 1,
    'IMD369': 1,
    'IMD370': 1,
    'IMD382': 1,
    'IMD386': 1,
    'IMD388': 1,
    'IMD393': 1,
    'IMD396': 1,
    'IMD398': 1,
    'IMD427': 1,
    'IMD430': 1,
    'IMD431': 1,
    'IMD432': 1,
    'IMD433': 1,
    'IMD434': 1,
    'IMD436': 1,
    'IMD437': 1,
    'IMD058': 2,
    'IMD061': 2,
    'IMD063': 2,
    'IMD064': 2,
    'IMD065': 2,
    'IMD080': 2,
    'IMD085': 2,
    'IMD088': 2,
    'IMD090': 2,
    'IMD091': 2,
    'IMD168': 2,
    'IMD211': 2,
    'IMD219': 2,
    'IMD240': 2,
    'IMD242': 2,
    'IMD284': 2,
    'IMD285': 2,
    'IMD348': 2,
    'IMD349': 2,
    'IMD403': 2,
    'IMD404': 2,
    'IMD405': 2,
    'IMD407': 2,
    'IMD408': 2,
    'IMD409': 2,
    'IMD410': 2,
    'IMD413': 2,
    'IMD417': 2,
    'IMD418': 2,
    'IMD419': 2,
    'IMD406': 2,
    'IMD411': 2,
    'IMD420': 2,
    'IMD421': 2,
    'IMD423': 2,
    'IMD424': 2,
    'IMD425': 2,
    'IMD426': 2,
    'IMD429': 2,
    'IMD435': 2
}

path = 'imgs'
pathSegmentation = 'our_segmentation'
magic(segmentationProcess=False, saveSegmentation=False, featuresProcess=False, trainAndTest=True)
