from pyAudioAnalysis import audioFeatureExtraction as aF
from pyAudioAnalysis import audioTrainTest as aT
import cv2, time, sys, glob, os, csv
import matplotlib
import ntpath
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy
import scipy
import scipy.signal
import scipy.cluster.hierarchy as hier
import scipy.spatial.distance as dist
import datetime
from scipy.interpolate import interp1d
import collections
from scipy.fftpack import rfft
from scipy.fftpack import fft
from scipy.fftpack.realtransforms import dct
from scipy.signal import fftconvolve
import sklearn.svm
import sklearn.decomposition
import sklearn.ensemble
import featureExtraction as iFE
import cPickle


eps = 0.00000001


def mtFeatureExtraction(signal, Fs, mtWin, mtStep, stWin, stStep):
    """
    Mid-term feature extraction
    """

    mtWinRatio = int(round(mtWin / stStep))
    mtStepRatio = int(round(mtStep / stStep))

    mtFeatures = []

    stFeatures = stFeatureExtraction2(signal, Fs, stWin, stStep)
    numOfFeatures = len(stFeatures)
    numOfStatistics = 6

    mtFeatures = []
    #for i in range(numOfStatistics * numOfFeatures + 1):
    for i in range(numOfStatistics * numOfFeatures):
        mtFeatures.append([])

    for i in range(numOfFeatures):        # for each of the short-term features:
        curPos = 0
        N = len(stFeatures[i])
        while (curPos < N):
            N1 = curPos
            N2 = curPos + mtWinRatio
            if N2 > N:
                N2 = N
            curStFeatures = stFeatures[i][N1:N2]

            mtFeatures[i].append(numpy.mean(curStFeatures))
            mtFeatures[i+numOfFeatures].append(numpy.std(curStFeatures))
            mtFeatures[i+2*numOfFeatures].append(numpy.max(curStFeatures))
            mtFeatures[i+3*numOfFeatures].append(numpy.min(curStFeatures))
            lower = numpy.sort(curStFeatures)[0:int(curStFeatures.shape[0]/3)]
            upper = numpy.sort(curStFeatures)[-int(curStFeatures.shape[0]/3)::]
            if lower.shape[0]>0:
                mtFeatures[i+4*numOfFeatures].append(numpy.mean(lower))
            else:
                mtFeatures[i+4*numOfFeatures].append(numpy.mean(curStFeatures))
            if upper.shape[0]>0:
                mtFeatures[i+5*numOfFeatures].append(numpy.mean(upper))
            else:
                mtFeatures[i+5*numOfFeatures].append(numpy.mean(curStFeatures))
            '''                
            if lower.shape[0]>0:
                mtFeatures[i+6*numOfFeatures].append(numpy.mean(lower))
            else:
                mtFeatures[i+6*numOfFeatures].append(numpy.mean(curStFeatures))
            if upper.shape[0]>0:
                mtFeatures[i+7*numOfFeatures].append(numpy.mean(upper))
            else:
                mtFeatures[i+7*numOfFeatures].append(numpy.mean(curStFeatures))
            '''
            curPos += mtStepRatio

    return numpy.array(mtFeatures), stFeatures



def stZCR(frame):
    """Computes zero crossing rate of frame"""
    count = len(frame)
    countZ = numpy.sum(numpy.abs(numpy.diff(numpy.sign(frame)))) / 2
    return (numpy.float64(countZ) / numpy.float64(count-1.0))


def stEnergy(frame):
    """Computes signal energy of frame"""
    return numpy.sum(frame ** 2) / numpy.float64(len(frame))


def stEnergyEntropy(frame, numOfShortBlocks=10):
    """Computes entropy of energy"""
    Eol = numpy.sum(frame ** 2)    # total frame energy
    L = len(frame)
    subWinLength = int(numpy.floor(L / numOfShortBlocks))
    if L != subWinLength * numOfShortBlocks:
            frame = frame[0:subWinLength * numOfShortBlocks]
    # subWindows is of size [numOfShortBlocks x L]
    subWindows = frame.reshape(subWinLength, numOfShortBlocks, order='F').copy()

    # Compute normalized sub-frame energies:
    s = numpy.sum(subWindows ** 2, axis=0) / (Eol + eps)

    # Compute entropy of the normalized sub-frame energies:
    Entropy = -numpy.sum(s * numpy.log2(s + eps))
    return Entropy


""" Frequency-domain audio features """


def stSpectralCentroidAndSpread(X, fs):
    """Computes spectral centroid of frame (given abs(FFT))"""
    ind = (numpy.arange(1, len(X) + 1)) * (fs/(2.0 * len(X)))

    Xt = X.copy()
    Xt = Xt / Xt.max()
    NUM = numpy.sum(ind * Xt)
    DEN = numpy.sum(Xt) + eps

    # Centroid:
    C = (NUM / DEN)

    # Spread:
    S = numpy.sqrt(numpy.sum(((ind - C) ** 2) * Xt) / DEN)

    # Normalize:
    C = C / (fs / 2.0)
    S = S / (fs / 2.0)

    return (C, S)


def stSpectralEntropy(X, numOfShortBlocks=10):
    """Computes the spectral entropy"""
    L = len(X)                         # number of frame samples
    Eol = numpy.sum(X ** 2)            # total spectral energy

    subWinLength = int(numpy.floor(L / numOfShortBlocks))   # length of sub-frame
    if L != subWinLength * numOfShortBlocks:
        X = X[0:subWinLength * numOfShortBlocks]

    subWindows = X.reshape(subWinLength, numOfShortBlocks, order='F').copy()  # define sub-frames (using matrix reshape)
    s = numpy.sum(subWindows ** 2, axis=0) / (Eol + eps)                      # compute spectral sub-energies
    En = -numpy.sum(s*numpy.log2(s + eps))                                    # compute spectral entropy

    return En


def stSpectralFlux(X, Xprev):
    """
    Computes the spectral flux feature of the current frame
    ARGUMENTS:
        X:        the abs(fft) of the current frame
        Xpre:        the abs(fft) of the previous frame
    """
    # compute the spectral flux as the sum of square distances:
    sumX = numpy.sum(X + eps)
    sumPrevX = numpy.sum(Xprev + eps)
    F = numpy.sum((X / sumX - Xprev/sumPrevX) ** 2)

    return F


def stSpectralRollOff(X, c, fs):
    """Computes spectral roll-off"""
    totalEnergy = numpy.sum(X ** 2)
    fftLength = len(X)
    Thres = c*totalEnergy
    # Ffind the spectral rolloff as the frequency position where the respective spectral energy is equal to c*totalEnergy
    CumSum = numpy.cumsum(X ** 2) + eps
    [a, ] = numpy.nonzero(CumSum > Thres)
    if len(a) > 0:
        mC = numpy.float64(a[0]) / (float(fftLength))
    else:
        mC = 0.0
    return (mC)



def stFeatureExtraction(signal, Fs, Win, Step):
    """
    This function implements the shor-term windowing process. For each short-term window a set of features is extracted.
    This results to a sequence of feature vectors, stored in a numpy matrix.

    ARGUMENTS
        signal:       the input signal samples
        Fs:           the sampling freq (in Hz)
        Win:          the short-term window size (in samples)
        Step:         the short-term window step (in samples)
    RETURNS
        stFeatures:   a numpy array (numOfFeatures x numOfShortTermWindows)
    """

    Win = int(Win)
    Step = int(Step)

    # Signal normalization
    signal = numpy.double(signal)

    signal = signal / (2.0 ** 15)
    DC = signal.mean()
    MAX = (numpy.abs(signal)).max()
    signal = (signal - DC) / MAX

    N = len(signal)                                # total number of samples
    curPos = 0
    countFrames = 0
    nFFT = Win / 2

    numOfTimeSpectralFeatures = 8
    totalNumOfFeatures = numOfTimeSpectralFeatures 

    stFeatures = []
    while (curPos + Win - 1 < N):                        # for each short-term window until the end of signal
        countFrames += 1
        x = signal[curPos:curPos+Win]                    # get current window
        curPos = curPos + Step                           # update window position
        X = abs(fft(x))                                  # get fft magnitude
        X = X[0:nFFT]                                    # normalize fft
        X = X / len(X)
        if countFrames == 1:
            Xprev = X.copy()                             # keep previous fft mag (used in spectral flux)
        curFV = numpy.zeros((totalNumOfFeatures, 1))
        curFV[0] = stZCR(x)                              # zero crossing rate
        curFV[1] = stEnergy(x)                           # short-term energy
        curFV[2] = stEnergyEntropy(x)                    # short-term entropy of energy
        [curFV[3], curFV[4]] = stSpectralCentroidAndSpread(X, Fs)    # spectral centroid and spread
        curFV[5] = stSpectralEntropy(X)                  # spectral entropy
        curFV[6] = stSpectralFlux(X, Xprev)              # spectral flux
        curFV[7] = stSpectralRollOff(X, 0.90, Fs)        # spectral rolloff
        stFeatures.append(curFV)
        Xprev = X.copy()

    stFeatures = numpy.concatenate(stFeatures, 1)
    return stFeatures


def stFeatureExtraction2(signal, Fs, Win, Step):
    """
    This function implements the shor-term windowing process. For each short-term window a set of features is extracted.
    This results to a sequence of feature vectors, stored in a numpy matrix.

    ARGUMENTS
        signal:       the input signal samples
        Fs:           the sampling freq (in Hz)
        Win:          the short-term window size (in samples)
        Step:         the short-term window step (in samples)
    RETURNS
        stFeatures:   a numpy array (numOfFeatures x numOfShortTermWindows)
    """

    Win = int(Win)
    Step = int(Step)

    # Signal normalization
    signal = numpy.double(signal)

    #signal = signal / (2.0 ** 15)
    DC = signal.mean()
    MAX = (numpy.abs(signal)).max()
    signal = (signal - DC)

    N = len(signal)                                # total number of samples
    curPos = 0
    countFrames = 0
    nFFT = Win / 2
    
    #numOfTimeSpectralFeatures = 16
    numOfTimeSpectralFeatures = 14*2
    totalNumOfFeatures = numOfTimeSpectralFeatures 

    stFeatures = []
    while (curPos + Win - 1 < N):                        # for each short-term window until the end of signal
        countFrames += 1
        x = signal[curPos:curPos+Win]                    # get current window
        curPos = curPos + Step                           # update window position
        X = abs(fft(x))                                  # get fft magnitude
        X = X[0:nFFT]                                    # normalize fft
        X = X / len(X)
        if countFrames == 1:
            Xprev = X.copy()                             # keep previous fft mag (used in spectral flux)
        curFV = numpy.zeros((totalNumOfFeatures, 1))
        #curFV[0] = stZCR(x)                              # zero crossing rate
        #curFV[1] = stEnergy(x)                           # short-term energy
        #curFV[2] = stEnergyEntropy(x)                    # short-term entropy of energy
        curFV[0] = numpy.mean(x)
        curFV[1] = numpy.std(x)
        curFV[2] = numpy.median(x)
        curFV[3] = stZCR(x)
        curFV[4] = numpy.max(x)
        curFV[5] = numpy.min(x)
        curFV[6] = numpy.max(numpy.abs(x))
        curFV[7] = numpy.min(numpy.abs(x))
        curFV[8] = stEnergyEntropy(x)                    # short-term entropy of energy        
        [curFV[9], curFV[10]] = stSpectralCentroidAndSpread(X, Fs)    # spectral centroid and spread        
        curFV[11] = stSpectralEntropy(X)                  # spectral entropy
        curFV[12] = stSpectralFlux(X, Xprev)              # spectral flux
        curFV[13] = stSpectralRollOff(X, 0.90, Fs)        # spectral rolloff
        # TODO: TEST DELTA
        if countFrames>1:
            curFV[numOfTimeSpectralFeatures/2::] = curFV[0:numOfTimeSpectralFeatures/2] - prevFV[0:numOfTimeSpectralFeatures/2]
        else:
            curFV[numOfTimeSpectralFeatures/2::] = curFV[0:numOfTimeSpectralFeatures/2]

        stFeatures.append(curFV)
        prevFV = curFV.copy()
        Xprev = X.copy()

    stFeatures = numpy.concatenate(stFeatures, 1)
    return stFeatures


def getNumOfRepetitions(fftSeq, duration):
    nFFT = fftSeq.shape[0]
    fi = numpy.arange(0,nFFT)
    Fs = fftSeq.shape[0] / float(duration)
    start = 10
    fftSeq[0:start] = 0    
    iMax = numpy.argmax(fftSeq)              
    maxFreq = (Fs/2)*fi[iMax] / float(nFFT)        
    #print duration
    #print iMax
    #print maxFreq
    maxT = 1 / maxFreq    
    return round(2 * duration / (maxT))


def processAccelerometer(X, Y, Z, duration):
    X = numpy.array(X)
    Y = numpy.array(Y)
    Z = numpy.array(Z)
    #plt.plot(X)
    #plt.plot(Y)
    #plt.plot(Z)
    #plt.show(block = False)

    nFFT = X.shape[0]/2 
    Xfft = abs(fft(X))     
    Xfft = Xfft[0:nFFT]
    Xfft = Xfft / len(Xfft)
    eX = getNumOfRepetitions(Xfft, duration)      

    nFFT = X.shape[0]/2 
    Yfft = abs(fft(Y))     
    Yfft = Xfft[0:nFFT]
    Yfft = Xfft / len(Yfft)
    eY = getNumOfRepetitions(Yfft, duration)      

    nFFT = Z.shape[0]/2 
    Zfft = abs(fft(Z))     
    Zfft = Zfft[0:nFFT]
    Zfft = Zfft / len(Xfft)
    eZ = getNumOfRepetitions(Zfft, duration)      

    All = X + Y + Z
    nFFT = All.shape[0]/2 
    Allfft = abs(fft(All))     
    Allfft = Allfft[0:nFFT]
    Allfft = Allfft / len(Allfft)
    eAll = getNumOfRepetitions(Allfft, duration)      

    '''
    plt.subplot(2,1,1)
    plt.plot(X)
    plt.plot(Y)
    plt.plot(Z)
    plt.plot(X+Y+Z)
    plt.subplot(2,1,2)
    plt.plot(Allfft)        
    plt.show()
    '''

    return eX, eY, eZ

def featureExtraction(csvFile, useAccelerometer, useAccelerometerOnlyX, useAccelerometerOnlyY, useAccelerometerOnlyZ, useImage):
    if useAccelerometer:
        ''' Analyze accelermoterer data '''    
        Xs, Ys, Zs, duration = readCSVFileAccelerometer(csvFile)
        Fs = round(len(Xs) / float(duration))            
        mtWin = 5.0
        mtStep = 1.0
        stWin = 0.5
        stStep = 0.5
        [MidTermFeatures, stFeatures] = mtFeatureExtraction(Xs, Fs, round(mtWin * Fs), round(mtStep * Fs), round(Fs * stWin), round(Fs * stStep))
        Fx = MidTermFeatures.mean(axis = 1)            
        [MidTermFeatures, stFeatures] = mtFeatureExtraction(Ys, Fs, round(mtWin * Fs), round(mtStep * Fs), round(Fs * stWin), round(Fs * stStep))
        Fy = MidTermFeatures.mean(axis = 1)            
        [MidTermFeatures, stFeatures] = mtFeatureExtraction(Zs, Fs, round(mtWin * Fs), round(mtStep * Fs), round(Fs * stWin), round(Fs * stStep))
        Fz = MidTermFeatures.mean(axis = 1)            

        ''' Fused features '''                                
        FeatureVectorFusion = numpy.concatenate([Fx, Fy, Fz])            

        if useAccelerometerOnlyX:
            FeatureVectorFusion = Fx.copy()
        if useAccelerometerOnlyY:
            FeatureVectorFusion = Fy.copy()
        if useAccelerometerOnlyZ:
            FeatureVectorFusion = Fz.copy()                             


        ''' get number of repetitions from accelerometer '''            
        #x, y, z = processAccelerometer(Xs, Ys, Zs, duration)
        #eX.append(x)
        #eY.append(y)
        #eZ.append(z)
    else:
        FeatureVectorFusion = numpy.array([])

    ''' Image Features '''
    if useImage:
        fileNameImg = csvFile.replace(".csv",".png")
        imF, imFnames = iFE.getFeaturesFromFile(fileNameImg)
        imF = numpy.array(imF)                
        #imF = numpy.dot(imF, rw)
        FeatureVectorFusion = numpy.concatenate([FeatureVectorFusion, imF])
        FeatureVectorFusion += numpy.random.rand(FeatureVectorFusion.shape[0]) * 0.000000000010                        
    return FeatureVectorFusion

def classifyDir(argv):
    dirName = argv[2]
    modelName = argv[3]
    useAccelerometer = ((argv[4]=="1") or (argv[4]=="2") or (argv[4]=="3")  or (argv[4]=="4"))
    useAccelerometerOnlyX = (argv[4]=="1")
    useAccelerometerOnlyY = (argv[4]=="2")
    useAccelerometerOnlyZ = (argv[4]=="3")
    useImage = (argv[5]=="1")    
    fileList  = sorted(glob.glob(os.path.join(dirName, "*.csv")))        

    try:
        fo = open(modelName+"MEANS", "rb")
    except IOError:
            print "Load SVM Model: Didn't find file"
            return
    try:
        MEAN = cPickle.load(fo)
        STD = cPickle.load(fo)
        classNames = cPickle.load(fo)        
    except:
        fo.close()
    fo.close()

    CM = numpy.zeros((len(classNames),len(classNames)))

    for i, m in enumerate(fileList):                
        gt = int(ntpath.basename(m).split("_")[-1].replace(".csv",""))
        className = ntpath.basename(m).split("_")[1]        
        result = classifySingleFile(m, modelName, useAccelerometer, useAccelerometerOnlyX, useAccelerometerOnlyY, useAccelerometerOnlyZ, useImage)
        print className, result
        CM[classNames.index(className), classNames.index(result)] += 1


    CM = CM + 0.0000000010

    Rec = numpy.zeros((CM.shape[0], ))
    Pre = numpy.zeros((CM.shape[0], ))

    for ci in range(CM.shape[0]):
        Rec[ci] = CM[ci, ci] / numpy.sum(CM[ci, :])
        Pre[ci] = CM[ci, ci] / numpy.sum(CM[:, ci])
    F1 = 2 * Rec * Pre / (Rec + Pre)
    print CM
    print numpy.mean(F1)
    numpy.save(modelName + "_results.npy", CM)




def classifySingleFile(fileName, modelName, useAccelerometer, useAccelerometerOnlyX, useAccelerometerOnlyY, useAccelerometerOnlyZ, useImage):

    fV = featureExtraction(fileName, useAccelerometer, useAccelerometerOnlyX, useAccelerometerOnlyY, useAccelerometerOnlyZ, useImage)        

    try:
        fo = open(modelName+"MEANS", "rb")
    except IOError:
            print "Load SVM Model: Didn't find file"
            return
    try:
        MEAN = cPickle.load(fo)
        STD = cPickle.load(fo)
        classNames = cPickle.load(fo)        
    except:
        fo.close()
    fo.close()

    MEAN = numpy.array(MEAN)
    STD = numpy.array(STD)    
    fV = (fV - MEAN) / STD

    COEFF = []
    with open(modelName, 'rb') as fid:
        SVM = cPickle.load(fid)    
    [Result, P] = aT.classifierWrapper(SVM, "svm", fV)    # classification            
    return classNames[int(Result)]




def evaluateClassifier(argv):
    dirName = argv[2]    
    useAccelerometer = ((argv[3]=="1") or (argv[3]=="2") or (argv[3]=="3")  or (argv[3]=="4"))
    useAccelerometerOnlyX = (argv[3]=="1")
    useAccelerometerOnlyY = (argv[3]=="2")
    useAccelerometerOnlyZ = (argv[3]=="3")

    useImage = (argv[4]=="1")    
    fileList  = sorted(glob.glob(os.path.join(dirName, "*.csv")))    
    GTs = []
    eX = []        
    eY = []    
    eZ = [] 

    featuresAll = []
    classNames = []
    

    for i, m in enumerate(fileList):                
        gt = int(ntpath.basename(m).split("_")[-1].replace(".csv",""))

        className = ntpath.basename(m).split("_")[1]        
        if not className in classNames:
            classNames.append(className)
            featuresAll.append([])         
        #if gt>0:
        if True:
            GTs.append(gt)
            FeatureVectorFusion = featureExtraction(m, useAccelerometer, useAccelerometerOnlyX, useAccelerometerOnlyY, useAccelerometerOnlyZ, useImage)
            print FeatureVectorFusion.shape
            if len(featuresAll[classNames.index(className)])==0:
                featuresAll[classNames.index(className)] = FeatureVectorFusion
            else:
                featuresAll[classNames.index(className)] = numpy.vstack((featuresAll[classNames.index(className)], FeatureVectorFusion))

    #featuresAll = featuresY
    (featuresAll, MEAN, STD) = aT.normalizeFeatures(featuresAll)
    bestParam = aT.evaluateClassifier(featuresAll, classNames, 1000, "svm", [0.05, 0.1, 0.5, 1, 2,3, 5, 10, 15, 20, 25, 50, 100, 200], 0, perTrain=0.80)

    MEAN = MEAN.tolist()
    STD = STD.tolist()    

    # STEP C: Save the classifier to file    
    Classifier = aT.trainSVM(featuresAll, bestParam)
    modelName = argv[5]
    with open(modelName, 'wb') as fid:                                            # save to file
        cPickle.dump(Classifier, fid)            
    fo = open(modelName + "MEANS", "wb")
    cPickle.dump(MEAN, fo, protocol=cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(STD, fo, protocol=cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(classNames, fo, protocol=cPickle.HIGHEST_PROTOCOL)
    fo.close()



    '''
    ClassifierX = aT.trainSVM(featuresX, 1)

    CM = numpy.zeros((len(classNames),len(classNames)))

    for i in range(len(featuresX)):
        for j in range(featuresX[i].shape[0]):            
            predicted = int(ClassifierX.predict(featuresX[i][j].reshape(1, -1))[0])

            CM[i][predicted] += 1
            print classNames[predicted], classNames[i]
    print CM
    '''

    #X = numpy.array([eX, eY, eZ]).T
    #GTs = numpy.array(GTs)
    #Cparam = 5
    #svm = sklearn.svm.SVR(C = Cparam, kernel = 'linear')
    #svm.fit(X,GTs)    
    #Ys = svm.predict(X)
    #for i in range(Ys.shape[0]):
    #    print "{0:.1f}\t{1:.1f}\t{2:.1f}\t{3:.1f}\t{4:.1f}\t{5:.1f}".format(GTs[i], eX[i], eY[i], eZ[i], Ys[i], numpy.abs(Ys[i]-GTs[i]))
    #trainError = numpy.median(numpy.abs(svm.predict(X) - GTs))
    #print trainError

    #plt.plot(GTs, eX, '*')
    #plt.show()

def readCSVFileAccelerometer(filePath):
    Xs = []
    Ys = []
    Zs = []                    

    with open(filePath, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for j, row in enumerate(reader):
            if j>0:
                Xs.append(float(row[1]))
                Ys.append(float(row[2]))
                Zs.append(float(row[3]))                        
                duration = float(row[0])      
    return Xs, Ys, Zs, duration

def showFileAccelerometer(argv):
    filePath = argv[2]
    Xs, Ys, Zs, duration = readCSVFileAccelerometer(filePath)
    Fs = round(len(Xs) / float(duration))
    T = numpy.arange(0,len(Xs) / float(Fs), 1.0/Fs)    
    plt.plot(T, Xs)
    plt.plot(T, Ys)
    plt.plot(T, Zs)
    plt.show()

if __name__ == '__main__':
    if sys.argv[1] == "classifier":
        evaluateClassifier(sys.argv)
    elif sys.argv[1] == "showFileAccelerometer":
        showFileAccelerometer(sys.argv)
    elif sys.argv[1] == "classifyFile":
        argv = sys.argv
        fileName = argv[2]
        modelName = argv[3]
        useAccelerometer = ((argv[4]=="1") or (argv[4]=="2") or (argv[4]=="3")  or (argv[4]=="4"))
        useAccelerometerOnlyX = (argv[4]=="1")
        useAccelerometerOnlyY = (argv[4]=="2")
        useAccelerometerOnlyZ = (argv[4]=="3")
        useImage = (argv[5]=="1")    

        print classifySingleFile(fileName, modelName, useAccelerometer, useAccelerometerOnlyX, useAccelerometerOnlyY, useAccelerometerOnlyZ, useImage)
    elif sys.argv[1] == "classifyDirAndEvaluate":
        classifyDir(sys.argv)

