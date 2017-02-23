import cv2, time, sys, glob, os
import matplotlib
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
import scipy.misc
import csv,math

# process and plot related parameters:
newWidth = 640; processStep = 0.1; plotStep = 1;

def plotCV(Fun, Width, Height, MAX):
    if len(Fun)>Width:
        hist_item = Height * (Fun[len(Fun)-Width-1:-1] / MAX)
    else:
        hist_item = Height * (Fun / MAX)
    h = numpy.zeros((Height, Width, 3))
    hist = numpy.int32(numpy.around(hist_item))

    for x,y in enumerate(hist):
        cv2.line(h,(x,Height),(x,Height-y),(255,0,255))
    return h


def resizeFrame(frame, targetWidth):    
    (Width, Height) = frame.shape[1], frame.shape[0]

    if targetWidth > 0:                             # Use FrameWidth = 0 for NO frame resizing
        ratio = float(Width) / targetWidth        
        newHeight = int(round(float(Height) / ratio))
        frameFinal = cv2.resize(frame, (targetWidth, newHeight))
    else:
        frameFinal = frame;

    return frameFinal
    
def median(lst):
    return numpy.median(numpy.array(lst))

def mean(weights,lst):
    #return numpy.mean(numpy.array(weights)*numpy.array(lst))
    return float(numpy.sum(numpy.array(weights)*numpy.array(lst)))/float(numpy.sum(numpy.array(weights)))

def processMovie(moviePath, processMode, PLOT):
    Tstart = time.time(); T0 = Tstart;
    capture = cv2.VideoCapture(moviePath)
    nFrames = capture.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    fps = capture.get(cv2.cv.CV_CAP_PROP_FPS)
    duration = nFrames / fps
    secondsD  = duration; HoursD   = int(secondsD/3600); MinutesD = int(secondsD/60); SecondsD = int(secondsD) % 60; DsecsD = int(100*(secondsD - int(secondsD)));
    StringTimeD = '{0:02d}:{1:02d}:{2:02d}.{3:02d}'.format(HoursD, MinutesD, SecondsD, DsecsD);

    pOld = numpy.array([])
    timeStamps = numpy.array([])
    FrameValue_Diffs = numpy.array([])
    FrameValue_Diffs_Grayscale = []
    processFPS = numpy.array([])
    processT   = numpy.array([])
    FramDiffSums  = numpy.array([])
    
    count = 0
    countProcess = 0
    countDiffs = 0

    nextTimeStampToProcess = 0.0
    PROCESS_NOW = False

    while (1):
        # cv.SetCaptureProperty( capture, cv.CV_CAP_PROP_POS_FRAMES, count*frameStep );     # THIS IS TOOOOO SLOW (MAKES THE READING PROCESS 2xSLOWER)
        ret, frame = capture.read()
        timeStamp = float(count) / fps
        if timeStamp >= nextTimeStampToProcess:
            nextTimeStampToProcess += processStep;
            PROCESS_NOW = True
        if ret:
            count += 1; 
            (Width, Height) = frame.shape[1], frame.shape[0]                # get frame dimensions
            frame2 = cv2.cvtColor(frame, cv2.cv.CV_BGR2RGB)                    # convert BGR to RGB
            RGB = resizeFrame(frame2, newWidth)                        # reduce frame size            
            Grayscale = cv2.cvtColor(RGB, cv2.cv.CV_RGB2GRAY)                # get grayscale image
            (Width, Height) = Grayscale.shape[1], Grayscale.shape[0]


            if PROCESS_NOW:
                curFV = numpy.array([])                                                                        # current feature vector initalization

                countProcess += 1
                timeStamps = numpy.append(timeStamps, timeStamp)
            
                if ((countProcess > 2) and (countProcess % plotStep ==0) and (PLOT==1)):
                    # draw RGB image and visualizations
                    vis = cv2.cvtColor(RGB, cv2.cv.CV_RGB2BGR)
    
                    # Time-related plots:
                    T2 = time.time();                
                    seconds  = float(count)/fps; Hours   = int(seconds/3600); Minutes = int(seconds/60); Seconds = int(seconds) % 60; Dsecs = int(100*(seconds - int(seconds)));
                    StringTime = '{0:02d}:{1:02d}:{2:02d}.{3:02d}'.format(Hours, Minutes, Seconds, Dsecs);
                    processFPS = numpy.append(processFPS, plotStep / float(T2-T0))
                    processT   = numpy.append(processT,   100.0 * float(T2-T0) / (processStep * plotStep))
                    if len(processFPS)>250:
                        processFPS_winaveg = numpy.mean(processFPS[-250:-1])
                        processT_winaveg = numpy.mean(processT[-250:-1])
                    else:
                        processFPS_winaveg = numpy.mean(processFPS)
                        processT_winaveg = numpy.mean(processT)

                    secondsRemain = processT_winaveg * float(secondsD - seconds) / 100.0; HoursRemain   = int(secondsRemain/3600); MinutesRemain = int(secondsRemain/60); SecondsRemain = int(secondsRemain) % 60; 

                    StringTimeRemain = '{0:02d}:{1:02d}:{2:02d}'.format(HoursRemain, MinutesRemain, SecondsRemain);
                    StringToPlot = '{0:s}/{1:s} {2:5.1f}fps,{3:2.1f}xR {4:s}'.format(StringTime, StringTimeD, processFPS_winaveg, 100.0/float(processT_winaveg),StringTimeRemain)                    
                    cv2.rectangle(vis, (0, 0), (Width, 17), (255,255,255), -1)
                    cv2.putText(vis, StringToPlot, (20, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0))

                    # Draw color image:
                    WidthPlot = 150;
                    WidthPlot2 = 150;            # used for static plot (e.g. 1D histogram)
                    cv2.imshow('Color', vis)
                    cv2.imshow('GrayNorm', Grayscale/256.0)                    
                    Grayscale = cv2.medianBlur(Grayscale, 3)                    
                    Grayscale = cv2.medianBlur(Grayscale, 5)                    
                    FrameDiff_Gray =  numpy.abs(GrayscalePrev.astype("float") - Grayscale.astype("float")) / 256.0

                    if countDiffs == 0:
                        FrameDiff_GrayAggregate = FrameDiff_Gray
                    else:
                        FrameDiff_Grayt = FrameDiff_Gray
                        FrameDiff_Grayt[FrameDiff_Grayt < 0.05] = 0.0
                        FrameDiff_Grayt[FrameDiff_Grayt > 0.1] = 1.0
                        FrameDiff_GrayAggregate += FrameDiff_Grayt
                    #FrameDiff_GrayAggregate /= float(countDiffs+1)                    
                    #print FrameDiff_GrayAggregate
                    nrowsB = FrameDiff_Gray.shape[0] / 4
                    ncolsB = FrameDiff_Gray.shape[1] / 4
                    
                    FrameDiff_GrayBlocks = blockshaped(FrameDiff_Gray, nrowsB, ncolsB)
                    for isub in range(FrameDiff_GrayBlocks.shape[0]):                        
                        FrameDiff_GrayBlocks_sub = FrameDiff_GrayBlocks[isub,:,:]                        
                        if countDiffs==0:
                            FrameValue_Diffs_Grayscale.append([])                                                    
                        FrameValue_Diffs_Grayscale[isub].append(float(FrameDiff_GrayBlocks_sub.sum()))                        
                        #print numpy.append(FrameDiff_GrayBlocks_sub, float(FrameDiff_GrayBlocks_sub.sum()))
                        #FrameValue_Diffs_Grayscale = numpy.append(FrameValue_Diffs_Grayscale, float(FrameDiff_Gray.sum()))                                        
                    countDiffs += 1
                    cv2.imshow('Frame Diff', FrameDiff_Gray)                    
                    cv2.imshow('Frame Diff Agg',  (FrameDiff_GrayAggregate / FrameDiff_GrayAggregate.max()))
                    cv2.moveWindow('Color', 0, 0)
                    cv2.moveWindow('GrayNorm', newWidth , 0)
                    cv2.moveWindow('Frame Diff', 0 , 500)
                    cv2.moveWindow('Frame Diff Agg', newWidth , 500)

                    ch = cv2.waitKey(1)
                    T0 = T2;
                PROCESS_NOW = False
                GrayscalePrev = Grayscale;                

        else:
            break;
    
    #FrameDiff_GrayAggregate = cv2.cv.fromarray(FrameDiff_GrayAggregate)
    FrameDiff_GrayAggregate /= FrameDiff_GrayAggregate.max()
    FrameDiff_GrayAggregate *= 256;
    FrameDiff_GrayAggregate = cv2.medianBlur(FrameDiff_GrayAggregate.astype('uint8'), 3)
    FrameDiff_GrayAggregate = cv2.medianBlur(FrameDiff_GrayAggregate.astype('uint8'), 5)    
    FrameDiff_GrayAggregate[FrameDiff_GrayAggregate < 30] = 0.0


    processingTime = time.time() - Tstart
    processingFPS = countProcess / float(processingTime); 
    processingRT = 100.0 * float(processingTime) / (duration);

    seconds  = processingTime; Hours   = int(seconds/3600); Minutes = int(seconds/60); Seconds = int(seconds) % 60; Dsecs = int(100*(seconds - Seconds));


    FrameValue_Diffs_Grayscale = numpy.array(FrameValue_Diffs_Grayscale)
   #FrameValue_Diffs_Grayscale = FrameValue_Diffs_Grayscale.sum(axis=0)
    BlockBasedRepetitions = []
    maxFreqs = []
    for line in range(0,FrameValue_Diffs_Grayscale.shape[0]):
        temp = FrameValue_Diffs_Grayscale[line,:]
        nFFT = temp.shape[0]/2    
        '''TODO'''
        Xfft = abs(fft(temp))     
        Xfft = Xfft[0:nFFT]
        Xfft = Xfft / len(Xfft)
        max_temp,rep_temp = getNumOfRepetitions(Xfft, duration)
        maxFreqs.append(max_temp)
        BlockBasedRepetitions.append(rep_temp)
    print BlockBasedRepetitions    
    #estimatedRepetitions = math.floor(median(BlockBasedRepetitions))
    estimatedRepetitions = math.floor(mean(maxFreqs,BlockBasedRepetitions))

       # print estimatedRepetitions  
    #plt.subplot(3,1,1)
    #plt.plot(FrameValue_Diffs_Grayscale)    
    #plt.subplot(3,1,2)
    #plt.plot(Xfft)    
    #plt.subplot(3,1,3)
    #plt.plot(Xfft)
    #plt.show()
    

    #plt.imshow(FrameDiff_GrayAggregate)
    #plt.show()

    #plt.subplot(3,1,1)
    #plt.plot(FrameValue_Diffs_Grayscale)    
    #plt.subplot(3,1,2)
    #plt.plot(Xfft)    
    #plt.subplot(3,1,3)
    #plt.plot(Xfft)
    #plt.show()
    
    
    #return F, FeatureMatrix
    print moviePath
    scipy.misc.imsave(moviePath.replace("mpeg", "png"), FrameDiff_GrayAggregate)
    return estimatedRepetitions

def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

def getNumOfRepetitions(fftSeq, duration):
    nFFT = fftSeq.shape[0]
    fi = numpy.arange(0,nFFT)
    Fs = 1 / processStep  
    start = 5
    fftSeq[0:start] = 0    
    iMax = numpy.argmax(fftSeq)              
    maxFreq = (Fs/2)*fi[iMax] / float(nFFT)        
    maxT = 1 / maxFreq
    return maxFreq , round(duration / (2*maxT))

def dirProcessMovie(dirName):
    """
    """
    dirNameNoPath = os.path.basename(os.path.normpath(dirName))

    allFeatures = numpy.array([])

    types = ('*.avi', '*.mpeg',  '*.mpg', '*.mp4', '*.mkv')
    movieFilesList = []
    for files in types:
        movieFilesList.extend(glob.glob(os.path.join(dirName, files)))    
    movieFilesList = sorted(movieFilesList)
    
    for movieFile in movieFilesList:            
        repetitions = processMovie(movieFile, 2, 1)
        with open("repetitions_Wmean.csv", "a") as fp:
            wr = csv.writer(fp, dialect='excel')
            wr.writerow([movieFile.split('/')[1].split('.')[0],movieFile.split('/')[1].split('.')[0].split('_')[-1],repetitions])
    # TODO:
    # 1. save FrameDiff_GrayAggregate in a png file to be used for classification of the whole video (before processMovie() returns)
    # 2. save estimatedRepetitions in a csv file when runnign in a batch mode (see line 242)

        print movieFile, repetitions
        

def dirsProcessMovie(dirNames):
    # feature extraction for each class:
    features = [];
    classNames = []
    fileNames = []
    for i,d in enumerate(dirNames):
        [f, fn] = dirProcessMovie(d)
        if f.shape[0] > 0: # if at least one audio file has been found in the provided folder:
            features.append(f)
            fileNames.append(fn)
            if d[-1] == "/":
                classNames.append(d.split(os.sep)[-2])
            else:
                classNames.append(d.split(os.sep)[-1])
    return features, classNames, fileNames

def npyToCSV(fileNameFeatures, fileNameNames):
    F = numpy.load(fileNameFeatures)
    N = numpy.load(fileNameNames)
    fp = open(fileNameFeatures.replace(".npy",".csv"), 'w')
    for i in range(len(N)):
        fp.write(os.path.basename(os.path.normpath(N[i])) + "\t"), 
        for f in F[i]:
            fp.write("{0:.6f}\t".format(f))
        fp.write("\n")

def analyze(fileNameFeatures, fileNameNames, startF = 0, endF = 108, particularFeatures = []):
    f = 0
    F = numpy.load(fileNameFeatures)
    N = numpy.load(fileNameNames)

    text_file = open("ground_names.txt", "r")
    gtNames = lines = text_file.readlines();
    gtNames = [g.replace("\n","") for g in gtNames]
    gtSim = numpy.load("ground_sim_numpy")    

    # normalize    
    MEAN = numpy.mean(F, axis = 0); STD  = numpy.std(F, axis = 0)    
    for i in range(F.shape[0]):
        F[i,:] = (F[i] - MEAN) / STD    
    
    firstPos = []
    secondPos = []
    top10 = []
    top10_second = []

    for i in range(len(N)):         # for each movie                
        curName = os.path.basename(os.path.normpath(N[i])).replace(".mkv","").replace(".mpg","").replace(".mp4","").replace(".avi","")
        gtIndex = gtNames.index(curName)
        curGTSim = gtSim[gtIndex, :]
        curGTSim[gtIndex] = 0
        iGTmin = numpy.argmax(curGTSim)
        gtSorted = [x for (y,x) in sorted(zip(curGTSim, gtNames), reverse=True)]
        #print curName
        #for c in range(10):
        #    print "   " + gtSorted[c]        

        featuresToUse = range(startF, endF)
        if len(particularFeatures) > 0:
            featuresToUse = particularFeatures
        else:
            featuresToUse = range(startF,endF)

        #featuresToUse = [5, 6, 7, 13, 16, 17, 25, 26, 27, 32, 34, 41, 42, 43, 44, 47]
        #featuresToUse = [0,1, 3, 4, 5, 6, 7, 8, 10,11,12,13,14,15, 16, 17, 25, 28, 35, 42, 43, 44, 48, 53, 59, 66, 71, 78, 94, 95, 97, 98, 100, 101, 102]
        #featuresToUse = [106, 12, 36, 51, 3, 2, 89, 93, 65, 16, 96, 76, 25, 80, 20, 79, 72, 7, 60, 44]
        F[:,featuresToUse].shape
        d = dist.cdist(F[i, featuresToUse].reshape(1, len(featuresToUse)), F[:,featuresToUse])
        d[0][i] = 100000000
        d = d.flatten()
        #print d.shape, len(N)
        rSorted = [os.path.basename(os.path.normpath(x)).replace(".mkv","").replace(".mpg","").replace(".mp4","").replace(".avi","") for (y,x) in sorted(zip(d.tolist(), N))]        

        firstPos.append(gtSorted.index(rSorted[0]) + 1)
        secondPos.append(gtSorted.index(rSorted[1]) + 1)
        if rSorted[0] in gtSorted[0:10]:
            top10.append(1)
        else:
            top10.append(0)
        if rSorted[1] in gtSorted[0:10]:
            top10_second.append(1)
        else:
            top10_second.append(0)

        #print rSorted
        #print curName
        #for c in range(3):
        #    print  "         " + rSorted[c]        
        #print "{0:60s}\t{1:60s}".format( os.path.basename(os.path.normpath(N[i])), os.path.basename(os.path.normpath(N[numpy.argmin(d)])))
    #print numpy.median(numpy.array(firstPos)), 100*numpy.sum(numpy.array(top10)) / len(top10)
    return numpy.median(numpy.array(firstPos)), 100*numpy.sum(numpy.array(top10)) / len(top10), numpy.median(numpy.array(secondPos)), 100*numpy.sum(numpy.array(top10_second)) / len(top10_second)

def scriptAnalyze():
    nExp = 1000
    allFeatureCombinations = []
    medPos = []
    top10 = []
    medPos2 = []
    top102 = []

    T1 = 50
    T2 = 20
    for nFeatures in [5, 10, 20, 30, 40, 50, 60, 70]:
        print nFeatures
        for e in range(nExp):
            curFeatures = numpy.random.permutation(range(108))[0:nFeatures]
            allFeatureCombinations.append(curFeatures)
            a1, a2, a3, a4 = analyze("featuresAll.npy", "namesAll.npy", 0, 0, curFeatures)
            medPos.append(a1)
            top10.append(a2)
            medPos2.append(a3)
            top102.append(a4)

    medPos = numpy.array(medPos)
    top10 = numpy.array(top10)    
    medPos2 = numpy.array(medPos2)
    top102 = numpy.array(top102)    

    iMinPos = numpy.argmin(medPos)
    iMaxPos = numpy.argmax(top10)
    iMinPos2 = numpy.argmin(medPos2)
    iMaxPos2 = numpy.argmax(top102)

    for i in range(len(top10)):
        if (medPos[i] < T1) and (top10[i] > T2) and (medPos2[i] < T1) and (top102[i] > T2):
            print "{0:.1f}\t{1:.1f}\t{2:.1f}\t{3:.1f}".format(medPos[i], top10[i], medPos2[i], top102[i]),
            if i == iMinPos:
                print "BEST medPos\t",
            else:
                print "-----------\t",
            if i == iMaxPos:
                print "BEST top10\t",
            else:
                print "----------\t",
            if i == iMinPos2:
                print "BEST medPos2\t",
            else:
                print "------------\t",
            if i == iMaxPos2:
                print "BEST top102\t",
            else:
                print "-----------\t",

            for f in allFeatureCombinations[i]:
                print "{0:d},".format(f),                
            print

def main(argv):
    if len(argv)==3:
        if argv[1]=="-f":  # single file
            processMovie(argv[2], 2, 1)
    if argv[1]=="-d":      # directory
        dirName = argv[2]
        allFeatures, movieFilesList = dirProcessMovie(dirName)
        print allFeatures.shape, movieFilesList
        #F = dirsProcessMovie(dirNames)
    if argv[1]=="evaluate":
        [a, b, a2, b2] = analyze("featuresAll.npy", "namesAll.npy")
        print "First returned result median position {0:.1f}".format(a)
        print "First returned result in top10 {0:.1f} %".format(b)
        print "Second returned result median position {0:.1f}".format(a2)
        print "Second returned result in top10 {0:.1f} %".format(b2)

    if argv[1]=="scriptDebug":         
        scriptAnalyze()

if __name__ == '__main__':
    main(sys.argv)

