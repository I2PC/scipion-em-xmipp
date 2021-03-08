# *****************************************************************************
# *
# * Authors:     Amaya Jimenez Moreno (ajimenez@cnb.csic.es)
# *
# * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# *
# *****************************************************************************

import os
import numpy
from pwem.constants import (ALIGN_2D, ALIGN_3D, ALIGN_PROJ, ALIGN_NONE)
import pwem.emlib.metadata as md
from pwem import emlib

from pwem.emlib.image import ImageHandler
from pwem.protocols import (ProtExtractMovieParticles, ProtProcessMovies)
from pyworkflow.protocol.constants import STEPS_PARALLEL
from pyworkflow.protocol.params import (PointerParam, IntParam, BooleanParam,
                                        Positive)
from pyworkflow.utils.path import cleanPath
from pyworkflow.utils import removeExt
from pwem.emlib.metadata import iterRows
from pwem.objects import Particle, Coordinate

from xmipp3.base import XmippMdRow
from xmipp3.convert import coordinateToRow
from xmipp3.convert import (readSetOfMovieParticles, xmippToLocation,
                            writeSetOfParticles, rowToParticle, particleToRow,
                            setXmippAttributes)


class XmippProtExtractMovieParticlesNew(ProtProcessMovies):
    """ Extract a set of Particles from each frame of a set of Movies.
    """
    _label = 'extract movie particles new'

    def __init__(self, **kwargs):
        ProtProcessMovies.__init__(self, **kwargs)
        self.stepsExecutionMode = STEPS_PARALLEL

    def _createFilenameTemplates(self):
        """ Centralize how files are called for iterations and references. """
        self.movieFolder = self._getTmpPath('movie_%(movieId)06d/')
        self.frameRoot = self.movieFolder + 'frame_%(frame)02d'
        myDict = {
            'frameImages': self.frameRoot + '_images',
            'frameMic': self.frameRoot + '.mrc',
            'frameMdFile': self.frameRoot + '_images.xmd',
            'frameCoords': self.frameRoot + '_coordinates.xmd',
            'frameStk': self.frameRoot + '_images.stk',
        }

        self._updateFilenamesDict(myDict)

    # --------------------------- DEFINE param functions ------------------------
    def _defineParams(self, form):
        ProtProcessMovies._defineParams(self, form)
        form.addParam('inputAlignMovieProt', PointerParam, important=True,
                      label="Input align movie protocol", pointerClass='XmippProtMovieCorr',
                      help='Select the protocol for movie alignment previously used.')
        form.addParam('inputParticles', PointerParam,
                      pointerClass='SetOfParticles',
                      important=True,
                      label='Input aligned particles')
        form.addParam('inputMicrographs', PointerParam,
                      pointerClass='SetOfMicrographs',
                      label='Input micrographs', important=True,
                      help='Select the micrographs to which you want to\n'
                           'associate the coordinates from the particles.')
        form.addParam('boxSize', IntParam, default=0,
                      label='Particle box size (px)', validators=[Positive],
                      help='In pixels. The box size is the size of the boxed '
                           'particles, actual particles may be smaller than '
                           'this.')
        #form.addParam('applyAlignment', BooleanParam, default=False,
        #              label='Apply movie alignments to extract?',
        #              help='If the input movies contains frames alignment, '
        #                   'you decide whether to use that information '
        #                   'for extracting the particles taking into account '
        #                   'the shifts between frames.')
        line = form.addLine('Frames range',
                            help='Specify the frames range to extract particles. '
                                 'The first frame is 1. If you set 0 in the '
                                 'last frame, it means that you will use until the '
                                 'last frame of the movie. If you apply the '
                                 'previous alignment of the movies, you only can use '
                                 'a frame range equal or less as used to alignment.')
        line.addParam('frame0', IntParam, label='First')
        line.addParam('frameN', IntParam, label='Last')

        form.addParallelSection(threads=3, mpi=1)

    # ------------------------- INSERT steps functions ------------------------

    def _insertAllSteps(self):
        self._createFilenameTemplates()

        # Build the list of all processMovieStep ids by
        # inserting each of the steps for each movie
        self.insertedDict = {}

        #A step to extract coordinates
        extractStep = self._insertFunctionStep('extractCoordinatesStep',
                                               prerequisites=[])

        # Conversion step is part of processMovieStep.
        movieSteps = []
        # For each movie insert the step to process it
        for movie in self.inputMovies.get():
            if movie.getObjId() not in self.insertedDict:
                movieDict = movie.getObjDict(includeBasic=True)
                stepId = self._insertFunctionStep('processMovieStep',
                                                       movieDict,
                                                       movie.hasAlignment(),
                                                       prerequisites=[extractStep])
                movieSteps.append(stepId)
                self.insertedDict[movie.getObjId()] = stepId

        # Do not use the extract particles finalStep method: wait = true.
        # finalSteps = self._insertFinalSteps(movieSteps)
        self._insertFunctionStep('createOutputStep',
                                 prerequisites=movieSteps)

        # -------------------------- STEPS functions -------------------------------

    def extractCoordinatesStep(self):
        inputParticles = self.inputParticles.get()
        inputMics = self.inputMicrographs.get()
        mdCoords = md.MetaData()
        rowCoord = XmippMdRow()
        alignType = inputParticles.getAlignment()

        scale = inputParticles.getSamplingRate() / inputMics.getSamplingRate()
        print("Scaling coordinates by a factor *%0.2f*" % scale)
        newCoord = Coordinate()
        firstCoord = inputParticles.getFirstItem().getCoordinate()
        hasMicName = firstCoord.getMicName() is not None

        # Create the micrographs dict using either micName or micId
        micDict = {}

        for mic in inputMics:
            micKey = mic.getMicName() if hasMicName else mic.getObjId()
            if micKey in micDict:
                print(">>> ERROR: micrograph key %s is duplicated!!!" % micKey)
                print("           Used in micrographs:")
                print("           - %s" % micDict[micKey].getLocation())
                print("           - %s" % mic.getLocation())
                raise Exception("Micrograph key %s is duplicated!!!" % micKey)
            micDict[micKey] = mic.clone()

        for particle in inputParticles:
            coord = particle.getCoordinate()
            micKey = coord.getMicName() if hasMicName else coord.getMicId()
            mic = micDict.get(micKey, None)

            if mic is None:
                print("Skipping particle, key %s not found" % micKey)
            else:
                newCoord.copyObjId(particle)
                x, y = coord.getPosition()
                #if inputParticles.hasAlignmentProj() and False:
                ##TODO: check if we need this condition or not, now it seems we dont need it
                #    shifts = self.getShifts(particle.getTransform(), alignType)
                #    xCoor, yCoor = x - int(shifts[0]), y - int(shifts[1])
                #else:
                #    xCoor, yCoor = x, y
                xCoor, yCoor = x, y
                newCoord.setPosition(xCoor * scale, yCoor * scale)
                newCoord.setMicrograph(mic)
                coordinateToRow(newCoord, rowCoord)
                rowCoord.writeToMd(mdCoords, mdCoords.addObject())
        mdCoords.write(self._getExtraPath('input_coords.xmd'))


    def _processMovie(self, movie):

        movId = movie.getObjId()
        #print("AQUI", movId, movie.getDim(), movie.getFramesRange())
        ih = emlib.image.ImageHandler()
        x, y, z, n = ih.getDimensions(movie.getFileName())
        print("AQUI", movId, x, y, z, n, movie.getFramesRange())
        #x, y, n = movie.getDim()
        iniFrame, lastFrame, _ = movie.getFramesRange()
        frame0, frameN = self._getRange(movie)
        boxSize = self.boxSize.get()

        if movie.hasAlignment():
            shiftX, shiftY = movie.getAlignment().getShifts()  # lists.
        else:
            shiftX = [0] * (lastFrame - iniFrame + 1)
            shiftY = shiftX

        stkIndex = 0
        movieStk = self._getMovieName(movie, '.stk')
        movieMdFile = self._getMovieName(movie, '.xmd')
        movieMd = md.MetaData()
        frameMd = md.MetaData()
        frameMdImages = md.MetaData()
        frameRow = md.Row()

        if self._hasCoordinates(movie):
            imgh = ImageHandler()

            for frame in range(frame0, frameN + 1):
                indx = frame - iniFrame
                frameName = self._getFnRelated('frameMic', movId, frame)
                frameMdFile = self._getFnRelated('frameMdFile', movId, frame)
                coordinatesName = self._getFnRelated('frameCoords', movId,
                                                     frame)
                frameImages = self._getFnRelated('frameImages', movId, frame)
                frameStk = self._getFnRelated('frameStk', movId, frame)


                self.info("Writing frame: %s" % frameName)
                # TODO: there is no need to write the frame and then operate
                # the input of the first operation should be the movie
                movieName = imgh.fixXmippVolumeFileName(movie)
                imgh.convert((frame, movieName), frameName)

                self.info("Extracting particles")

                frameNum = frame-1
                fnRoot = self.inputAlignMovieProt.get()._getExtraPath()
                fnMovie = removeExt(movie.getFileName())
                fnAlign = fnRoot + '/' + fnMovie.split('/')[-1] + '_shifts.xmd'
                fnOutFile = self.inputAlignMovieProt.get()._getExtraPath('auxOutputFile.txt')


                try:
                    #print("Applying " + "localAlignment@" + fnAlign)
                    mdAlign = md.MetaData("localAlignment@"+fnAlign)
                    shiftX = [0] * (lastFrame - iniFrame + 1)
                    shiftY = shiftX
                    self._writeXmippPosFile(movie, coordinatesName,
                                            shiftX[indx], shiftY[indx])
                    args = '-i %(frameName)s --pos %(coordinatesName)s ' \
                           '-o %(frameImages)s --Xdim %(boxSize)d --Xdmov %(x)d ' \
                           '--Ydmov %(y)d --Ndmov %(n)d --curFrame %(frameNum)d ' \
                           '-iAlign %(fnAlign)s ' % locals()
                    args += " --downsampling %f " % self.getBoxScale()
                    self.runJob('xmipp_micrograph_scissor_new', args)
                except:
                    print("WARNING: No applying new version of xmipp_micrograph_scissor for " + frameName)
                    self._writeXmippPosFile(movie, coordinatesName,
                                            shiftX[indx], shiftY[indx])
                    args = '-i %(frameName)s --pos %(coordinatesName)s ' \
                           '-o %(frameImages)s --Xdim %(boxSize)d ' % locals()
                    args += " --downsampling %f " % self.getBoxScale()
                    self.runJob('xmipp_micrograph_scissor', args)



                cleanPath(frameName)

                self.info("Combining particles into one stack.")

                frameMdImages.read(frameMdFile)
                frameMd.read('particles@%s' % coordinatesName)
                frameMd.merge(frameMdImages)

                for objId in frameMd:
                    stkIndex += 1
                    frameRow.readFromMd(frameMd, objId)
                    location = xmippToLocation(frameRow.getValue(md.MDL_IMAGE))
                    newLocation = (stkIndex, movieStk)
                    imgh.convert(location, newLocation)

                    # Fix the name to be accessible from the Project directory
                    # so we know that the movie stack file will be moved
                    # to final particles folder
                    newImageName = '%d@%s' % newLocation
                    frameRow.setValue(md.MDL_IMAGE, newImageName)
                    frameRow.setValue(md.MDL_MICROGRAPH_ID, int(movId))
                    frameRow.setValue(md.MDL_MICROGRAPH, str(movId))
                    frameRow.setValue(md.MDL_FRAME_ID, int(frame))
                    frameRow.setValue(md.MDL_PARTICLE_ID,
                                      frameRow.getValue(md.MDL_ITEM_ID))
                    frameRow.writeToMd(movieMd, movieMd.addObject())
                movieMd.addItemId()
                movieMd.write(movieMdFile)
                cleanPath(frameStk)


    def createOutputStep(self):
        inputMovies = self.inputMovies.get()
        particleSet = self._createSetOfMovieParticles()
        particleSet.copyInfo(inputMovies)

        # Create a folder to store the resulting micrographs

        #         particleFolder = self._getPath('particles')
        #         makePath(particleFolder)
        mData = md.MetaData()
        mdAll = md.MetaData()

        self._micNameDict = {}

        for movie in inputMovies:
            self._micNameDict[movie.getObjId()] = movie.getMicName()
            movieName = self._getMovieName(movie)
            movieStk = movieName.replace('.mrc', '.stk')
            movieMdFile = movieName.replace('.mrc', '.xmd')

            # Store particle stack and metadata files in final particles folder
            if os.path.exists(movieStk):
                mData.read(movieMdFile)
                mdAll.unionAll(mData)

        particleMd = self._getPath('movie_particles.xmd')
        mdAll.addItemId()
        mdAll.write(particleMd)

        if not self.inputParticles.get().hasAlignmentProj():
            readSetOfMovieParticles(particleMd, particleSet,
                                    removeDisabled=False,
                                    postprocessImageRow=self._postprocessImageRow)
            self._defineOutputs(outputParticles=particleSet)
            self._defineSourceRelation(self.inputMovies, particleSet)

        else:
            frame0, frameN = self._getRange(movie)
            imgsFn = self._getExtraPath('input_particles.xmd')
            inputPart = self.inputParticles.get()
            writeSetOfParticles(inputPart, imgsFn)

            particleSetOut = self._createSetOfMovieParticles()
            particleSetOut.copyInfo(inputPart)
            particleSetOut.setAlignmentProj()

            self.writeNewOutputMetadata(imgsFn, particleMd, self._getPath('movie_particles_new.xmd'), frameN, frame0)

            mdInputParts = md.MetaData(imgsFn)
            mdOutputParts = md.MetaData(particleMd)
            mdFinal = md.MetaData()
            rowsInputParts = iterRows(mdInputParts)
            for rowIn in rowsInputParts:
                partIn = rowToParticle(rowIn)
                if partIn.hasCTF():
                    ctfModel = partIn.getCTF()
                idIn = rowIn.getValue(md.MDL_ITEM_ID)
                shiftXIn = rowIn.getValue(emlib.MDL_SHIFT_X)
                shiftYIn = rowIn.getValue(emlib.MDL_SHIFT_Y)
                rotIn = rowIn.getValue(emlib.MDL_ANGLE_ROT)
                tiltIn = rowIn.getValue(emlib.MDL_ANGLE_TILT)
                psiIn = rowIn.getValue(emlib.MDL_ANGLE_PSI)
                flipIn = rowIn.getValue(emlib.MDL_FLIP)
                ccIn = rowIn.getValue(emlib.MDL_MAXCC)
                costIn = rowIn.getValue(emlib.MDL_COST)
                wIn = rowIn.getValue(emlib.MDL_WEIGHT)
                contXIn = rowIn.getValue(emlib.MDL_CONTINUOUS_X)
                contYIn = rowIn.getValue(emlib.MDL_CONTINUOUS_Y)

                count = 0
                rowsOutputParts = iterRows(mdOutputParts)

                for rowOut in rowsOutputParts:
                    if rowOut.getValue(md.MDL_PARTICLE_ID) == idIn:
                        partId = rowOut.getValue(md.MDL_PARTICLE_ID)
                        frId = rowOut.getValue(md.MDL_FRAME_ID)
                        partOut = rowToParticle(rowOut)
                        if partIn.hasCTF():
                            partOut.setCTF(ctfModel)
                        # setXmippAttributes(partOut, rowIn, md.MDL_SHIFT_X)
                        # setXmippAttributes(partOut, rowIn, md.MDL_SHIFT_Y)
                        # setXmippAttributes(partOut, rowIn, md.MDL_ANGLE_ROT)
                        # setXmippAttributes(partOut, rowIn, md.MDL_ANGLE_TILT)
                        # setXmippAttributes(partOut, rowIn, md.MDL_ANGLE_PSI)
                        # setXmippAttributes(partOut, rowIn, md.MDL_FLIP)

                        rowOutFinal = md.Row()
                        particleToRow(partOut, rowOutFinal)
                        rowOutFinal.setValue(md.MDL_PARTICLE_ID, int(partId))
                        rowOutFinal.setValue(md.MDL_FRAME_ID, int(frId))
                        rowOutFinal.setValue(emlib.MDL_SHIFT_X, shiftXIn)
                        rowOutFinal.setValue(emlib.MDL_SHIFT_Y, shiftYIn)
                        rowOutFinal.setValue(emlib.MDL_ANGLE_ROT, rotIn)
                        rowOutFinal.setValue(emlib.MDL_ANGLE_TILT, tiltIn)
                        rowOutFinal.setValue(emlib.MDL_ANGLE_PSI, psiIn)
                        rowOutFinal.setValue(emlib.MDL_FLIP, flipIn)
                        if contXIn is not None:
                            rowOutFinal.setValue(emlib.MDL_CONTINUOUS_X, contXIn)
                        if contYIn is not None:
                            rowOutFinal.setValue(emlib.MDL_CONTINUOUS_Y, contYIn)
                        if ccIn is not None:
                            rowOutFinal.setValue(emlib.MDL_MAXCC, ccIn)
                        if costIn is not None:
                            rowOutFinal.setValue(emlib.MDL_COST, costIn)
                        if wIn is not None:
                            rowOutFinal.setValue(emlib.MDL_WEIGHT, wIn)
                        rowOutFinal.addToMd(mdFinal)
                        count += 1
                        if count == (frameN - frame0 + 1):
                            break
            mdFinal.write(particleMd)

            readSetOfMovieParticles(particleMd, particleSetOut,
                                    removeDisabled=False,
                                    postprocessImageRow=self._postprocessImageRow)
            self._defineOutputs(outputParticles=particleSetOut)
            self._defineSourceRelation(self.inputMovies, particleSetOut)


    def writeNewOutputMetadata(self, imgsFn, particleMd, fnOut, frameN, frame0):

        mdInputParts = md.MetaData(imgsFn)
        mdOutputParts = md.MetaData(particleMd)
        mdFinal = md.MetaData()
        rowsInputParts = iterRows(mdInputParts)
        for rowIn in rowsInputParts:
            partIn = rowToParticle(rowIn)
            if partIn.hasCTF():
                ctfModel = partIn.getCTF()
            idIn = rowIn.getValue(md.MDL_ITEM_ID)
            shiftXIn = rowIn.getValue(emlib.MDL_SHIFT_X)
            shiftYIn = rowIn.getValue(emlib.MDL_SHIFT_Y)
            rotIn = rowIn.getValue(emlib.MDL_ANGLE_ROT)
            tiltIn = rowIn.getValue(emlib.MDL_ANGLE_TILT)
            psiIn = rowIn.getValue(emlib.MDL_ANGLE_PSI)
            flipIn = rowIn.getValue(emlib.MDL_FLIP)
            ccIn = rowIn.getValue(emlib.MDL_MAXCC)
            costIn = rowIn.getValue(emlib.MDL_COST)
            wIn = rowIn.getValue(emlib.MDL_WEIGHT)

            count = 0
            rowsOutputParts = iterRows(mdOutputParts)
            vectorIdx=[]
            flagenable=1
            for rowOut in rowsOutputParts:
                if count<(frameN-frame0) and rowOut.getValue(md.MDL_PARTICLE_ID) == idIn:
                    imName = rowOut.getValue(emlib.MDL_IMAGE)
                    vectorIdx.append(float(imName.split('@')[0]))
                    if rowOut.getValue(md.MDL_ENABLED)==0:
                        flagenable=0
                if count==(frameN-frame0) and rowOut.getValue(md.MDL_PARTICLE_ID) == idIn:
                    imName = rowOut.getValue(emlib.MDL_IMAGE)
                    vectorIdx.append(float(imName.split('@')[0]))
                    partId = rowOut.getValue(md.MDL_PARTICLE_ID)
                    frId = rowOut.getValue(md.MDL_FRAME_ID)
                    partOut = rowToParticle(rowOut)
                    if rowOut.getValue(md.MDL_ENABLED)==0:
                        flagenable=0
                    if partIn.hasCTF():
                        partOut.setCTF(ctfModel)
                    rowOutFinal = md.Row()
                    particleToRow(partOut, rowOutFinal)
                    rowOutFinal.setValue(md.MDL_PARTICLE_ID, int(partId))
                    rowOutFinal.setValue(md.MDL_FRAME_ID, int(frId))
                    rowOutFinal.setValue(emlib.MDL_SHIFT_X, shiftXIn)
                    rowOutFinal.setValue(emlib.MDL_SHIFT_Y, shiftYIn)
                    rowOutFinal.setValue(emlib.MDL_ANGLE_ROT, rotIn)
                    rowOutFinal.setValue(emlib.MDL_ANGLE_TILT, tiltIn)
                    rowOutFinal.setValue(emlib.MDL_ANGLE_PSI, psiIn)
                    rowOutFinal.setValue(emlib.MDL_FLIP, flipIn)
                    rowOutFinal.setValue(emlib.MDL_ENABLED, flagenable)
                    if ccIn is not None:
                        rowOutFinal.setValue(emlib.MDL_MAXCC, ccIn)
                    if costIn is not None:
                        rowOutFinal.setValue(emlib.MDL_COST, costIn)
                    if wIn is not None:
                        rowOutFinal.setValue(emlib.MDL_WEIGHT, wIn)
                    rowOutFinal.setValue(emlib.MDL_DM3_VALUE, vectorIdx)
                    rowOutFinal.addToMd(mdFinal)
                    break
                if rowOut.getValue(md.MDL_PARTICLE_ID) == idIn:
                    count+=1
        mdFinal.write(fnOut)

        # --------------------------- INFO functions ------------------------------

    def _validate(self):
        errors = []

        inputSet = self.inputMovies.get()

        # Although getFirstItem is not recommended in general, here it is
        # used only once, for validation purposes, so performance
        # problems should not appear.
        movie = inputSet.getFirstItem()
        if (not movie.hasAlignment()) and self.applyAlignment:
            errors.append("Your movies has not alignment. Please, set *No* "
                          "the parameter _Apply movie alignments to extract?_")

        firstFrame, lastFrame, _ = inputSet.getFramesRange()
        if lastFrame == 0:
            # Although getFirstItem is not recommended in general, here it is
            # used only once, for validation purposes, so performance
            # problems should not appear.
            frames = inputSet.getFirstItem().getNumberOfFrames()
            lastFrame = frames
        else:
            frames = lastFrame - firstFrame + 1

        if frames is not None:
            # Avoid validation when the range is not defined
            if not (hasattr(self, 'frame0') or hasattr(self, 'frameN')):
                return

            f0, fN = self._getRange(movie)
            if fN < firstFrame or fN > lastFrame:
                errors.append("Check the selected last frame. "
                              "Last frame (%d) should be in range: %s "
                              % (fN, (firstFrame, lastFrame)))

            if f0 < firstFrame or f0 > lastFrame:
                errors.append("Check the selected first frame. "
                              "First frame (%d) should be in range: %s "
                              % (f0, (firstFrame, lastFrame)))
            if fN < f0:
                errors.append("Check the selected frames range. Last frame "
                              "(%d) should be greater than first frame (%d)"
                              % (fN, f0))

        return errors

    def _warnings(self):
        warnings = []
        if not self.inputParticles.get().hasAlignmentProj():
            warnings.append("Running the extraction with particles without "
                            "alignment implies not using the shifts in the "
                            "transformation matrix to calculate the exact "
                            "coordinates of every particle.")
        return warnings

    def _methods(self):
        methods = []
        return methods

    def _summary(self):
        summary = []
        return summary

    # -------------------------- UTILS functions ------------------------------

    def _getFnRelated(self, keyFile, movId, frameIndex):
        return self._getFileName(keyFile, movieId=movId, frame=frameIndex)

    def _writeXmippPosFile(self, movie, coordinatesName, shiftX, shiftY):
        """ Create Xmipp coordinate files to be extracted
        from the frames of the movie.
        """
        mData = md.MetaData()
        mdCoords = md.MetaData(self._getExtraPath('input_coords.xmd'))
        for row in iterRows(mdCoords):
            if int(row.getValue(md.MDL_MICROGRAPH))==movie.getObjId():
                newX = row.getValue(md.MDL_XCOOR) - int(round(float(shiftX)))
                row.setValue(md.MDL_XCOOR, newX)
                newY = row.getValue(md.MDL_YCOOR) - int(round(float(shiftY)))
                row.setValue(md.MDL_YCOOR, newY)
                row.writeToMd(mData, mData.addObject())
        self.info("Writing coordinates metadata: %s, with shifts: %s %s"
                  % (coordinatesName, shiftX, shiftY))
        mData.write('particles@' + coordinatesName)

    def _postprocessImageRow(self, img, imgRow):
        img.setFrameId(imgRow.getValue(md.MDL_FRAME_ID))
        img.setParticleId(imgRow.getValue(md.MDL_PARTICLE_ID))
        micName = self._micNameDict[imgRow.getValue(md.MDL_MICROGRAPH_ID)]
        img.getCoordinate().setMicName(micName)

    def _getRange(self, movie):
        n = self._getNumberOfFrames(movie)
        iniFrame, _, indxFrame = movie.getFramesRange()

        first = self.getAttributeValue('frame0')
        last = self.getAttributeValue('frameN')

        if first <= 1:
            first = 1

        if last <= 0:
            last = n

        if iniFrame != indxFrame:
            first -= (iniFrame - 1)
            last -= (iniFrame - 1)

        return first, last

    def _getNumberOfFrames(self, movie):
        _, lstFrame, _ = movie.getFramesRange()

        if movie.hasAlignment():
            _, lastFrmAligned = movie.getAlignment().getRange()
            if lastFrmAligned != lstFrame:
                return lastFrmAligned
            else:
                return movie.getNumberOfFrames()
        else:
            return movie.getNumberOfFrames()

    def getBoxScale(self):
        """ Computing the sampling factor between input and output.
        We should take into account the differences in sampling rate between
        micrographs used for picking and the ones used for extraction.
        The downsampling factor could also affect the resulting scale.
        """
        samplingPicking = self.inputMicrographs.get().getSamplingRate()
        samplingExtract = self.inputMovies.get().getSamplingRate()
        return samplingPicking / samplingExtract

    def _hasCoordinates(self, movie):
        len = 0
        mdCoords = md.MetaData(self._getExtraPath('input_coords.xmd'))
        for row in iterRows(mdCoords):
            if int(row.getValue(md.MDL_MICROGRAPH))==movie.getObjId():
                len += 1
                break
        if len > 0:
            return True
        else:
            return False


    def getShifts(self, transform, alignType):
        """
        is2D == True-> matrix is 2D (2D images alignment)
                otherwise matrix is 3D (3D volume alignment or projection)
        invTransform == True  -> for xmipp implies projection
                              -> for xmipp implies alignment
        """
        if alignType == ALIGN_NONE:
            return None

        inverseTransform = alignType == ALIGN_PROJ
        # only flip is meaningful if 2D case
        # in that case the 2x2 determinant is negative
        flip = False
        matrix = transform.getMatrix()
        if alignType == ALIGN_2D:
            # get 2x2 matrix and check if negative
            flip = bool(numpy.linalg.det(matrix[0:2, 0:2]) < 0)
            if flip:
                matrix[0, :2] *= -1.  # invert only the first two columns keep x
                matrix[2, 2] = 1.  # set 3D rot
            else:
                pass

        elif alignType == ALIGN_3D:
            flip = bool(numpy.linalg.det(matrix[0:3, 0:3]) < 0)
            if flip:
                matrix[0, :4] *= -1.  # now, invert first line including x
                matrix[3, 3] = 1.  # set 3D rot
            else:
                pass

        else:
            pass
            # flip = bool(numpy.linalg.det(matrix[0:3,0:3]) < 0)
            # if flip:
            #    matrix[0,:4] *= -1.#now, invert first line including x
        shifts = self.geometryFromMatrix(matrix, inverseTransform)

        return shifts

    def geometryFromMatrix(self, matrix, inverseTransform):
        from pyworkflow.em.convert.transformations import translation_from_matrix
        if inverseTransform:
            matrix = numpy.linalg.inv(matrix)
            shifts = -translation_from_matrix(matrix)
        else:
            shifts = translation_from_matrix(matrix)
        return shifts



