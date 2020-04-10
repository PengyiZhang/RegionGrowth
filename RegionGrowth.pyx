# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 18:50:26 2018
@author: kaany
---------------------------------------
Modified on Thursday April 9 08:56 2020
Support 2D grayscale image
@author: PengyiZhang 
"""

from collections import deque
import numpy as np
cimport numpy as np
cimport cython

# import logging
# logging.basicConfig(level=logging.debug)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class RegionGrow2D:
    cdef np.uint8_t[:,:] images
    cdef np.uint8_t[:,:] masks
    cdef np.uint8_t[:,:] outputMask

    cdef int sx, sy
    cdef int upperThreshold
    cdef int lowerThreshold
    cdef neighborMode
    cdef queue


    def __cinit__(self, np.uint8_t[:,:] images, np.uint8_t[:,:] masks,
                  int upperThreshold, int lowerThreshold, neighborMode):
        self.images = images
        self.masks = masks
        self.outputMask = np.zeros_like(self.images)

        self.sy = images.shape[0]
        self.sx = images.shape[1]

        self.lowerThreshold = lowerThreshold
        self.upperThreshold = upperThreshold
        self.neighborMode = neighborMode

        assert neighborMode in ["4n", "8n"], neighborMode

        self.queue = deque()

    def main(self, seeds, update=False):
        """
        Basically, seed is a pixel coordinate. However, we support many seeds here
        """
        cdef int newItem[2]
        for seed in seeds:
            newItem = seed
            self.outputMask[newItem[0], newItem[1]] = 1
            self.queue.append((seed[0], seed[1]))
        
        if update:
            self.updateThreshold()

        while len(self.queue) != 0:
            newItem = self.queue.pop()
            neighbors = self.getNeighbors(newItem)
            for neighbor in neighbors:
                self.checkNeighbour(neighbor[0], neighbor[1])
        return self.outputMask


    cdef checkNeighbour(self, int y, int x):
        cdef int intensity
        if (x < self.sx and y < self.sy
            and x > -1 and y > -1
            and self.masks[y,x] == 1):
            intensity = self.images[y, x]
            if self.isIntensityAcceptable(intensity) and self.outputMask[y,x] == 0:
                self.outputMask[y,x] = 1
                self.queue.append((y, x))
    
    cdef isIntensityAcceptable(self, int intensity):
        if intensity < self.upperThreshold and intensity > self.lowerThreshold:
            return True
        return False

    cdef updateThreshold(self, int lower_margin=3):
        """
        Update the lower and upper threshold dynamically 
        based on the statistics of known foreground pixel values
        """        
 
        cdef int mean, std 
        cdef np.ndarray[long long, ndim=1] sy, sx
        cdef np.ndarray[np.uint8_t, ndim=2] outputMask
        outputMask = np.asarray(self.outputMask)

        sy, sx = np.where(outputMask>0)
        cdef np.ndarray[np.uint8_t, ndim=1] intensities 

        intensities = np.asarray(self.images)[sy, sx]

        mean, std = np.mean(intensities), max(np.std(intensities), lower_margin)

        self.lowerThreshold = max((mean-3*std),0)
        self.upperThreshold = min((mean+3*std), 255)

    cdef int[:,:] getNeighbors(self, int[:] newItem):
        if self.neighborMode == "8n":
            neighbors = [
                [newItem[0]-1, newItem[1]],
                [newItem[0]+1, newItem[1]],
                [newItem[0], newItem[1]-1],
                [newItem[0], newItem[1]+1],
                [newItem[0]-1, newItem[1]-1],
                [newItem[0]+1, newItem[1]-1],
                [newItem[0]-1, newItem[1]+1],
                [newItem[0]+1, newItem[1]+1],                
            ] 
                                
        elif self.neighborMode == "4n":
            neighbors = [
                [newItem[0]-1, newItem[1]],
                [newItem[0]+1, newItem[1]],
                [newItem[0], newItem[1]-1],
                [newItem[0], newItem[1]+1],
            ]

        return np.array(neighbors)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class RegionGrow3D:
    cdef np.uint8_t[:,:,:] images
    cdef np.uint8_t[:,:,:] masks
    cdef np.uint8_t[:,:,:] outputMask

    cdef int sx, sy, sz
    cdef int upperThreshold
    cdef int lowerThreshold
    cdef neighborMode
    cdef queue
    
    def __cinit__(self, np.uint8_t[:,:,:] images, np.uint8_t[:,:,:] masks, 
                  int upperThreshold, int lowerThreshold, neighborMode):
        self.images = images
        self.masks = masks
        self.outputMask = np.zeros_like(self.images)
        self.sz = images.shape[0]
        self.sy = images.shape[1]
        self.sx = images.shape[2]

        self.upperThreshold = upperThreshold
        self.lowerThreshold = lowerThreshold
        self.neighborMode = neighborMode
        self.queue = deque()
    
    def main(self, int seeds, update=False):
        """
        seed: list of (z,y,x)
        """
        cdef int newItem[3]
        for seed in seeds:
            newItem = seed
            self.outputMask[newItem[0], newItem[1], newItem[2]] = 1
            self.queue.append((seed[0], seed[1], seed[2]))
        
        if update:
            self.updateThreshold()

        while len(self.queue) != 0:
            newItem = self.queue.pop()
            neighbors = self.getNeighbors(newItem)
            for neighbor in neighbors:
                self.checkNeighbour(neighbor[0], neighbor[1], neighbor[2])
        return self.outputMask

    cdef int[:,:,:] getNeighbors(self, int[:] newItem):
        if self.neighborMode == "26n":
            neighbors = [
                [newItem[0]-1, newItem[1]-1, newItem[2]-1],   [newItem[0]-1, newItem[1]-1, newItem[2]],   [newItem[0]-1, newItem[1]-1, newItem[2]+1],
                [newItem[0]-1, newItem[1], newItem[2]-1],     [newItem[0]-1, newItem[1], newItem[2]],     [newItem[0]-1, newItem[1], newItem[2]+1],
                [newItem[0]-1, newItem[1]+1, newItem[2]-1],   [newItem[0]-1, newItem[1]+1, newItem[2]],   [newItem[0]-1, newItem[1]+1, newItem[2]+1],
                [newItem[0], newItem[1]-1, newItem[2]-1],     [newItem[0], newItem[1]-1, newItem[2]],     [newItem[0], newItem[1]-1, newItem[2]+1],
                [newItem[0], newItem[1], newItem[2]-1],       [newItem[0], newItem[1], newItem[2]+1],     [newItem[0], newItem[1]+1, newItem[2]-1],
                [newItem[0], newItem[1]+1, newItem[2]],       [newItem[0], newItem[1]+1, newItem[2]+1],   [newItem[0]+1, newItem[1]-1, newItem[2]-1],
                [newItem[0]+1, newItem[1]-1, newItem[2]],     [newItem[0]+1, newItem[1]-1, newItem[2]+1], [newItem[0]+1, newItem[1], newItem[2]-1],
                [newItem[0]+1, newItem[1], newItem[2]],       [newItem[0]+1, newItem[1], newItem[2]+1],   [newItem[0]+1, newItem[1]+1, newItem[2]-1],
                [newItem[0]+1, newItem[1]+1, newItem[2]],     [newItem[0]+1, newItem[1]+1, newItem[2]+1]
            ] 
                                
        elif self.neighborMode == "6n":
            neighbors = [
                [newItem[0]-1, newItem[1], newItem[2]],
                [newItem[0]+1, newItem[1], newItem[2]],
                [newItem[0], newItem[1]-1, newItem[2]],
                [newItem[0], newItem[1]+1, newItem[2]],
                [newItem[0], newItem[1], newItem[2]-1],
                [newItem[0], newItem[1], newItem[2]+1],
            ]
        # 
        return np.array(neighbors)



    cdef checkNeighbour(self, int z, int y, int x):
        cdef int intensity
        if (x < self.sx and y < self.sy and z < self.sz 
            and x > -1 and y > -1 and z > -1 
            and self.masks[z,y,x] == 1):
            intensity = self.images[z, y, x]
            if self.isIntensityAcceptable(intensity) and self.outputMask[z,y,x] == 0:
                self.outputMask[z,y,x] = 1
                self.queue.append((z, y, x))
    
    cdef isIntensityAcceptable(self, int intensity):
        if intensity < self.upperThreshold and intensity > self.lowerThreshold:
            return True
        return False

    cdef updateThreshold(self, int lower_margin=3):
        """
        Update the lower and upper threshold dynamically 
        based on the statistics of known foreground pixel values
        """        
 
        cdef int mean, std 
        cdef np.ndarray[long long, ndim=1] sz, sy, sx
        cdef np.ndarray[np.uint8_t, ndim=3] outputMask

        outputMask = np.asarray(self.outputMask)
        sz, sy, sx = np.where(outputMask>0)

        cdef np.ndarray[np.uint8_t, ndim=1] intensities 

        intensities = np.asarray(self.images)[sz, sy, sx]

        mean, std = np.mean(intensities), max(np.std(intensities), lower_margin)

        self.lowerThreshold = max((mean-3*std),0)
        self.upperThreshold = min((mean+3*std), 255)



