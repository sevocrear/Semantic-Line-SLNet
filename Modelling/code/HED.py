import cv2
import os
import argparse

class CropLayer(object):
    def __init__(self,   params, blobs):
        # initialize our starting and ending (x, y)-coordinates of
        # the crop
        self.startX = 0
        self.startY = 0
        self.endX = 0
        self.endY = 0

    def getMemoryShapes(self, inputs):
        # the crop layer will receive two inputs -- we need to crop
        # the first input blob to match the shape of the second one,
        # keeping the batch size and number of channels
        (inputShape, targetShape) = (inputs[0], inputs[1])
        (batchSize, numChannels) = (inputShape[0], inputShape[1])
        (H, W) = (targetShape[2], targetShape[3])
        # compute the starting and ending crop coordinates
        self.startX = int((inputShape[3] - targetShape[3]) / 2)
        self.startY = int((inputShape[2] - targetShape[2]) / 2)
        self.endX = self.startX + W
        self.endY = self.startY + H
        # return the shape of the volume (we'll perform the actual
        # crop during the forward pass
        return [[batchSize, numChannels, H, W]]
    def forward(self, inputs):
        # use the derived (x, y)-coordinates to perform the crop
        return [inputs[0][:, :, self.startY:self.endY,
                self.startX:self.endX]]

class HED():
    '''
    Holystically-Nested Edge Detection Implementation
    '''
    def __init__(self, edge_detector_path) -> None:
        # load our serialized edge detector from disk
        print("[INFO] loading edge detector...")
        protoPath = os.path.sep.join([edge_detector_path,
            "deploy.prototxt"])
        modelPath = os.path.sep.join([edge_detector_path,
            "hed_pretrained_bsds.caffemodel"])
        self.net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

        # register our new layer with the model
        cv2.dnn_registerLayer("Crop", CropLayer)
        
    def detect(self, image, des_shape = (400, 400)):
        '''
        detect edges
        
        Inputs:
            image (np.array) - input image
            
            des_shape (Tuple) - desired shape of the HED
            
        Outputs:
        
            hed (np.array) - HED
        '''
        # load the input image and grab its dimensions
        (H, W) = image.shape[:2]

        # construct a blob out of the input image for the Holistically-Nested
        # Edge Detector
        blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(W, H),
            mean=(104.00698793, 116.66876762, 122.67891434),
            swapRB=False, crop=False)
        # set the blob as the input to the network and perform a forward pass
        # to compute the edges
        self.net.setInput(blob)
        hed = self.net.forward()
        hed = cv2.resize(hed[0, 0], (des_shape[0], des_shape[1]))
        hed = (255 * hed).astype("uint8")
        return hed