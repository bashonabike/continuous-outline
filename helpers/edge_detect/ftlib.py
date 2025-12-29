import cv2
import numpy as np

show_images = 1

def fastThin(img_ori):
  """
  Fast thinning of a binary image.

  This function applies a series of morphological operations to thin a binary image.

  Parameters
  ----------
  img_ori : numpy array
      The binary image to be thinned.

  Returns
  -------
  numpy array
      The thinned binary image.

  """
  obj = 0
  bg = 255
  img = img_ori.copy()
  # if(show_images):
    # cv2.imshow('pre morphology',img)
  img = morphology(img)
  cleanCorners(img)
  # if(show_images):
    # cv2.imshow('morphology',img)
  eraseTwoByTwos(img)
  # if(show_images):
    # cv2.imshow('morphology + eraseTwoByTwos',img)
  eraseLadders(img)
  # if(show_images):
    # cv2.imshow('morphology + eraseTwoByTwos + eraseLadders',img)
  return img

def morphology(img):
  # inverts the image to execute easier operations (sum and subtraction)
  """
  Performs a series of morphological operations to thin a binary image.

  Parameters
  ----------
  img : numpy array
      The binary image to be thinned.

  Returns
  -------
  numpy array
      The thinned binary image.

  Notes
  -----
  The algorithm works by subtracting the result of a dilation operation from the original image and then adding the result of an erosion operation to the original image. This process is repeated until the result from the previous iteration is the same as the current iteration.

  The kernel used for the dilation and erosion operations is a 3x3 cross kernel.

  The image is inverted at the start and end of the algorithm to make the operations easier to execute.

  """
  a, img = cv2.threshold(img,100,255,cv2.THRESH_BINARY_INV)
  # generates 3 by 3 cross kernel
  kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3)) 
  # iteration counter
  iteration = 0
  print("Starting fast thin...")
  while 1:
    iteration += 1
    print("Running iteration",iteration,"of morphology") # "Running iteration x"
    #erosion
    last_img = img.copy()
    ero = cv2.erode(img,kernel,iterations = 1)
    #dilation
    dil = cv2.dilate(ero,kernel,iterations = 1)
    # result = original - dilated + eroded
    img -= dil
    img += ero
    # ends loop if result is the same from last iteration
    if cv2.compare(img, last_img, cv2.CMP_EQ).all():
      break
  a, img = cv2.threshold(img,100,255,cv2.THRESH_BINARY_INV) # inverts back the image
  return img

# came up with this one by myself, if anyone finds a better way of detecting/reducing interest areas please tell me
def eraseTwoByTwos(img):
  """
  Erases 2x2 regions of interest in a binary image.

  Parameters
  ----------
  img : numpy array
      The binary image to be processed.

  Returns
  -------
  numpy array
      The processed binary image with 2x2 regions of interest erased.

  Notes
  -----
  The algorithm works by iterating over the image and checking each 2x2 region if all the pixels are the same color (either object or background). If all the pixels are the same color, it erases the region.

  The algorithm also checks the 12 surrounding pixels of each 2x2 region to see if they are of a different color. If any of the surrounding pixels are of a different color, it doesn't erase the region.

  The image is not modified if no 2x2 regions of interest are found.

  """
  altura = img.shape[0]
  largura = img.shape[1]
  obj = 0
  bg = 255
  for y in range(1,altura-2):
    for x in range(1,largura-2):
      #centrais
      c1 = img[y,x]
      c2 = img[y,x+1]
      c3 = img[y+1,x]
      c4 = img[y+1,x+1]
      if(c1 == obj and c2 == obj and c3 == obj and c4 == obj):
        if img[y-1,x-1]!=obj:
          img[y,x] = bg
          pass
        elif img[y-1,x+2]!=obj:
          img[y,x+1]=bg
          pass
        elif img[y+2,x-1]!=obj:
          img[y+1,x]=bg
          pass
        elif img[y+2,x+2]!=obj:
          img[y+1,x+1]=bg
          pass
        #vizinhos
        v1  = img[y-1,x-1]
        v2  = img[y-1,x]
        v3  = img[y-1,x+1]
        v4  = img[y-1,x+2]
        v5  = img[y,x+2]
        v6  = img[y+1,x+2]
        v7  = img[y+2,x+2]
        v8  = img[y+2,x+1]
        v9  = img[y+2,x]
        v10 = img[y+2,x-1]
        v11 = img[y+1,x-1]
        v12 = img[y,x-1]
        vizinhos = [v12,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11]

# sets the borders of the image as bg
def cleanCorners(img):
  """
  Sets the borders of the image as bg.

  Parameters
  ----------
  img : numpy array
    The image to set its borders as bg.

  Returns
  -------
  None
  """
  altura = img.shape[0]
  largura = img.shape[1]
  bg = 255
  img[0:altura,0] = bg
  img[0,0:largura] = bg
  img[0:altura,largura-1] = bg
  img[altura-1,0:largura] = bg

# used in Zhang Suen algorithms to eliminate undesired corners
def eraseLadders(img):
  """
  Erases ladders from an image.

  Ladders are patterns of 2x2 pixels where all pixels have the same value and are surrounded by pixels of a different value. This function erases these patterns from an image.

  Parameters
  ----------
  img : numpy array
      The image to erase ladders from.

  Returns
  -------
  None

  """
  altura = img.shape[0]
  largura = img.shape[1]
  obj = 0
  bg = 255
  m1 = [[255,0,7  ],
        [0,  0,7  ],
        [7,  7,255]]
  m2 = [[7,  0,255],
        [7,  0,0  ],
        [255,7,7  ]]
  m3 = [[7,  7,255],
        [0,  0,7  ],
        [255,0,7  ]]
  m4 = [[255,7,7  ],
        [7,  0,0  ],
        [7,  0,255]]
  mask = [m1,m2,m3,m4]
  for y in range(1,altura-1):
    for x in range(1,largura-1):
      p5 = img[y,x]
      if p5 == obj:
        p1 = img[y-1,x-1]
        p2 = img[y-1,x]
        p3 = img[y-1,x+1]
        p4 = img[y,x-1]
        p6 = img[y,x+1]
        p7 = img[y+1,x-1]
        p8 = img[y+1,x]
        p9 = img[y+1,x+1]
        p = [[p1,p2,p3],
             [p4,p5,p6],
             [p7,p8,p9]]
        for m in mask:
          pairing = 1
          for i in range(0,3):
            for j in range(0,3):
              if m[i][j] != 7:
                if m[i][j] != p[i][j]:
                  pairing = 0
          if pairing:
            img[y,x] = bg
            break
