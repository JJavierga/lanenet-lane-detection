
import cv2
import numpy as np

class bin2seg():
    def __init__(self):
        self.img=0
        [self.fils, self.cols]=[0,0]
        self.img_embedding=0
        self.used=0
        self.threshold=255


    def reuse(self,img,threshold):

        # padding
        [orig_fils,orig_cols]=img.shape
        border = np.zeros((orig_fils+2,orig_cols+2))
        #print(img.shape)
        #print(border.shape)
        #print(border[1:-2,1:-2].shape)
        #print(border[1:-1,1:-1].shape)
        border[1:-1,1:-1]=img
        self.img=border

        [self.fils, self.cols]=self.img.shape        
        self.used=np.zeros((self.fils,self.cols))
        self.img_embedding=self.used

        self.threshold=threshold


    def __del__(self):
        print('Object erased')


    def extend_colour(self,pos_f, pos_c, count):
        """

        :input: 
        """

        pendiente_f = [pos_f]
        pendiente_c = [pos_c]

        while pendiente_f:
            pos_f = pendiente_f[0]
            pos_c = pendiente_c[0]

            if (pos_f - 1 >= 0) and (pos_f + 1 <= self.fils - 1) and (pos_c - 1 >= 0) and (pos_c + 1 <= self.cols - 1):
                for i in range(-1,2):
                    for j in range(-1,2):
                        new_f = pos_f + i
                        new_c = pos_c + j
                        #print(new_f)
                        #print(new_c)
                        if(self.img[new_f, new_c]>=self.threshold and self.used[new_f, new_c]==False):
                            #print('G')
                            pendiente_f.append(new_f)
                            pendiente_c.append(new_c)
                            self.used[new_f, new_c]=True
                            self.img_embedding[new_f, new_c]= count*30
                            
            pendiente_f.pop(0)
            pendiente_c.pop(0)



    def colour_lines(self):
        """

        :input: image main path, images path new images path
        """
        """ This should be done by calling programm
        complete_path=ops.join(folder_path, subimg_path)

        #print(complete_path)

        img=cv2.imread(complete_path,cv2.IMREAD_GRAYSCALE)
        """

        count=1
        for col in range(self.cols):
            for fil in reversed(range(self.fils)):
                if self.img[fil,col]>=self.threshold and self.used[fil,col]==False:
                    #print('F')
                    self.extend_colour(fil, col, count)
                    count+=1

        #print(new_img_path)
        #print(folder_path)
        #print(subimg_path)

        """ This should be done by calling programm
        cv2.imwrite(new_img_path,img_embedding)
        """
        return self.img_embedding