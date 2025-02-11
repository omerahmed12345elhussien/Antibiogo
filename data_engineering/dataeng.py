"""
* dataeng.py contains data cleaning, preparation, and analysis classes that are needed throughout the data pipeline.

"""
from os import path, listdir,makedirs,PathLike
import skimage.measure
from tqdm import tqdm
import cv2 as cv
import numpy as np
import pandas as pd
from natsort import index_natsorted,natsorted
from PIL import Image
import skimage
import tensorflow as tf
import tensorflow_models as tfm

class DataAnalysis:
    """Data analysis class for analyzing the images, masks, and compute IoU."""
    def __init__(self):
        pass
            
    def mask_unique_values(self,mask_dir,output_dir):
        """Get the frequency of [0,1] masks"""
        all_masks = listdir(mask_dir)
        if ".DS_Store" in all_masks : all_masks.remove(".DS_Store")
        img_name_list, value,frequency=[],[],[]
        for m_name in tqdm(all_masks):
            mask_full_path = path.join(mask_dir,m_name)
            img = cv.imread(mask_full_path,cv.IMREAD_GRAYSCALE)
            result = np.unique(img,return_counts=True)
            img_name_list.append(m_name)
            value.append(result[0])
            frequency.append(result[1])
            img = None
        df = pd.DataFrame({'Image_Name':img_name_list,'Values':value,'Frequency':frequency})
        df = df.sort_values(by='Image_Name',axis=0,key=lambda x: np.argsort(index_natsorted(df["Image_Name"])))
        df.reset_index(inplace=True)
        df.drop('index',axis=1,inplace=True)
        makedirs(output_dir,exist_ok=True)
        df.to_csv(path.join(output_dir,'Masks01_info.csv'))
        df,value,frequency,img_name_list=None,None,None,None
        
    def img_info(self, img_dir: str | PathLike, output_dir: str | PathLike, one_channel:bool=True)->None:
        """Get image information such as height and width; for the masks get further the white to black ratio of pixels
        and common descriptive statistics.

        Args:
            img_dir (str | PathLike): An input image directory.
            output_dir (str | PathLike): The directory to write the generated dataframe.
            one_channel (bool, optional): Is the passed image one channel or not. Defaults to True.
        """
        all_imgs = listdir(img_dir)
        if ".DS_Store" in all_imgs: all_imgs.remove(".DS_Store")
        if ".tmp.driveupload" in all_imgs:all_imgs.remove(".tmp.driveupload")
        h,w,w_r,img_name,img_shape = [],[],[],[],[]
        # Loop over the images.
        for file_name in tqdm(all_imgs):
            img_path = path.join(img_dir,file_name)
            img = cv.imread(img_path,cv.IMREAD_GRAYSCALE) if one_channel else cv.imread(img_path)
            if one_channel:
                white_pixel = np.sum(img>=245)
                white_ratio = round(white_pixel/(img.shape[0]*img.shape[1]),ndigits=3)
                w_r.append(white_ratio)
            h.append(img.shape[0]); w.append(img.shape[1]); img_name.append(file_name); img_shape.append(img.shape)
        data_dict = {'Img_Name':img_name,'Height':h,'Width':w,'Img_Shape':img_shape,
                     'White_Ratio':w_r} if one_channel else {'Img_Name':img_name,'Height':h,'Width':w,
                                                             'Img_Shape':img_shape}
        df = pd.DataFrame(data_dict)
        # Sort the dataframe using Img_Name column.
        df = df.sort_values(by='Img_Name',axis=0,key=lambda x: np.argsort(index_natsorted(df["Img_Name"])))
        df.reset_index(inplace=True)
        df.drop('index',axis=1,inplace=True)
        makedirs(output_dir,exist_ok=True)
        df_name = 'Mask' if one_channel else 'Image'
        df.to_csv(path.join(output_dir,f"{df_name}_info.csv"))
        h,w,data_dict,w_r,img_shape=None,None,None,None,None
        if one_channel:
            df_desc = df[['White_Ratio']].apply(self._desc_sta,axis=0)
            df_desc.reset_index(inplace=True)
            df_desc.rename(columns={"index":"Statistics"},inplace=True)
            df_desc.to_csv(path.join(output_dir,f"{df_name}_descriptive_stat.csv"))
            df_desc = None
            def _f(x):
                """Group White Ratio."""
                if x>=0 and x<.1: return 1
                elif x>=.1 and x<.2: return 2
                elif x>=.2 and x<.3: return 3
                elif x>=.3 and x<.4: return 4
                elif x>=.4 and x<.5: return 5
                elif x>=.5 and x<.6: return 6
                elif x>=.6 and x<.7: return 7
                elif x>=.7 and x<.8: return 8
                elif x>=.8 and x<.9: return 9
                else: return 10
            def _f_apply(df_row:pd.Series)->pd.Series: return pd.Series(_f(df_row['White_Ratio']),index=['G_WhiteRatio'])
            df1 = df[['White_Ratio']].apply(_f_apply,axis=1)
            df = None
            def _freq_wr(df_col: pd.Series)->pd.Series:
                return pd.Series([len(df_col[df_col==idx]) for idx in range(1,10)],
                                index=[f'{idx}' for idx in range(1,10)])
            #
            df_freq = df1.apply(_freq_wr,axis=0)
            df_freq.reset_index(inplace=True)
            df_freq.rename(columns={"index":"Groups"},inplace=True)
            df_freq.to_csv(path.join(output_dir,f"{df_name}_freq.csv"))
            df_freq = None
            def _perce_wr(df_col: pd.Series)->pd.Series:
                return pd.Series([round(100*len(df_col[df_col==idx])/len(df_col),2) for idx in range(1,10)],
                                index=[f'{idx}' for idx in range(1,10)])
            #
            df_perce = df1.apply(_perce_wr,axis=0)
            df_perce.reset_index(inplace=True)
            df_perce.rename(columns={"index":"Groups"},inplace=True)
            df_perce.to_csv(path.join(output_dir,f"{df_name}_perc.csv"))
            df_perce,df1 = None,None
        
    def read_img_metadata(self, img_dir: str | PathLike,
                          output_dir: str | PathLike,
                          DF_name: str = 'Rotation')->None:
        """Read Image metadata.

        Args:
            img_dir (str | PathLike): Image directory.
            output_dir (str | PathLike): Output directory.
            DF_name (str, optional): The desired output dataframe name. Defaults to 'Rotation'.
        """
        all_imgs = listdir(img_dir)
        if ".DS_Store" in all_imgs :all_imgs.remove(".DS_Store")
        rotation,image_name=[],[]
        for file_name in tqdm(all_imgs):
            image = Image.open(path.join(img_dir,file_name))
            if hasattr(image,'_getexif'):
                exif_data = image._getexif()
                if exif_data is not None:
                    if 0x0112 in exif_data:
                        orient=exif_data[0x0112]
                    else: orient = 000 # No orientation tag found in Exif data
                else: orient = 000
            else: orient = 9999 #Means No Efxif data found in the image
            rotation.append(orient);image_name.append(file_name)
        data_dict = {'Image_Name':image_name,'Rotation':rotation}
        df = pd.DataFrame(data_dict)
        df = df.sort_values(by='Img_Folder',axis=0)
        df.reset_index(inplace=True)
        df.drop('index',axis=1,inplace=True)
        df.to_csv(path.join(output_dir,f"{DF_name}_info.csv"))
        rotation,image_name=None,None
    
    def get_bbox_from_mask(self, mask: np.ndarray,
                           use_lcc: bool=False)->None:
        """Convert a single mask to bounding-box coordinates.

        Args:
            mask (np.ndarray): mask of values [0-255].
            use_lcc (bool, optional): Use largest connected component 
            to remove noise from the mask. Defaults to False.

        Returns:
            tuple[int,int,int,int]: tuple contains of x_min,x_max,y_min, and y_max.
        """
        # We assume the mask values in the range [0-255]
        mask1=mask.copy()
        mask1[mask1<=127.5]=0;mask1[mask1>127.5]=1
        # Remove noise using Largest Connected Component
        if use_lcc:
            labels = skimage.measure.label(mask1, return_num=False)
            maxCC_nobcg = labels == np.argmax(np.bincount(labels.flat, weights=mask1.flat))
            mask1=maxCC_nobcg
        segmentation = np.where(mask1==1)
        self.x_min = int(np.min(segmentation[1]))
        self.x_max = int(np.max(segmentation[1]))
        self.y_min = int(np.min(segmentation[0]))
        self.y_max = int(np.max(segmentation[0]))
        mask1,segmentation,maxCC_nobcg,labels=None,None,None,None
    
    def get_bbox_from_mask_st2(self,mask: np.ndarray)->None:
        """Convert a single mask to list of bounding boxes. We expect the mask has several objects
        which are the antibiotic disks, or a single object in the case of passing the petri dish.

        Args:
            mask (np.ndarray): Mask of value [0,1].
        """
        mask = skimage.measure.label(mask)
        props_mask = skimage.measure.regionprops(mask)
        self.bboxes = [[prop.bbox[1],prop.bbox[0],prop.bbox[3],prop.bbox[2]] for prop in props_mask]
        mask = None
    
    def bbox_from_model_pred(self, data_dir: str | PathLike,
                             save_result_path: str | PathLike,
                             df_name: str,
                             use_lcc: bool=False,
                             std_size: int=1024)->None:
        """It converts the generated masks of a model prediction to bounding boxes
        then writes them to a dataframe. The case of stage 1: only petri dish is detected.

        Args:
            data_dir (str | PathLike): Data directory.
            save_result_path (str | PathLike): Output directory.
            output_name (str): The name of the output dataframe.
            use_lcc (bool, optional): Use largest Connected Component to remove noise from the model prediction. Defaults to False.
            std_size (int, optional): Standard image size. Defaults to 1024.
        """
        makedirs(save_result_path,exist_ok=True)
        all_masks=listdir(data_dir)
        mask_list,X1,Y1,X2,Y2=[],[],[],[],[]
        if ".DS_Store" in all_masks : all_masks.remove(".DS_Store")  
        for mask_name in tqdm(all_masks):
            mask=cv.imread(path.join(data_dir,mask_name),cv.IMREAD_GRAYSCALE)
            mask=cv.resize(mask,(std_size,std_size),interpolation = cv.INTER_CUBIC)
            self.get_bbox_from_mask(mask,use_lcc)
            x_min,y_min,x_max,y_max = self.x_min,self.y_min,self.x_max,self.y_max
            mask_list.append(mask_name);X1.append(x_min);Y1.append(y_min)
            X2.append(x_max);Y2.append(y_max)
        data_dict={"Mask_Name":mask_list,"X1":X1,"Y1":Y1,"X2":X2,"Y2":Y2}
        df = pd.DataFrame(data_dict)
        df = df.sort_values(by='Mask_Name',axis=0,key=lambda x: np.argsort(index_natsorted(df["Mask_Name"])))
        df.reset_index(inplace=True)
        df.drop('index',axis=1,inplace=True)
        df.to_csv(path.join(save_result_path,df_name+'_bbox.csv'))
        mask_list,X1,Y1,X2,Y2=None,None,None,None,None
        
    def bbox_from_model_pred_st2(self,
                                 data_dir: str | PathLike,
                                 output_dir: str | PathLike,
                                 df_name:str,
                                 use_lcc: bool=False,
                                 only_st1:bool=False,
                                 only_st2: bool=False,
                                 std_size: int=1024)->None:
        """Convert the generated masks of a model prediction to bounding boxes then writes them to a dataframe.
        Here, we expect to have more than one label for the foreground. We expect the mask values to be [0,1,...,(max_no_classes-1)].

        Args:
            data_dir (str | PathLike): Data directory. 
            output_dir (str | PathLike): Output directory.
            df_name (str): dataframe name.
            use_lcc (bool): Use largest Connected Component to remove noise from the predicted mask. Defaults to False.
            only_st1 (bool): Use only petri dish labels (stage 1). Defaults to False.
            only_st2 (bool): Use only antibiotic disk labels (stage 2). Defaults to False.
            std_size (int, optional): Standard image size. Defaults to 1024.
        """
        makedirs(output_dir,exist_ok=True)
        all_masks = listdir(data_dir)
        if ".DS_Store" in all_masks : all_masks.remove(".DS_Store")
        mask_list,info_list,detect_obj_list=[],[None]*len(all_masks),[]
        for idx,mask_name in tqdm(enumerate(all_masks)):
            mask = cv.imread(path.join(data_dir,mask_name),cv.IMREAD_GRAYSCALE)
            mask = cv.resize(mask,(std_size,std_size),interpolation=cv.INTER_CUBIC)
            unique_values = np.unique(mask)
            temp_list=[]
            for i in unique_values:
                if i==0: continue
                elif i==1 and not only_st2:
                    mask_copy = mask.copy()
                    mask_copy[mask_copy!=0]=255
                    self.get_bbox_from_mask(mask_copy,use_lcc)
                    temp_list = temp_list +[[self.x_min,self.y_min,self.x_max,self.y_max]]
                    continue
                elif i==2 and not only_st1:
                    mask_copy = mask.copy()
                    mask_copy[mask_copy!=i] = 0
                    self.get_bbox_from_mask_st2(mask_copy)
                    temp_list = temp_list + self.bboxes
            mask_list.append(mask_name); detect_obj_list.append(len(temp_list))
            info_list[idx] = temp_list
            unique_values,mask,mask_copy=None,None,None
        df = pd.DataFrame({"Mask_Name":mask_list,"Info":info_list,"Detected_objects":detect_obj_list})
        df = df.sort_values(by='Mask_Name',axis=0,key=lambda x: np.argsort(index_natsorted(df["Mask_Name"])))
        df.reset_index(inplace=True)
        df.drop('index',axis=1,inplace=True)
        df.to_csv(path.join(output_dir,df_name+'_bbox.csv'))
        df,detect_obj_list,temp_list,info_list = None,None,None,None
    
    def _desc_sta(self, df_col:pd.Series)->pd.Series:
            """Compute Descriptive statistics."""
            return pd.Series([round(df_col.mean(),4),round(df_col.std(),4),
                              round(df_col.median(),4),
                              round(df_col.quantile(0.01,'midpoint'),4), # 1%
                              round(df_col.quantile(0.05,'midpoint'),4), # 5%
                              round(df_col.quantile(0.25,'midpoint'),4), # 25%
                              round(df_col.quantile(0.5,'midpoint'),4), # 50%
                              round(df_col.quantile(0.75,'midpoint'),4), # 75%
                              round(df_col.quantile(0.95,'midpoint'),4), # 95%
                              round(df_col.quantile(0.99,'midpoint'),4), # 99%
                              round(df_col.min(),4),
                              round(df_col.max(),4)],
                             index=['mean','std','median','1%','5%','25%','50%','75%','95%','99%','min','max']) 
    
    def compute_iou_stage1(self, gt_dir: str | PathLike,
                    improc_dir: str | PathLike,
                    model_names_df_dir: str | PathLike,
                    model_pred_dir: str | PathLike,
                    output_dir: str | PathLike,
                    USE_LCC:bool=True)->None:
        """Compute IoU for Improc and other models for stage 1(cropping the petri dish).

        Args:
            gt_dir (str | PathLike): Ground-truth dataframe directory.
            improc_dir (str | PathLike): Improc dataframe directory.
            model_names_df_dir (str | PathLike): Model names dataframe directory.
            model_pred_dir (str | PathLike): The predictions of all the models that need to be compared.
            output_dir (str | PathLike): The desired output directory.
            USE_LCC (bool, optional): Use largest connected component to remove noise from the
            masks. Defaults to True.
        """
        all_models_names = listdir(model_pred_dir)
        if ".DS_Store" in all_models_names : all_models_names.remove(".DS_Store") 
        makedirs(output_dir,exist_ok=True)
        for model_name in tqdm(all_models_names):
            self.bbox_from_model_pred(path.join(model_pred_dir,model_name),
                                      output_dir,
                                      model_name,
                                      USE_LCC)
        # Ground truth.
        df_gt=pd.read_csv(gt_dir)
        df_gt.drop('Unnamed: 0',axis=1,inplace=True)
        # Improc.
        df_improc=pd.read_csv(improc_dir,usecols=['Img_Name','X1','Y1','X2','Y2'])
        df_improc['Mask_Name']=df_improc.Img_Name.map(lambda p:f"{path.splitext(p)[0]}.png")
        df_improc.drop('Img_Name',axis=1,inplace=True)
        df_final=pd.merge(df_gt,df_improc,how='inner',on='Mask_Name',suffixes=(None, '_imp'))
        df_gt,df_improc=None,None  
        #
        model_names_df = pd.read_csv(model_names_df_dir)
        model_names_df.drop('Unnamed: 0',axis=1,inplace=True)
        for idx, model_name in enumerate(model_names_df['New_names']):
            df1=pd.read_csv(path.join(output_dir,f"{model_name}_bbox.csv"))
            df1.drop('Unnamed: 0',axis=1,inplace=True)
            df1['Mask_Name']=df1.Mask_Name.map(lambda p:f"{path.splitext(p)[0]}.png")
            df_final=pd.merge(df_final,df1,how='inner',on='Mask_Name',suffixes=(None, f"_{idx}"))
            df1=None
        df_cols = len(df_final.columns)//4
        new_col_names = ['Mask_Name','IoU_Improc']+[f'IoU_{idx}' for idx in range(df_cols-2)]
        #
        def _iou_apply(df_row:pd.Series)->pd.Series:
            """Compute IoU for Improc and other models."""
            # Ground truth bounding box.
            gt_bbox=tf.reshape(tf.constant([df_row.iloc[1:5]],dtype=tf.float32),[1, -1])
            # Improc's and other models' bounding boxes.
            improc_bbox=tf.reshape(tf.constant([df_row.iloc[5:]],dtype=tf.float32),[-1, 4])
            iou_computed=tfm.vision.iou_similarity.iou(gt_bbox,improc_bbox).numpy()#[0][0]
            output_list = [None]*df_cols
            output_list[0] = df_row['Mask_Name']
            for idx in range(1,df_cols):
                output_list[idx] = iou_computed[0][idx-1]
            return pd.Series(output_list,index=new_col_names)
        # Compute IoU for the bboxes.
        df2=df_final.apply(_iou_apply,axis=1)
        df_final=pd.merge(df_final,df2,how='inner',on='Mask_Name')
        # Compute Descriptive statistics for Improc IoU and other models.
        df2_desc = df2[new_col_names[1:]].apply(self._desc_sta,axis=0)
        df2_desc.reset_index(inplace=True)
        df2_desc.rename(columns={"index":"Statistics"},inplace=True)
        df2_desc.to_csv(path.join(output_dir,"desc_stat.csv"))
        df2,df2_desc = None,None
        def _f(x):
            """Group IoU."""
            if x>=0 and x<.5: return 1
            elif x>=.5 and x<.7: return 2
            elif x>=.7 and x<.9: return 3
            elif x>=.9 and x<.95: return 4
            else: return 5
        #
        col_names =['IoU_Improc']+[f'IoU_{idx}' for idx in range(df_cols-2)]
        new_col_names = ['Mask_Name','G_Improc']+[f'G_{idx}' for idx in range(df_cols-2)]
        #
        def _f_apply(df_row:pd.Series)->pd.Series:
            output_list = [None]*df_cols
            output_list[0] = df_row['Mask_Name']
            for idx,col_name in enumerate(col_names):
                output_list[idx+1] = _f(df_row[col_name])
            return pd.Series(output_list,index=new_col_names)
        df3=df_final.apply(_f_apply,axis=1)
        df_final=pd.merge(df_final,df3,how='inner',on='Mask_Name')
        col_names = None
        df_final.to_csv(path.join(output_dir,"IoU_comp.csv"))
        df_final = df3
        def _freq_iou(df_col: pd.Series)->pd.Series:
            return pd.Series([len(df_col[df_col==idx]) for idx in range(1,6)],
                             index=[f'{idx}' for idx in range(1,6)])
        #
        df3_freq = df_final[new_col_names[1:]].apply(_freq_iou,axis=0)
        df3_freq.reset_index(inplace=True)
        df3_freq.rename(columns={"index":"Groups"},inplace=True)
        df3_freq.to_csv(path.join(output_dir,"Freq_iou.csv"))
        df3_freq = None
        def _perce_iou(df_col: pd.Series)->pd.Series:
            return pd.Series([round(100*len(df_col[df_col==idx])/len(df_col),2) for idx in range(1,6)],
                             index=[f'{idx}' for idx in range(1,6)])
        #
        df3_perce = df_final[new_col_names[1:]].apply(_perce_iou,axis=0)
        df3_perce.reset_index(inplace=True)
        df3_perce.rename(columns={"index":"Groups"},inplace=True)
        df3_perce.to_csv(path.join(output_dir,"Perce_iou.csv"))
        df3_perce,df3,df_final = None,None,None
        
    def compute_iou_stage2(self,
                           gt_dir: str | PathLike,
                           improc_dir: str | PathLike,
                           model_names_df_dir: str | PathLike,
                           model_pred_dir: str | PathLike,
                           output_dir: str | PathLike,
                           USE_LCC:bool=True,
                           ONLY_ST1:bool=True,
                           ONLY_ST2:bool=False,
                           include_improc:bool=True)->None:
        """Compute IoU for Improc and other models for stage 1 & 2 masks
        (cropping the petri dish & Identifying the antibiotic disks).

        Args:
            gt_dir (str | PathLike): Ground-truth dataframe directory for stage 1 & 2.
            improc_dir (str | PathLike): Improc dataframe directory for stage 2 only.
            model_names_df_dir (str | PathLike): Model names dataframe directory.
            model_pred_dir (str | PathLike): The predictions of all the models that will be compared.
            output_dir (str | PathLike): The desired output directory.
            USE_LCC (bool, optional): Use largest connected component to remove noise from the
            masks. Defaults to True.
            ONLY_ST1 (bool, optional): Use only stage 1 data (crop the petri dish). Defaults to True.
            ONLY_ST2 (bool, optional): Use only stage 2 data (the antibiotic disks). Defaults to False.
            include_improc (bool, optional): Include Improc in the comparison. Defaults to True.
        """
        all_models_names = listdir(model_pred_dir)
        if ".DS_Store" in all_models_names : all_models_names.remove(".DS_Store") 
        makedirs(output_dir,exist_ok=True)
        for model_name in tqdm(all_models_names):
            self.bbox_from_model_pred_st2(path.join(model_pred_dir,model_name),
                                      output_dir,
                                      model_name,
                                      USE_LCC,
                                      ONLY_ST1,
                                      ONLY_ST2)
        # Ground truth.
        df_gt=pd.read_csv(gt_dir,usecols=['Mask_Name','Info','Detected_objects'])
        if include_improc:
            # Improc.
            df_improc=pd.read_csv(improc_dir,usecols=['Mask_Name','Info','Detected_objects'])
            df_final=pd.merge(df_gt,df_improc,how='inner',on='Mask_Name',suffixes=(None, '_imp'))
            df_gt,df_improc=None,None  
        else: df_final = df_gt
        model_names_df = pd.read_csv(model_names_df_dir)
        model_names_df.drop('Unnamed: 0',axis=1,inplace=True)
        for idx, model_name in enumerate(model_names_df['New_names']):
            df1=pd.read_csv(path.join(output_dir,f"{model_name}_bbox.csv"),usecols=['Mask_Name','Info','Detected_objects'])
            df_final=pd.merge(df_final,df1,how='inner',on='Mask_Name',suffixes=(None, f"_{idx}"))
            df1=None
        df_cols = model_names_df.shape[0]
        info_col_names = ['Info_imp'] + [f'Info_{_}' for _ in range(df_cols)] if include_improc else [f'Info_{_}' for _ in range(df_cols)]
        new_col_names = ['Mask_Name','IoU_Improc']+[f'IoU_{_}' for _ in range(df_cols)] if include_improc else ['Mask_Name'] + [f'IoU_{_}' for _ in range(df_cols)]
        #
        def _iou_apply(df_row:pd.Series)->pd.Series:
            """Compute IoU for Improc and other models."""
            # Ground truth bounding box.
            gt_bbox=tf.reshape(tf.constant(eval(df_row['Info']),dtype=tf.float32),[-1, 4])
            # Improc's and other models' bounding boxes.
            output_list = [None]*len(new_col_names)
            output_list[0] = df_row['Mask_Name']
            for idx in range(1,len(new_col_names)):
                improc_bbox=tf.reshape(tf.constant(eval(df_row[info_col_names[idx-1]]),dtype=tf.float32),[-1, 4])
                iou_computed=tfm.vision.iou_similarity.iou(gt_bbox,improc_bbox).numpy()
                reduce_iou = np.max(iou_computed,axis=1)
                output_list[idx] = np.mean(reduce_iou)
            return pd.Series(output_list,index=new_col_names)
        # Compute IoU for the bboxes.
        df2=df_final.apply(_iou_apply,axis=1)
        df_final=pd.merge(df_final,df2,how='inner',on='Mask_Name')
        # Compute Descriptive statistics for Improc IoU and other models.
        df2_desc = df2[new_col_names[1:]].apply(self._desc_sta,axis=0)
        df2_desc.reset_index(inplace=True)
        df2_desc.rename(columns={"index":"Statistics"},inplace=True)
        df2_desc.to_csv(path.join(output_dir,"desc_stat.csv"))
        df2,df2_desc = None,None
        def _f(x):
            """Group IoU."""
            if x>=0 and x<.5: return 1
            elif x>=.5 and x<.7: return 2
            elif x>=.7 and x<.9: return 3
            elif x>=.9 and x<.95: return 4
            elif x>=.95 and x<.96: return 5
            elif x>=.96 and x<.97: return 6
            elif x>=.97 and x<.98: return 7
            elif x>=.98 and x<.99: return 8
            else: return 9
        #
        new_col_groups = ['Mask_Name','G_Improc']+[f'G_{_}' for _ in range(df_cols)] if include_improc else ['Mask_Name'] + [f'G_{_}' for _ in range(df_cols)]
        #
        def _f_apply(df_row:pd.Series)->pd.Series:
            output_list = [None]*len(new_col_groups)
            output_list[0] = df_row['Mask_Name']
            for idx,col_name in enumerate(new_col_names[1:]):
                output_list[idx+1] = _f(df_row[col_name])
            return pd.Series(output_list,index=new_col_groups)
        df3=df_final.apply(_f_apply,axis=1)
        df_final=pd.merge(df_final,df3,how='inner',on='Mask_Name')
        new_col_names = None
        df_final.to_csv(path.join(output_dir,"IoU_comp.csv"))
        df_final = df3
        #
        def _freq_iou(df_col: pd.Series)->pd.Series:
            return pd.Series([len(df_col[df_col==idx]) for idx in range(1,10)],
                             index=[f'{idx}' for idx in range(1,10)])
        #
        df3_freq = df_final[new_col_groups[1:]].apply(_freq_iou,axis=0)
        df3_freq.reset_index(inplace=True)
        df3_freq.rename(columns={"index":"Groups"},inplace=True)
        df3_freq.to_csv(path.join(output_dir,"Freq_iou.csv"))
        df3_freq = None
        def _perce_iou(df_col: pd.Series)->pd.Series:
            return pd.Series([round(100*len(df_col[df_col==idx])/len(df_col),2) for idx in range(1,10)],
                             index=[f'{idx}' for idx in range(1,10)])
        #
        df3_perce = df_final[new_col_groups[1:]].apply(_perce_iou,axis=0)
        df3_perce.reset_index(inplace=True)
        df3_perce.rename(columns={"index":"Groups"},inplace=True)
        df3_perce.to_csv(path.join(output_dir,"Perce_iou.csv"))
        df3_perce,df3,df_final,new_col_groups = None,None,None,None
        
    def check_circle_stage2(self,
                            model_names_df_dir: str | PathLike,
                            model_pred_dir: str | PathLike,
                            output_dir: str | PathLike,
                            tolerance_value:int=0)->None:
        all_models_names = listdir(model_pred_dir)
        if ".DS_Store" in all_models_names : all_models_names.remove(".DS_Store") 
        makedirs(output_dir,exist_ok=True)
        for model_name in tqdm(all_models_names):
            self.bbox_from_model_pred_st2(path.join(model_pred_dir,model_name),
                                      output_dir,
                                      model_name,
                                      False,
                                      False,
                                      True)
        model_names_df = pd.read_csv(model_names_df_dir)
        model_names_df.drop('Unnamed: 0',axis=1,inplace=True)
        df_final = pd.read_csv(path.join(output_dir,f"{model_names_df['New_names'][0]}_bbox.csv"),usecols=['Mask_Name','Info','Detected_objects'])
        for idx, model_name in enumerate(model_names_df['New_names'][1:]):
            df1 = pd.read_csv(path.join(output_dir,f"{model_name}_bbox.csv"),usecols=['Mask_Name','Info','Detected_objects'])
            df_final = pd.merge(df_final,df1,how='inner',on='Mask_Name',suffixes=(None, f"_{idx+1}"))
            df1 = None
        df_final.rename(columns={"Info":"Info_0"},inplace=True)
        df_cols = model_names_df.shape[0]
        info_col_names = [f'Info_{_}' for _ in range(df_cols)]
        new_col_names = ['Mask_Name'] + [f'Correct_{_}' for _ in range(df_cols)]
        #
        def _circle_apply(df_row:pd.Series)->pd.Series:
            """Check the predictions are circles for the given models"""
            output_list = [None]*len(new_col_names)
            output_list[0] = df_row['Mask_Name']
            for idx in range(1,len(new_col_names)):
                bboxes = eval(df_row[info_col_names[idx-1]])
                count = 0
                for bbox in bboxes:
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]
                    if abs(width-height)<=tolerance_value:
                        count+=1
                output_list[idx] = round(count/len(bboxes),4)
            width,height = None,None
            return pd.Series(output_list,index=new_col_names)
        # Check the bboxes are circles.
        df2=df_final.apply(_circle_apply,axis=1)
        df_final=pd.merge(df_final,df2,how='inner',on='Mask_Name')
        # Compute Descriptive statistics for the given models.
        df2_desc = df2[new_col_names[1:]].apply(self._desc_sta,axis=0)
        df2_desc.reset_index(inplace=True)
        df2_desc.rename(columns={"index":"Statistics"},inplace=True)
        df2_desc.to_csv(path.join(output_dir,"desc_stat.csv"))
        df2,df2_desc = None,None
        def _f(x):
            """Group the circles ratios."""
            if x>=0 and x<.5: return 1
            elif x>=.5 and x<.7: return 2
            elif x>=.7 and x<.9: return 3
            elif x>=.9 and x<.95: return 4
            elif x>=.95 and x<.96: return 5
            elif x>=.96 and x<.97: return 6
            elif x>=.97 and x<.98: return 7
            elif x>=.98 and x<.99: return 8
            else: return 9
        #
        new_col_groups = ['Mask_Name'] + [f'G_{_}' for _ in range(df_cols)]
        #
        def _f_apply(df_row:pd.Series)->pd.Series:
            output_list = [None]*len(new_col_groups)
            output_list[0] = df_row['Mask_Name']
            for idx,col_name in enumerate(new_col_names[1:]):
                output_list[idx+1] = _f(df_row[col_name])
            return pd.Series(output_list,index=new_col_groups)
        df3=df_final.apply(_f_apply,axis=1)
        df_final=pd.merge(df_final,df3,how='inner',on='Mask_Name')
        new_col_names = None
        df_final.to_csv(path.join(output_dir,"Circle_comp.csv"))
        df_final = df3
        #
        def _freq_circle(df_col: pd.Series)->pd.Series:
            return pd.Series([len(df_col[df_col==idx]) for idx in range(1,10)],
                             index=[f'{idx}' for idx in range(1,10)])
        #
        df3_freq = df_final[new_col_groups[1:]].apply(_freq_circle,axis=0)
        df3_freq.reset_index(inplace=True)
        df3_freq.rename(columns={"index":"Groups"},inplace=True)
        df3_freq.to_csv(path.join(output_dir,"Freq_circle.csv"))
        df3_freq = None
        def _perce_circle(df_col: pd.Series)->pd.Series:
            return pd.Series([round(100*len(df_col[df_col==idx])/len(df_col),2) for idx in range(1,10)],
                             index=[f'{idx}' for idx in range(1,10)])
        #
        df3_perce = df_final[new_col_groups[1:]].apply(_perce_circle,axis=0)
        df3_perce.reset_index(inplace=True)
        df3_perce.rename(columns={"index":"Groups"},inplace=True)
        df3_perce.to_csv(path.join(output_dir,"Perce_circle.csv"))
        df3_perce,df3,df_final,new_col_groups = None,None,None,None

            
class DataPreparation:
    def __init__(self):
        pass
    def covert_df_st2(self,
                      df_dir: str | PathLike,
                      output_dir: str | PathLike,
                      df_name: str = 'Output')->None:
        """Coverts (x,y,w,h) to (x1,y1,x2,y2) in Info column of the dataframe"""
        df1 = pd.read_csv(df_dir)
        df1.drop('Unnamed: 0',axis=1,inplace=True)
        df1['Mask_Name']=df1.Img_Name.map(lambda p:f"{path.splitext(p)[0]}.png")
        df1.drop('Img_Name',axis=1,inplace=True)
        def _df_f_(df_row: pd.Series)->pd.Series:
            bboxes=eval(df_row['Info'])
            new_info = []
            for _,bbox in bboxes:
                x,y,w,h=bbox
                new_info.append([int(x),int(y),int(x+w),int(y+h)])
            return pd.Series([df_row['Mask_Name'],new_info,len(new_info)],
                             index=['Mask_Name','Info','Detected_objects'])
        df2 = df1.apply(_df_f_,axis=1)
        df1 = None
        df2.to_csv(path.join(output_dir,f'{df_name}.csv'))
        df2 = None
        pass
    
    def merge_data_st1_st2(self,
                           df1_dir: str | PathLike,
                           df2_dir: str | PathLike,
                           output_dir: str | PathLike,
                           df_name:str='Output')->None:
        """Merge dataframes of stage 1, and 2. The bbox in df1 are saved in the format (x1,y1,x2,y2),
        while in df2 are (x,y,w,h). So, we convert df2 to be similar to df1 and then merge it.

        Args:
            df1_dir (str | PathLike): The path to the 1st df.
            df2_dir (str | PathLike): The path to the 2nd df.
            output_dir (str | PathLike): The output path.
            df_name (str, optional): The name of the output df. Defaults to 'Output'.

        """
        df1 = pd.read_csv(df1_dir)
        df1.drop('Unnamed: 0',axis=1,inplace=True)
        df2 = pd.read_csv(df2_dir)
        df2.drop('Unnamed: 0',axis=1,inplace=True)
        df2['Mask_Name']=df2.Img_Name.map(lambda p:f"{path.splitext(p)[0]}.png")
        df2.drop('Img_Name',axis=1,inplace=True)
        df1=pd.merge(df1,df2,how='inner',on='Mask_Name')
        df2 = None
        def _df_f_(df_row: pd.Series)->pd.Series:
            bboxes=eval(df_row['Info'])
            new_info = [[df_row['X1'],df_row['Y1'],df_row['X2'],df_row['Y2']]]
            for _,bbox in bboxes:
                x,y,w,h=bbox
                new_info.append([int(x),int(y),int(x+w),int(y+h)])
            return pd.Series([df_row['Mask_Name'],new_info,len(new_info)],
                             index=['Mask_Name','Info','Detected_objects'])
        df3 = df1.apply(_df_f_,axis=1)
        df1 = None
        makedirs(output_dir,exist_ok=True)
        df3.to_csv(path.join(output_dir,f"{df_name}.csv"))
        df3 = None
        
    def convert_data_st1(self,
                         df1_dir: str | PathLike,
                         output_dir: str | PathLike,
                         df_name:str='Output')->None:
        """Convert data of stage1"""
        df1 = pd.read_csv(df1_dir)
        df1.drop('Unnamed: 0',axis=1,inplace=True)
        def _df_f_(df_row: pd.Series)->pd.Series:
            new_info = [[df_row['X1'],df_row['Y1'],df_row['X2'],df_row['Y2']]]
            return pd.Series([df_row['Mask_Name'],new_info,len(new_info)],
                             index=['Mask_Name','Info','Detected_objects'])
        df3 = df1.apply(_df_f_,axis=1)
        df1 = None
        makedirs(output_dir,exist_ok=True)
        df3.to_csv(path.join(output_dir,f"{df_name}.csv"))
        df3 = None
    
    def convert_masks_to_binary(self,mask_dir: str | PathLike,
                             output_dir: str | PathLike)->None:
        """It converts masks of float values in range [0-255] to binary labels [0,1]. 

        Args:
            mask_dir (str | PathLike): Masks directory of values [0-255].
            output_dir (str | PathLike): Output directory.
        """
        makedirs(path.join(output_dir,"Masks_01") ,exist_ok=True) # output_dir/Masks_01
        all_masks = listdir(mask_dir)
        if ".DS_Store" in all_masks : all_masks.remove(".DS_Store")
        for mask_name in tqdm(all_masks):
            mask = cv.imread(path.join(mask_dir,mask_name),cv.IMREAD_GRAYSCALE)
            mask[mask<=127.5] = 0; mask[mask>127.5] =1
            cv.imwrite(path.join(path.join(output_dir,"Masks_01") ,mask_name),mask)
            mask = None
            
    def merge_masks(self,mask1_dir,mask2_dir,output_dir):
        # We assume mask1 and mask2 of values [0,1]
        all_masks = listdir(mask1_dir)
        if ".DS_Store" in all_masks:all_masks.remove(".DS_Store")
        for mask_name in tqdm(all_masks):
            mask1 = cv.imread(path.join(mask1_dir,mask_name),cv.IMREAD_GRAYSCALE)
            mask2 = cv.imread(path.join(mask2_dir,mask_name),cv.IMREAD_GRAYSCALE)
            mask1= mask1 + mask2
            makedirs(output_dir,exist_ok=True)
            cv.imwrite(path.join(output_dir,mask_name),mask1)
            mask1,mask2=None,None
    
    def resize_img(self,img_dir,mask_dir,output_dir,stage:int=1,img_size:int =256,png_mask=True)->None:
        makedirs(path.join(output_dir,f"Data{img_size}_st{stage}","Processed"),exist_ok=True)
        makedirs(path.join(output_dir,f"Data{img_size}_st{stage}","Mask"),exist_ok=True)
        all_imgs = listdir(img_dir)
        if ".DS_Store" in all_imgs:all_imgs.remove(".DS_Store")
        for img_name in tqdm(all_imgs):
            img_wo_ext, ext = path.splitext(img_name)
            mask_png_name = f"{img_wo_ext}.png" if png_mask else None
            img = cv.imread(path.join(img_dir,img_name))
            img = cv.resize(img,(img_size,img_size),interpolation=cv.INTER_CUBIC)
            cv.imwrite(path.join(output_dir,f"Data{img_size}_st{stage}","Processed",f"{img_wo_ext}.png"),img)
            img = None
            mask = cv.imread(path.join(mask_dir,mask_png_name),cv.IMREAD_GRAYSCALE) if png_mask else cv.imread(path.join(mask_dir,img_name),cv.IMREAD_GRAYSCALE)
            mask = cv.resize(mask,(img_size,img_size),interpolation=cv.INTER_CUBIC)
            cv.imwrite(path.join(output_dir,f"Data{img_size}_st{stage}","Mask",mask_png_name),mask) if png_mask else cv.imwrite(path.join(output_dir,f"Data{img_size}_st{stage}","Mask",img_name),mask)
            mask = None
    
    def merge_df(self, df_dir: str | PathLike, output_dir: str | PathLike, DF_name : str='Output')->None:
        """Merge several dataframes into a single one.

        Args:
            df_dir (str | PathLike): Dataframe directory.
            output_dir (str | PathLike): The desired output directory.
            DF_name (str, optional): Dataframe name. Defaults to 'Output'.
        """
        all_df = listdir(df_dir)
        if ".DS_Store" in all_df:all_df.remove(".DS_Store")
        if ".tmp.driveupload" in all_df:all_df.remove(".tmp.driveupload")
        df_list = [None]*len(all_df)
        for idx,df_name in tqdm(enumerate(all_df)): 
            read_df = pd.read_csv(path.join(df_dir,df_name)) 
            if 'New_Info' in read_df.columns:read_df.rename(columns={"New_Info":"Info","Detected_obj":"Detected_objects"},inplace=True)
            df_list[idx]=read_df
        all_df = pd.concat(df_list,ignore_index=True)
        all_df.drop('Unnamed: 0',axis=1,inplace=True)
        makedirs(output_dir,exist_ok=True)
        all_df.to_csv(path.join(output_dir,f"{DF_name}.csv"))
    
    def drop_df_rows(self, df1_dir: str | PathLike,
                     df2_dir: str | PathLike,
                     output_dir: str | PathLike,
                     DF_name:str='Output')->None:
        """Given dataframe1 and dataframe2, remove df2 from df1.
        """
        makedirs(output_dir,exist_ok=True)
        df1 = pd.read_csv(df1_dir)
        df1.drop('Unnamed: 0',axis=1,inplace=True)
        df1.reset_index(inplace=True)
        df2 = pd.read_csv(df2_dir)
        df2.drop('Unnamed: 0',axis=1,inplace=True)
        df2.reset_index(inplace=True)
        new_df = df1[df1.Img_Name.isin(df2.Img_Name)]
        new_df.reset_index(inplace=True)
        new_df.drop(['level_0','index'],axis=1,inplace=True)
        new_df.to_csv(path.join(output_dir,DF_name))
        
    def remove_folder_from_df(self, img_dir: str | PathLike,
                            df_dir: str | PathLike,
                            output_dir: str | PathLike,
                            USE_COl: list=[1,2],
                            df_name:str='Output')->None:
        """Given dataframe and folder, remove folder elements from the dataframe"""
        all_imgs = listdir(img_dir)
        if ".DS_Store" in all_imgs:all_imgs.remove(".DS_Store")
        output_df = pd.read_csv(df_dir,
                                header=0,
                                index_col=0,
                                usecols=USE_COl
                                ) # Read the DF and set the 'Img_Name' column as index .
        new_df = output_df.drop(all_imgs,
                                axis=0,
                                errors='ignore'
                                ).reset_index().rename(columns={"index":"Img_Name"})	
        new_df.to_csv(path.join(output_dir,f"{df_name}.csv"))
        new_df,output_df=None,None
        
    def plot_sam2_pred(self,df_dir,img_dir,output_dir,df_name:str='Output')->None:
        """Plot SAM2.1 predictions from a dataframe."""
        df=pd.read_csv(df_dir)
        df.drop('Unnamed: 0',axis=1,inplace=True)
        output_img_dir = path.join(output_dir,"Imgs")
        makedirs(output_img_dir,exist_ok=True)
        def df_fun(single_row:pd.Series):
            Img=cv.imread(path.join(img_dir,single_row['Img_Name']))
            bboxes=eval(single_row['Info'])
            for _,bbox in bboxes:
                x,y,w,h=bbox    
                # represents the bottom left corner of rectangle
                start_point = (int(x), int(y))
                # represents the top right corner of rectangle
                end_point = (int(x+w), int(y+h))
                x,y,w,h=None,None,None,None
                # Yellow color in BGR
                color = (0, 255, 255)
                # Line thickness of 2 px
                thickness = 2
                Img=cv.rectangle(Img,start_point,end_point,color,thickness)
            cv.imwrite(path.join(output_img_dir,single_row['Img_Name']),Img)
            Img,start_point,end_point,color,thickness=None,None,None,None,None
            return pd.Series([single_row['Img_Name'],bboxes,len(bboxes)],
                            index=['Img_Name','Info','Detected_objects'])
        df2=df.apply(df_fun,axis=1)
        df2.to_csv(path.join(output_dir,f"{df_name}.csv"))
        df2 = None
        
    def split_detected_atb_improc(self,df_gt_dir,df_imp_dir,output_dir,df_name:str='Output'):
        """Save images where Improc is able to detect all the antibiotics on them to a new dataframe."""
        df_gt = pd.read_csv(df_gt_dir)
        df_gt.drop('Unnamed: 0',axis=1,inplace=True)
        df_improc = pd.read_csv(df_imp_dir,usecols=['Mask_Name','Info','Detected_objects'])
        df_final=pd.merge(df_gt,df_improc,how='inner',on='Mask_Name',suffixes=(None, '_imp'))
        df_gt,df_improc = None,None
        df_final['Missing_atb'] = df_final['Detected_objects']-df_final['Detected_objects_imp']
        df_similar_atb = df_final.loc[df_final.Missing_atb==0]
        df_similar_atb.reset_index(inplace=True)
        df_similar_atb = df_similar_atb[['Mask_Name','Info_imp','Detected_objects_imp']]
        df_similar_atb.rename(columns={"Info_imp":"Info","Detected_objects_imp":"Detected_objects"},inplace=True)
        makedirs(output_dir,exist_ok=True)
        df_similar_atb.to_csv(path.join(output_dir,f'{df_name}.csv'))
        df_similar_atb = None
        df_difference = df_final.loc[df_final.Missing_atb!=0]
        df_difference.reset_index(inplace=True)
        df_difference = df_difference[['Mask_Name','Info_imp','Detected_objects_imp','Missing_atb']]
        df_difference.rename(columns={"Info_imp":"Info","Detected_objects_imp":"Detected_objects"},inplace=True)
        df_difference.to_csv(path.join(output_dir,f'missing_atb_improc.csv'))    
        df_difference,df_final = None,None
        
    def generate_masks(self,
                       model_pred_dir: str | PathLike,
                       output_dir: str | PathLike)->None:
        """Generate masks from models that predict stage 1 and 2. The given masks are of labels [0,1,2]."""
        all_model_folders = listdir(model_pred_dir)
        if ".DS_Store" in all_model_folders : all_model_folders.remove(".DS_Store") 
        for model_name in tqdm(all_model_folders):
            all_mask_names = listdir(path.join(model_pred_dir,model_name))
            if ".DS_Store" in all_mask_names : all_mask_names.remove(".DS_Store") 
            makedirs(path.join(output_dir,model_name),exist_ok=True)
            for mask_name in tqdm(all_mask_names):
                mask = cv.imread(path.join(model_pred_dir,model_name,mask_name),cv.IMREAD_GRAYSCALE)
                height,width = mask.shape
                mask1 = mask.copy()
                mask1[mask1==1] = 255
                img = np.zeros((height,width,3))
                img[:,:,0] = mask1.copy() # Petri dish.
                mask1 = None
                mask2 = mask.copy()
                mask2[mask2==2] = 255
                img[:,:,2] = mask2.copy() # Antibiotic disk.
                mask2 = None
                cv.imwrite(path.join(output_dir,model_name,mask_name),img)
                img,mask = None,None
                        