"""
* improc.py contains Improc class to interact with the Image Processing module of Antibiogo.

"""
import astimp 
from os import path, listdir,makedirs,PathLike
from tqdm import tqdm
import numpy as np
from pandas import DataFrame
from imageio.v2 import imread
from PIL import Image
import cv2 as cv
from skimage.measure import label, regionprops

class Improc:
    """
    Improc class to get masks and bounding box using Antibiogo image processing module of .
    """
    def __init__(self,ast_dir:str | PathLike,output_dir:str | PathLike)->None:
        """Initialize Improc object.

        Args:
            ast_dir (str | PathLike): AST image path.
            output_dir (str | PathLike): Output path.
        """
        self.ast_dir = ast_dir
        self.output_dir = output_dir
        
    def create_mask_stage1(self):
        """Get the mask and bounding-box of the petri dish from the given AST image.
        """
        result_path = path.join(self.output_dir,"Results") # output_dir/Results
        makedirs(result_path,exist_ok=True)
        mask_path = path.join(result_path,"Mask") # output_dir/Results/Mask
        makedirs(mask_path,exist_ok=True)
        proc_img_path = path.join(result_path,"Processed") #output_dir/Results/Processed
        makedirs(proc_img_path,exist_ok=True)        
        all_dirs = listdir(self.ast_dir)
        if ".DS_Store" in all_dirs : all_dirs.remove(".DS_Store")
        x_axis, y_axis, height, width,is_Round,img_name,folder_name = [],[],[],[],[],[],[]
        # Loop over the folders.
        for dir in tqdm(all_dirs):
            all_files = listdir(path.join(self.ast_dir,dir) )
            if ".DS_Store" in all_files: all_files.remove(".DS_Store")
            dir_path_mask = path.join(mask_path,dir) # output_dir/Results/Mask/dir
            makedirs(dir_path_mask,exist_ok=True)
            dir_path_proc =  path.join(proc_img_path,dir) # output_dir/Results/Processed/dir
            makedirs(dir_path_proc,exist_ok=True)
            # Loop over the ast images.
            for ast_picture in tqdm(all_files):
                ast_picture_path = path.join(self.ast_dir,dir, ast_picture)
                # Process files only, skip directories.
                if not path.isfile(ast_picture_path):
                    continue
                try:
                    # Load the AST image.
                    img = np.array(imread(ast_picture_path))
                    # Process the AST.
                    ast = astimp.AST(img)
                    # Get the processed image, the mask, the bounding box, Is the Petri dish round or not?
                    procc_img,mask,bound_box,round_value = ast.get_mask
                    img_name.append(ast_picture);folder_name.append(int(dir[:2]))
                    x_axis.append(bound_box.x); y_axis.append(bound_box.y); width.append(bound_box.width)
                    height.append(bound_box.height); is_Round.append(round_value)
                    
                    name_wout_ext, ext = path.splitext(ast_picture)
                    Image.fromarray(procc_img).save(path.join(dir_path_proc,ast_picture) ) # output_dir/Results/Mask/dir/Img
                    Image.fromarray(mask).save(path.join(dir_path_mask,f"{name_wout_ext}.png"),format="PNG" ) # output_dir/Results/Processed/dir/Img
                except Exception as e:
                    print("Failed for", ast_picture_path, e)
        # Create a dictionary for the results
        saved_values = {'Img_Name':img_name,'X':x_axis,'Y':y_axis,'Height':height,'Width':width,'Is_Round':is_Round,'Folder_name':folder_name}
        df = DataFrame(saved_values)
        # Save the bounding box information to csv file.
        df.to_csv(path_or_buf=path.join(result_path,'Output1.csv'))
        saved_values,df,x_axis, y_axis, height, width,is_Round,img_name=None,None,None,None,None,None,None,None
    
    def get_pellets_from_ast(self,df_name='Output',img_size:int=1024):
        all_imgs = listdir(self.ast_dir)
        if ".DS_Store" in all_imgs: all_imgs.remove(".DS_Store")
        mask_list,info_list,detected_list,confirm_list = [],[],[],[]
        problem_mask_list = []
        for img_name in tqdm(all_imgs):
            img_wo_ext, ext = path.splitext(img_name)
            # Process the AST.
            img = cv.imread(path.join(self.ast_dir,img_name))
            ast = astimp.AST(img)
            ## The image is cropped, so we need to locate the location of antibiotics in the original image.
            # Load the original and cropped images
            original_img = img.copy()
            # The case when Improc is able to identify the petri_dish.
            try:
                cropped_img = ast.crop.copy()
                pellet_r = int(ast.px_per_mm * astimp.config.Pellets_DiamInMillimeters/2)
                img_zeros = np.zeros_like(original_img)
            except:
                problem_mask_list.append(f"{img_wo_ext}.png")
                continue
            # Convert images to grayscale (needed for template matching)
            original_gray = cv.cvtColor(original_img, cv.COLOR_BGR2GRAY)
            cropped_gray = cv.cvtColor(cropped_img, cv.COLOR_BGR2GRAY)
            result = cv.matchTemplate(original_gray, cropped_gray, cv.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
            original_img,cropped_img,original_gray,cropped_gray,min_val, max_val, min_loc = None,None,None,None,None,None,None
            # The top-left corner of the cropped area in the original image
            top_left_crop = max_loc
            pellet_indices = range(len(ast.circles))
            sub_info_list = []
            for pellet_idx in pellet_indices:
                cx,cy = list(map(int,ast.circles[pellet_idx].center))
                cv.circle(img_zeros, (int(cx+top_left_crop[0]),int(cy+top_left_crop[1])), int(pellet_r), (255,255,255), thickness=-1)
            img_zeros = cv.resize(img_zeros,(img_size,img_size),interpolation=cv.INTER_CUBIC)
            img_zeros = cv.threshold(img_zeros, 127, 255, cv.THRESH_BINARY)[1]
            mask = img_zeros[:,:,0]
            label_mask = label(mask)
            props_mask = regionprops(label_mask)
            bboxes = [[prop.bbox[1],prop.bbox[0],prop.bbox[3],prop.bbox[2]] for prop in props_mask]
            label_mask,props_mask,mask,img_zeros = None,None,None,None
            confirm_length = 1 if len(bboxes)==len(ast.circles) else 0
            mask_list.append(f"{img_wo_ext}.png");info_list.append(bboxes);detected_list.append(len(ast.circles)); confirm_list.append(confirm_length)
        top_left_crop,max_loc = None,None
        df1 = DataFrame({'Mask_Name':mask_list,'Info':info_list,'Detected_objects':detected_list,'Confirm_similarity':confirm_list})
        makedirs(self.output_dir,exist_ok=True)
        df1.to_csv(path.join(self.output_dir,f'{df_name}.csv'))
        df1 = None
        df2 = DataFrame({'Mask_Name':problem_mask_list})
        df2.to_csv(path.join(self.output_dir,'Problematic_atb.csv'))
        df2 = None
            
            
        
        
