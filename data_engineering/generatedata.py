"""
* generatedata.py is used to generate the data for stage 1 and stage 2, then merges the masks.
Also, it analyzes the masks of two stages.

"""
import os
from dataeng import DataAnalysis,DataPreparation

if __name__ == "__main__":
    root_dir="add_your_root_directory"
    data_an_obj = DataAnalysis()
    data_pr_obj = DataPreparation()
    img_size = 512 # The desired image size.
    data_pr_obj.resize_img(os.path.join(root_dir,"Data/Processed"),
                           os.path.join(root_dir,"Data/Mask"),
                           root_dir,
                           1,img_size)
    data_pr_obj.resize_img(os.path.join(root_dir,"Data/Processed"),
                           os.path.join(root_dir,"SAM2/Stage2_masks_png"),
                           root_dir,
                           2,img_size)
    data_an_obj.img_info(os.path.join(root_dir,f"Data{img_size}_st1/Mask"),
                                    os.path.join(root_dir,f"Data{img_size}_st1/analysis_mask_st1"))
    data_an_obj.img_info(os.path.join(root_dir,f"Data{img_size}_st2/Mask"),
                                   os.path.join(root_dir,f"Data{img_size}_st2/analysis_mask_st2"))
    data_pr_obj.convert_masks_to_binary(os.path.join(root_dir,f"Data{img_size}_st1/Mask"),
                                        os.path.join(root_dir,f"Data{img_size}_st1"))
    data_an_obj.mask_unique_values(os.path.join(root_dir,f"Data{img_size}_st1/Masks_01"),
                                   os.path.join(root_dir,f"Data{img_size}_st1/analysis_mask_st1"))
    data_pr_obj.convert_masks_to_binary(os.path.join(root_dir,f"Data{img_size}_st2/Mask"),
                                        os.path.join(root_dir,f"Data{img_size}_st2"))
    data_an_obj.mask_unique_values(os.path.join(root_dir,f"Data{img_size}_st2/Masks_01"),
                                   os.path.join(root_dir,f"Data{img_size}_st2/analysis_mask_st2"))
    data_pr_obj.merge_masks(os.path.join(root_dir,f"Data{img_size}_st1/Masks_01"),
                            os.path.join(root_dir,f"Data{img_size}_st2/Masks_01"),
                            os.path.join(root_dir,"Merged_masks"))
    data_an_obj.mask_unique_values(os.path.join(root_dir,"Merged_masks"),
                                   os.path.join(root_dir))
    
    
