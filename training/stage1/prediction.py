"""
* prediction.py is using the trained models to generate masks which will be compared with ground truth ones.
"""

from utils import predict_newimgs
from os import path,makedirs,PathLike,listdir
from tqdm import tqdm
import pandas as pd

def make_model_pred(img_dir: str | PathLike, model_dir: str | PathLike, output_dir: str | PathLike, stage2=False):
    all_models = listdir(model_dir)
    if ".DS_Store" in all_models: all_models.remove(".DS_Store")
    makedirs(output_dir,exist_ok=True)
    new_models_names = [f'Model_{idx}' for idx,_ in enumerate(all_models)]
    model_df = pd.DataFrame({"Original_model_names":all_models,"New_names":new_models_names})
    makedirs(path.join(output_dir,"Dataframes"),exist_ok=True)
    model_df.to_csv(path.join(output_dir,"Dataframes","model_names.csv"))
    model_df = None
    for idx,model_name in tqdm(enumerate(all_models)):
        output_path = path.join(output_dir,"Model_prediction",new_models_names[idx])
        makedirs(output_path,exist_ok=True)
        if 'Effi' in model_name:
            predict_newimgs(img_dir,
                            path.join(model_dir,model_name),
                            output_path,stand_img=False) if not stage2 else predict_newimgs(img_dir,path.join(model_dir,model_name),output_path,False,False)
        else:
            predict_newimgs(img_dir,
                            path.join(model_dir,model_name),
                            output_path) if not stage2 else predict_newimgs(img_dir,path.join(model_dir,model_name),output_path,img_255=False)
            

if __name__=="__main__":
    root_dir = "add_your_root_directory"
    img_path = path.join(root_dir,"Img_models_png")
    output_path = path.join(root_dir,"Stage1_model_prediction")
    model_path_root = path.join(root_dir,"stage1_models")
    make_model_pred(img_path,model_path_root,output_path,False)