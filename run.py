__author__ = "Zhang, Haoling [zhanghaoling@genomics.cn]"

import warnings
warnings.filterwarnings('ignore')
import numpy as np
from evaluation import EvaluationPipeline

# from coder import Coder
from coder_new import Coder
import os


if __name__ == "__main__":
    # "0001" is provided by the competition management party.
    # please see "record.txt" for the process and score records for details.
    coder = Coder(team_id="0001")


    # pipeline = EvaluationPipeline(coder=coder, error_free=True)
    # pipeline(input_image_path="./images_0713/10DPI_3.bmp", output_image_path="ob_base.bmp",
    #          source_dna_path="o.fasta", target_dna_path="p.fasta", random_seed=2006)
    # print()


    pipeline = EvaluationPipeline(coder=coder, error_free=False)

    # pipeline(input_image_path="./images_0713/10DPI_3.bmp", output_image_path="ob_base_w.bmp",
    #          source_dna_path="o.fasta", target_dna_path="p.fasta", random_seed=2006)

    
    
    root_dir = './images_0713/'
    list_img_names = os.listdir(root_dir)
    list_scores = []
    for temp_img_name in list_img_names:
        temp_img_dir = root_dir + temp_img_name

        score = pipeline(input_image_path=temp_img_dir, output_image_path='./temp/'+temp_img_name,
             source_dna_path='./temp/'+temp_img_name.split('.')[0]+'.fasta', target_dna_path='./temp/'+temp_img_name.split('.')[0]+'_r.fasta', random_seed=2006)
        list_scores.append(score)
    
    print(list_scores)
    print('avg score:', np.mean(list_scores))
