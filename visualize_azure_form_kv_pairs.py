### VISUALIZE AZURE FORM RECOGNIZER PREDICTIONS

# Usage python visualize_azure_form_predictions.py [predictions json path]
# out: folder with doc images with drawn gt
import numpy as np
import matplotlib as plt
import numpy as np
import cv2
import sys
import json
import pdb
from pdf2image import convert_from_path
import os

def draw_kv_pairs(document_image,annotations,page_number):
    kv_pairs = annotations.get('kv_pairs',[])      
    if len(kv_pairs)>0:
        for kv_idx,kv_pair in enumerate(kv_pairs):      
            key = kv_pair['key']
             
            box = key['bounding_regions'][0]['bounding_box']
            page_number_key = int(key['bounding_regions'][0]['page_number'])
            if page_number!=page_number_key: continue

            key_x0 = int((float(box[0]['x'])/width_inches)*document_image.shape[1])
            key_y0 = int((float(box[0]['y'])/height_inches)*document_image.shape[0])
            key_x1 = int((float(box[2]['x'])/width_inches)*document_image.shape[1])
            key_y1 = int((float(box[2]['y'])/height_inches)*document_image.shape[0])
            

            document_image = cv2.rectangle(document_image, (key_x0,key_y0),(key_x1,key_y1), key_color, thickness)
            
            value = kv_pair['value']
           
            box = value['bounding_regions'][0]['bounding_box']
        
            value_x0 = int((float(box[0]['x'])/width_inches)*document_image.shape[1])
            value_y0 = int((float(box[0]['y'])/height_inches)*document_image.shape[0])
            value_x1 = int((float(box[2]['x'])/width_inches)*document_image.shape[1])
            value_y1 = int((float(box[2]['y'])/height_inches)*document_image.shape[0])
            
            kv_arrow_x0 =int( (key_x1+key_x0)/2)
            kv_arrow_y0 = int((key_y1+key_y0)/2)
            kv_arrow_x1 = int((value_x1+value_x0)/2)
            kv_arrow_y1 = int((value_y1+value_y0)/2)
            
            document_image = cv2.line(document_image, (kv_arrow_x0,kv_arrow_y0), (kv_arrow_x1,kv_arrow_y1), kv_arrow_color,4)

            document_image = cv2.rectangle(document_image, (value_x0,value_y0),(value_x1,value_y1), value_color, thickness)

    return document_image

def draw_tables(document_image,annotations,page_number):
    tables = annotations.get('tables',[])      
    if len(tables)>0:
        for table_idx,table in enumerate(tables):      
            cells = table['cells']
            for cell in cells:
                page_number_cell = int(cell['bounding_regions'][0]['page_number'])
                if page_number!=page_number_cell: continue
                
                box = cell['bounding_regions'][0]['bounding_box']
            
                cell_x0 = int((float(box[0]['x'])/width_inches)*document_image.shape[1])
                cell_y0 = int((float(box[0]['y'])/height_inches)*document_image.shape[0])
                cell_x1 = int((float(box[2]['x'])/width_inches)*document_image.shape[1])
                cell_y1 = int((float(box[2]['y'])/height_inches)*document_image.shape[0])
                

                document_image = cv2.rectangle(document_image, (cell_x0,cell_y0),(cell_x1,cell_y1), cell_color, thickness)

    return document_image

# Get gt file path
annot_path = sys.argv[1]
with open(annot_path) as f:
    annotations = json.load(f)

pdf_path = annot_path.split('_out')[0]+'.pdf'
document_images = convert_from_path(pdf_path)  
width_inches = float(annotations['width'])
height_inches = float(annotations['height'])

n_pages = len(document_images)
current_page = 0
out_path ='out'
if not os.path.exists(out_path): os.makedirs(out_path)
new_page = True
page = 0
thickness=2
key_color = (0,255,0)
value_color = (0,0,255)
kv_arrow_color =(255,0,0)
cell_color = (255,255,0)
for page in range(n_pages):
    document_image=None
    kv_pair_image = None
    table_image =None
    
    document_image = np.array(document_images[page]).copy()

    kv_pair_image=document_image.copy()
    table_image=document_image.copy()
    
    kv_pair_image = draw_kv_pairs(kv_pair_image,annotations,page+1)
    im_path = os.path.join(out_path,os.path.basename(pdf_path)+'_kv_'+str(page)+'.jpg') 
    cv2.imwrite(im_path, kv_pair_image) 
    
    table_image = draw_tables(table_image,annotations,page+1)
    im_path = os.path.join(out_path,os.path.basename(pdf_path)+'_tables_'+str(page)+'.jpg') 
    cv2.imwrite(im_path,table_image)

    document_image=None
    kv_pair_image = None
    table_image =None
