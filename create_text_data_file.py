import os
import json
import glob
import sys

def create_file():
    text_out_f = open('text_data.txt','w')
    for f in glob.glob(os.path.join(sys.argv[1],'*.json')):
        data = json.load(open(f,'r'))
        for entity in data['form']:
            text_out_f.write(entity['text']+' ')

