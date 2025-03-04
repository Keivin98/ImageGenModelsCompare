import json
import os 

file = json.load(open("/home/local/QCRI/kisufaj/image-generation/app/output/metadata.json", "r"))
res = []
files = list(os.listdir("/home/local/QCRI/kisufaj/image-generation/app/output/"))

prefix = "https://hbkuedu-my.sharepoint.com/personal/keisufaj_hbku_edu_qa/Documents/label-studio/image-gen-label-studio/"
for filename, prompt in file.items():
    files1 = sorted([x for x in files if x.startswith(filename)])
    res.append({
        "prompt": prompt, 
        "sd_image": prefix + files1[2],
        "flux_image": prefix + files1[1],
        "df_image": prefix + files1[0]
    })


with open("label_studio.json", "w") as f:
    json.dump(res, f, indent=4)