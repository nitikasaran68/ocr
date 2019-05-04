from PIL import Image, ImageDraw, ImageFont
     
'''img = Image.new('RGB', (100, 33), color = (73, 109, 137))
fontsize = 1  # starting font size
txt = "Hello World"
#txt.rota
# portion of image width you want text width to be
img_fraction = 1

font = ImageFont.truetype("Devnew.ttf", fontsize)
while font.getsize(txt)[0] < img_fraction*img.size[0]:
    # iterate until the text size is just larger than the criteria
    fontsize += 1
    font = ImageFont.truetype("Devnew.ttf", fontsize)

# optionally de-increment to be sure it is less than criteria
fontsize -= 1
font = ImageFont.truetype("Devnew.ttf", fontsize)

d = ImageDraw.Draw(img)
d.text((0,0), txt ,font=font,fill=(255,255,0))
     
img.save('pil_text1.png')'''

import codecs
import os

def create_image(text,savename): 
    img = Image.new('RGB', (100, 33), color = (73, 109, 137))
    fontsize = 1  # starting font size
    txt = text
    #txt.rota
    # portion of image width you want text width to be
    img_fraction = 1

    font = ImageFont.truetype("ARIALUNI.ttf", fontsize)
    while font.getsize(txt)[0] < img_fraction*img.size[0]:
        # iterate until the text size is just larger than the criteria
        fontsize += 1
        font = ImageFont.truetype("ARIALUNI.ttf", fontsize)

    # optionally de-increment to be sure it is less than criteria
    fontsize -= 1
    font = ImageFont.truetype("ARIALUNI.ttf", fontsize)

    d = ImageDraw.Draw(img)
    d.text((0,0), txt ,font=font,fill=(255,255,0))
        
    img.save(savename)

def create_dataset(root_dir):
    if os.path.exists(root_dir) == 0:
        os.mkdir(root_dir)
    with codecs.open("hi_IN.dic",encoding="utf-8") as f:
        lines = f.readlines()
    lines = [s.split("/")[0].rsplit("\n")[0] for s in lines]
    for i in range(10):
        savename = str(i)+"_"+lines[i]+"_"+str(i)+".jpg"
        savename = os.path.join(root_dir,savename)
        create_image(lines[i],savename)

create_dataset("./hindi_dataset")