from zipfile import ZipFile as z

from PIL import Image,ImageDraw
import pytesseract
import cv2 as cv
import numpy as np

# loading the face detection classifier
face_cascade = cv.CascadeClassifier('readonly/haarcascade_frontalface_default.xml')
im_name = []
with z('readonly/small_img.zip', 'r') as zip:
    a = zip.infolist()
    zip.extractall()
    for x in a:
        im_name.append(x.filename)
print(im_name)
# the rest is up to you!
def get_img_lst(a):
    temp = []
    image = Image.open(a)
    cv_image = cv.imread(a)
    cv_gray=cv.cvtColor(cv_image, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(cv_gray,scaleFactor = 1.15,minNeighbors = 7)
    for x,y,w,h in faces:
        temp.append(image.crop((x,y,x+w,y+h)))
    return temp
def binarize(image_to_transform, threshold):
    output_image=image_to_transform.convert("L")
    for x in range(output_image.width):
        for y in range(output_image.height):
            if output_image.getpixel((x,y))< threshold:
                output_image.putpixel( (x,y), 0 )
            else:
                output_image.putpixel( (x,y), 255 )
    return output_image

def word_list(a):
    a = pytesseract.image_to_string(binarize(Image.open(a), 128))
    char = [',','.',':','"',"'",';','<','>','?','|']
    for x in a:
        if x in char:
            x.replace(x," ")
    wor_lst = a.split()
    return wor_lst


def combine_img_lst(a):
    img_lst = get_img_lst(a)
    len_img = len(img_lst)
    if len_img == 0:
        return 0
    if len_img != 0:
        width, height = img_lst[0].size
        tot_width = width * 5
        if len_img <= 5:
            row = 1
        else:
            rem = len_img % 5
            if rem == 0:
                row = int(len_img // 5)
            else:
                row = int(len_img // 5 + 1)
        max_height = height * row
        cc_im = Image.new('RGB', (tot_width, max_height))

    x = 0
    y = 0
    index = 1
    for im in img_lst:

        w = img_lst[0].width
        h = int((img_lst[0].width / im.width) * im.height)
        im = im.resize((w, h))

        if index % 6 == 0:
            x = 0
            y += im.size[1]

        cc_im.paste(im, (x, y))
        x += im.size[0]
        index += 1
    return cc_im
def find_word(word):
    for x in im_name:
        if word in word_list(x):
            print("Results found in file "+x)
            img = combine_img_lst(x)
            if img == 0:
                print("But there were no faces in that file!")
            else:
                img.show()
find_word("Christopher")