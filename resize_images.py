import PIL.Image as PIL
import re, os, sys, urlparse

#this script crops a centered frame from an image, rescales it and saves it to a new location.
#it processes all files in a given folder's subfolders and puts all new images into one new folder

#input parameters
crop_dimensions = 105,105
rescale_dimensions = 64,64
base_output = "G:/machine learning/welding_data_resized_64/"
todo = [("G:/machine learning/welding data raw/P76/logfiles/videodata", base_output + "P76"),
    ("G:/machine learning/welding data raw/P77/logfiles/videodata", base_output + "P77"),
    ("G:/machine learning/welding data raw/P80/logfiles/videodata", base_output + "P80"),
    ("G:/machine learning/welding data raw/P81/logfiles/videodata", base_output + "P81")
    ]

#static variables
JPG = re.compile(".*\.(jpg|jpeg)", re.IGNORECASE)
PNG = re.compile(".*\.png", re.IGNORECASE)

#helper functions
def get_new_file_path(path, output_dir):
    """returns the path the resized image is saved to given the original path and the output directory"""
    base, ext = os.path.splitext(os.path.basename(path))
    return output_dir + base + ext

def resize_folder(image_dir, output_dir):
    """resizes all files in image_dir's subfolders and copies them to output_dir"""
    print "processing " + image_dir
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        os.chdir(output_dir)

    files = os.listdir(image_dir)
    count = 0
    for file in files:
        if PNG.match(file):
            count += 1

            f = image_dir.rstrip("/") + "/" + file
            img = PIL.open(f)

            width = img.size[0]
            height = img.size[1]

            crop_width = min(width,crop_dimensions[0])
            crop_height = min(height,crop_dimensions[0])

            cutoff_x = width - crop_width
            cutoff_y = height - crop_height

            #crop centered rectangle
            cropped = img.crop((cutoff_x/2, cutoff_y/2, width-cutoff_x/2, height-cutoff_y/2))

            #shrink it to final proportions using bicubic interpolation
            resized = cropped.resize((rescale_dimensions[0],rescale_dimensions[1]), PIL.BICUBIC)

            resized.save(get_new_file_path(f,output_dir))
    print "processed %i images" % count

#main script
for tuple in todo:
    for subfolder in os.listdir(tuple[0]):
        if os.path.isdir(tuple[0] + "/" + subfolder):
            resize_folder(tuple[0] + "/" + subfolder, tuple[1] + "/" + subfolder + "/")
sys.exit()