from utils.dataset_coverter import convert_dataset, taco_categories_to_detectwaste
from utils.split_coco_dataset import split_coco_dataset
# update all annotations in one run

# define sources
epi_source = '/dih4/dih4_2/wimlds/data/annotations_epi.json'
taco_source = '/dih4/dih4_2/wimlds/TACO-master/data/annotations.json'

detectwaste_dest = 'annotations/annotations_detectwaste.json'
epi_dest = 'annotations/annotations-epi.json'
split_dest = 'annotations/annotations'


# first, move all category ids from taco annotation style (60 categories)
# to detectwaste (7 categories)
taco_categories_to_detectwaste(source = taco_source, 
                               dest = detectwaste_dest)
# convert form epi to taco
convert_dataset(detectwaste_dest, epi_source, epi_dest)

# now you can both taco and epinote in the training
# at first, you can try using detectwaste_dest annotations for train_set
# and epi_dest annotations for validation

# split files into train and test files
list_of_datasets = [detectwaste_dest, epi_dest]
test_size = 0.2
split_coco_dataset(list_of_datasets, split_dest, test_size)