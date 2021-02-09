import json


def taco_to_detectwaste(label):
    # converts taco categories names to detectwaste
    glass = ["Glass bottle", "Broken glass", "Glass jar", "Glass", "glass",
             # open litter map
             "beerBottle", "wineBottle", "juice_bottles", "waterBottle",
              "glass_jar", "ice_tea_bottles", "spiritBottle", ]
    metals_and_plastic = [ # TACO
                          "Aluminium foil", "Clear plastic bottle",
                          "Other plastic bottle", "Plastic bottle cap",
                          "Metal bottle cap", "Aerosol", "Drink can",
                          "Food can", "Drink carton",
                          "Disposable plastic cup",
                          "Other plastic cup", "Plastic lid", "Metal lid",
                          "Single-use carrier bag", "Polypropylene bag",
                          "Plastic Film", "Six pack rings", "Spread tub",
                          "Tupperware", "Disposable food container",
                          "Other plastic container", "Plastic glooves",
                          "Plastic utensils", "Pop tab", "Scrap metal",
                          "Plastic straw", "Other plastic", "Plastic film",
                          "Food Can", "Crisp packet",
                          # Other datasets
                          "plastic", "trash_plastic", 
                          "trash_metal", "HDPEM", "PET", "AluCan", 
                          "metal", "trash_rubber", "rubber", "metals_and_plastic",
                          # open litter map
                          "ice_tea_can", "energy_can", "beerCan", "tinCan",
                          "degraded_plasticbottle", "fizzyDrinkBottle",
                          "milk_bottle",  "plastic_cups", "plasticCutlery", 
                          "plastic_cup_tops", "mediumplastics", "plasticFoodPackaging", 
                          "macroplastics", "plasticAlcoholPackaging", "degraded_plasticbag",
                          "alcohol_plastic_cups", "microplastics", "smoking_plastic",
                          "glass_jar_lid", "crisp_large", "crisp_small", 
                          "aluminium_foil", "bottleTops", "bottleLabel", 
                           ]
    non_recyclable = ["Aluminium blister pack", "Carded blister pack",
                      "Meal carton", "Pizza box", "Cigarette",
                      "Paper cup", "Meal carton", "Foam cup",
                      "Glass cup", "Wrapping paper",
                      "Magazine paper", "Garbage bag",
                      "Plastified paper bag",
                      "Other plastic wrapper", "Foam food container",
                      "Rope", "Shoe", "Squeezable tube", "Paper straw",
                      "Styrofoam piece", "Rope & strings", "Tissues",
                      "trash_fabric", "cloth", "non_recyclable",
                      # open litter map
                      "rope_medium", "tooth_brush", "styro_small",
                      "rope_small", "rope_large", "styro_medium",
                      "styrofoam_plate", "styro_large", "napkins",
                      "election_posters", "paperFoodPackaging"]

    other = ["Battery", "trash_fishing_gear", "other",
             # open litter map
            "batteries", "overflowing_bins", "fishing_gear_nets",
            "dogshit", "elec_large", "chemical", "tyre"]
    paper = ["Corrugated carton", "Egg carton", "Toilet tube",
             "Other carton", "Normal paper", "Paper bag",
             "trash_paper", "paper"]
    bio = ["Food waste","trash_wood", "wood", "bio"]
    unknown = ["Unlabeled litter", "trash_etc", "unknown"]
    ignore = ["rubbish", "Rubbish", "litter"]

    if (label in glass):
        label = "glass"
    elif (label in metals_and_plastic):
        label = "metals_and_plastic"
    elif(label in non_recyclable):
        label = "non_recyclable"
    elif(label in other):
        label = "other"
    elif (label in paper):
        label = "paper"
    elif(label in bio):
        label = "bio"
    elif(label in unknown):
        label = "unknown"
    elif(label in ignore):
        label = "ignore"
    else:
        # print(label, "is non-taco label")
        label = "unknown"
    return label

def label_to_detectwaste(label):
    return taco_to_detectwaste(label)

def taco_categories_to_detectwaste(source, dest):
    # function that updates taco annotations to detectwaste categories
    # from sixty categories to glass, metals_and_plastics, non_recyclable
    # other, paper, bio, unknown

    with open(source, 'r') as f:
        dataset = json.loads(f.read())

    categories = dataset['categories']
    anns = dataset['annotations']
    info = dataset['info']

    # update info about dataset
    info['description'] = 'detectwaste'
    info['year'] = 2021

    # change supercategories and categories from taco to detectwaste
    detectwaste_categories = dataset['categories']
    for ann in anns:
        cat_id = ann['category_id']
        cat_taco = categories[cat_id-1]['name']
        detectwaste_categories[cat_id-1]['supercategory'] = \
            taco_to_detectwaste(cat_taco)

    # bug fix: As there is no representation of
    # "Plastified paper bag" in annotated data,
    # change of this supercategory was done manually.
    try:
        detectwaste_categories[34]['supercategory'] = \
            taco_to_detectwaste("Plastified paper bag")
    except Exception as e:
        print(f"{e}: no plastified paper bag category, ignoring removal")

    detectwaste_ids = {}
    detectwaste_cat_names = []
    cat_id = 1
    for cat in detectwaste_categories:
        if cat['supercategory'] not in detectwaste_ids:
            detectwaste_cat_names.append(cat['supercategory'])
            detectwaste_ids[cat['supercategory']] = cat_id
            cat_id += 1

    # get dictionary to switch ids
    taco_to_detectwaste_ids = {}
    for i, cat in enumerate(detectwaste_categories):
        taco_to_detectwaste_ids[cat['id']] = \
            detectwaste_ids[cat['supercategory']]

    anns_temp = anns.copy()
    anns_detectwaste = anns
    for i, ann in enumerate(anns):
        anns_detectwaste[i]['category_id'] = \
            taco_to_detectwaste_ids[ann['category_id']]
        anns_detectwaste[i].pop('segmentation', None)

    for ann in anns_temp:
        cat_id = ann['category_id']
        try:
            detectwaste_categories[cat_id]['category'] = \
                detectwaste_categories[cat_id]['supercategory']
        except:
            continue
        try:
            detectwaste_categories[cat_id]['name'] = \
                detectwaste_categories[cat_id]['supercategory']
        except:
            continue
        

    anns = anns_detectwaste

    for cat, items in zip(dataset['categories'], detectwaste_ids.items()):
        dataset['categories'] = [cat for cat in dataset['categories']
                                 if cat['id'] < len(detectwaste_ids)]
        category, id = items
        cat['name'] = category
        cat['supercategory'] = category
        cat['category'] = category
        cat['id'] = id

    if dest == None:
        return dataset
    else:
        with open(dest, 'w') as f:
            json.dump(dataset, f)
    #print('Finished converting ids. New ids:', detectwaste_ids)

def convert_categories_to_detectwaste(source, dest):
    return taco_categories_to_detectwaste(source, dest)


def convert_to_binary(source, dest):
    with open(source, 'r') as f:
        dataset = json.loads(f.read())

    anns = dataset['annotations']
    info = dataset['info']

    # update info about dataset
    info['description'] = 'detectwaste_binary'
    info['year'] = 2021

    # update categories
    categories = dict()
    categories['name'] = 'litter'
    categories['category'] = 'litter'
    categories['id'] = 1
    dataset['categories'] = [categories]

    for i, ann in enumerate(anns):
        anns[i]['category_id'] = int(categories['id'])

    with open(dest, 'w') as f:
        json.dump(dataset, f)

def convert_dataset(annotations_template_path,
                    annotations_to_convert_path,
                    save_path):
    # function converts dataset_to_convert
    # to match categories in dataset_template
    # it is designed to work with detectwaste and epinote categories
    with open(annotations_template_path, 'r') as f:
        dataset_template = json.loads(f.read())
    with open(annotations_to_convert_path, 'r') as f:
        dataset_to_convert = json.loads(f.read())

    template_categories = []
    categories = []
    template_to_new = {}
    template_id_to_new_id = {}

    for i, (template_category,
            category) in enumerate(zip(dataset_template['categories'],
                                       dataset_to_convert['categories'])):
        template_categories.append(template_category['name'])
        categories.append(category['name'])
        template_to_new[category['name']] = template_category['name']
        category['name'] = template_category['name']
        category['category'] = template_category['name']
        category['supercategory'] = ''

    # make a dictionary template_id = new_id
    for template_id, category in enumerate(categories):
        new_id = template_categories.index(category)
        template_id_to_new_id[template_id+1] = new_id+1

    # print('Old categories:', categories)
    # print('New categories:', template_categories)
    for annt in dataset_to_convert['annotations']:
        annt['category_id'] = template_id_to_new_id[annt['category_id']]

    with open(save_path, 'w') as f:
        json.dump(dataset_to_convert, f)

    # print('Finished converting dataset')


def concatenate_datasets(list_of_datasets, dest=None):
    # concatenate list of datasets into one single file
    # the first dataset in the list will be used as a base
    # and the rest of datasets will be appended
    last_ann_id = 0
    last_im_id = 0
    concat_dataset = {}
    concat_dataset['images'] = []
    concat_dataset['annotations'] = []

    for i, annot in enumerate(list_of_datasets):
        with open(annot, 'r') as f:
            dataset = json.loads(f.read())

        anns = dataset['annotations'].copy()
        images = dataset['images'].copy()

        img_dict = {}
        for im in images:
            img_dict[im['id']] = last_im_id
            im['id'] = last_im_id
            last_im_id += 1

        for ann in dataset['annotations']:
            ann['image_id'] = img_dict[ann['image_id']]
            ann['id'] = last_ann_id
            last_ann_id += 1

        concat_dataset['images'] += images
        concat_dataset['annotations'] += anns

    concat_dataset['info'] = dataset['info']
    concat_dataset['licenses'] = dataset['licenses']
    concat_dataset['categories'] = dataset['categories']


    if dest is None:
        return concat_dataset
    else:
        with open(dest, 'w') as f:
            json.dump(concat_dataset, f)
        print('Saved results to', dest)
