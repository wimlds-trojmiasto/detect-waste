import json

# Read annotations
with open('annotations.json', 'r') as f:
    dataset = json.loads(f.read())

def convert_taco_to_detectwaste(dataset):
    def taco_to_detectwaste(label):
        # converts taco categories names to detectwaste
        glass = ['Glass bottle','Broken glass','Glass jar']
        metals_and_plastics = ['Aluminium foil', "Clear plastic bottle","Other plastic bottle",
                             "Plastic bottle cap","Metal bottle cap","Aerosol","Drink can",
                             "Food can","Drink carton","Disposable plastic cup","Other plastic cup",
                             "Plastic lid","Metal lid","Single-use carrier bag","Polypropylene bag",
                             "Plastic Film","Six pack rings","Spread tub","Tupperware",
                             "Disposable food container","Other plastic container",
                             "Plastic glooves","Plastic utensils","Pop tab","Scrap metal",
                             "Plastic straw","Other plastic", "Plastic film", "Food Can"]

        non_recyclable = ["Aluminium blister pack","Carded blister pack",
                        "Meal carton","Pizza box","Cigarette","Paper cup",
                        "Meal carton","Foam cup","Glass cup","Wrapping paper",
                        "Magazine paper","Garbage bag","Plastified paper bag",
                        "Crisp packet","Other plastic wrapper","Foam food container",
                        "Rope","Shoe","Squeezable tube","Paper straw","Styrofoam piece",
                        "Rope & strings", "Tissues"]

        other = ["Battery"]
        paper = ["Corrugated carton","Egg carton","Toilet tube","Other carton", "Normal paper", "Paper bag"]
        bio = ["Food waste"]
        unknown = ["Unlabeled litter"]

        if (label in glass):
                label="glass"
        elif (label in metals_and_plastics):
                label="metals_and_plastics"
        elif(label in non_recyclable):
                label="non-recyclable"
        elif(label in other):
                label="other"
        elif (label in paper):
                label="paper"
        elif(label in bio):
                label="bio"
        elif(label in unknown):
                label="unknown"
        else:
            print(label, "is non-taco label")
            label = "unknown"
        return label

    # convert all taco anns to detect-waste anns
    categories = dataset['categories']
    anns = dataset['annotations']
    info = dataset['info']
    
    # update info abou dataset 
    info['description'] = 'detectwaste'
    info['year'] = 2020
    
    # change supercategories from taco to detectwaste
    detectwaste_categories = dataset['categories']
    for ann in anns:
        cat_id = ann['category_id']
        cat_taco = categories[cat_id]['name']
        detectwaste_categories[cat_id]['supercategory'] = taco_to_detectwaste(cat_taco)
    
    # bug fix: As there is no representation of "Plastified paper bag" in annotated data,
    # change of this supercategory was done manually.
    detectwaste_categories[35]['supercategory'] = taco_to_detectwaste("Plastified paper bag")
    
    detectwaste_ids = {}
    detectwaste_cat_names = []
    cat_id = 0
    for cat in detectwaste_categories:
        if cat['supercategory'] not in detectwaste_ids:
            detectwaste_cat_names.append(cat['supercategory'])
            detectwaste_ids[cat['supercategory']] = cat_id
            cat_id += 1

    taco_to_detectwaste_ids = {}
    for i, cat in enumerate(detectwaste_categories):
        taco_to_detectwaste_ids[cat['id']] = detectwaste_ids[cat['supercategory']]

    anns_detectwaste = anns.copy()
    for i, ann in enumerate(anns):
        anns_detectwaste[i]['category_id'] = taco_to_detectwaste_ids[ann['category_id']]
        anns_detectwaste[i].pop('segmentation', None)
    
    print('New ids:', detectwaste_ids)
    return dataset

detectwaste_dataset = convert_taco_to_detectwaste(dataset)
with open('./annotations/annotations_detectwaste.json', 'w') as f:
    json.dump(detectwaste_dataset, f)
