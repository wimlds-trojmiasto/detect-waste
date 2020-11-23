def ChangeLabelFunction(label):
        glass=['Glass bottle','Broken glass','Glass jar']
        metals_and_plastics=['Aluminium foil', "Clear plastic bottle","Other plastic bottle","Plastic bottle cap","Metal bottle cap","Aerosol","Drink can",
        "Food can","Drink carton","Disposable plastic cup","Other plastic cup","Plastic lid","Metal lid","Single-use carrier bag","Polypropylene bag","Plastic Film",
        "Six pack rings","Spread tub","Tupperware","Disposable food container","Other plastic container","Plastic glooves","Plastic utensils","Pop tab","Scrap metal","Plastic straw","Other plastic"]
        non_recyclable=["Aluminium blister pack","Carded blister pack","Meal carton","Pizza box","Cigarette","Paper cup","Meal carton","Foam cup","Glass cup","Wrapping paper","Magazine paper",
        "Plastified paper bag","Garbage bag","Crisp packet","Other plastic wrapper","Foam food container","Rope","Shoe","Squeezable tube","Paper straw","Styrofoam piece", "Tissues", "Ciggarette"]
        other=["Battery","Unlabeled litter"]
        paper=["Corrugated carton","Egg carton","Toilet tube","Other carton","Paper bag", "Normal paper"]
        bio=["Food waste"]

        if (label in glass):
                label="glass"
                return label

        if (label in metals_and_plastics):
                label="metals_and_plastics"
                return label
        if(label in non_recyclable):
                label="non_recyclable"
                return label
        if(label in other):
                label="other"
                return label
        if (label in paper):
                label="paper"
                return label
        if(label in bio):
                label="bio"
                return label
        else:
                label="none"
                print("non-taco label")

def ChangeIdFunction(id):
        glass_id=[6, 9, 26]
        metals_and_plastics_id=[0, 4, 5, 7, 8, 10, 11, 12, 16, 21, 24, 27, 28, 29, 36, 37, 40, 41, 43, 44, 45, 47, 48, 49, 50, 52, 55]
        non_recyclable_id=[2, 3, 18, 19, 20, 22, 23, 30, 31, 32, 33, 35, 38, 39, 42, 46, 51, 53, 54, 56, 57, 59]
        other_id=[1, 58]
        paper_id=[13, 14, 15, 17, 33, 34]
        bio_id=[25]

        if (id in glass_id):
                id = 0
                return id
        if (id in metals_and_plastics_id):
                id = 1
                return id
        if (id in non_recyclable_id):
                id = 2
                return id
        if (id in other_id):
                id = 3
                return id
        if (id in paper_id):
                id = 4
                return id
        if (id in bio_id):
                id = 5
                return id
        else:
                id = 3
                return id

def main():
        print(ChangeLabelFunction('Glass bottle'))
        print(ChangeLabelFunction('1'))
