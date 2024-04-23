from utils.utils import MedicineDatabase

md = MedicineDatabase()

for i in md.find('注射用盐酸吉西他滨'):
    print(i)
    for j in i['Adjacent Drug Information']:
        print(j)
        

print(md.find('注射用盐酸吉西他滨', Shelf = 'A'))