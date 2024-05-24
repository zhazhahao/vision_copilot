from utils.utils import MedicineDatabase
from pprint import pprint
database = MedicineDatabase()
# pprint(database.find("速碧林(那屈肝素钙注射液)"))
pprint(database[416])