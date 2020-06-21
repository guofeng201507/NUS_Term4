import numpy as np
import pandas as pd

reports = pd.read_table('osha.txt', header=None, names=["ID", "Title", "Content"])

reports['Text'] = reports['Title'] + '. ' + reports['Content']

pass