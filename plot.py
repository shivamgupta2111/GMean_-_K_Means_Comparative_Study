from miml import datasets
from miml.cluster import GMeans
from miml.utils import smile_util

fn = os.path.join(datasets.get_data_home(), 'clustering', 'gaussian',
    'six.txt')
df = DataFrame.read_table(fn, header=None, names=['x1','x2'],
    format='%2f')
x = df.values

model = GMeans()
y = model.fit_predict(x)

scatter(x[:,0], x[:,1], c=y, edgecolor=None, s=3)
title('G-Mean clustering example')