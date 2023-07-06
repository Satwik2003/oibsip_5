import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

#importing csv
df = pd.read_csv('Advertising.csv')

#cleaning data
df = df.drop_duplicates()
df = df.dropna()

#preparing data
tget = df['Sales']
feature = df[['TV', 'Radio', 'Newspaper']]

x_train, x_test, y_train, y_test = train_test_split(feature, tget, test_size= 0.1)

#training model
reg = LinearRegression()
reg.fit(x_train, y_train)

print("\n")
#get inputs
inp = []

inp.append(float(input("Enter Cost of TV ads:\t")))
inp.append(float(input("Enter Cost of Radio ads:\t")))
inp.append(float(input("Enter Cost of Newspaper ads:\t")))

inp = np.array(inp).reshape(1,3)

inp_df = pd.DataFrame(data= inp, columns=['TV', 'Radio', 'Newspaper'])

#predict
pred = reg.predict(inp_df)

print('\nTotal sales is:\t{}'.format(pred[0]))