
import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt

arr=np.array([1,2,3])
matrix=np.array([[1,2],[3,4]])
mean=np.mean(arr)
median=np.median(arr)
std=np.std(arr)
print(matrix)
reshaped=matrix.reshape(4,1)
print(f"mean:{mean}")
print(f"median:{median}")
print(f"standard deviation:{std}")

print("after reshaping:",matrix)

data={'Name':['Alice','Bob'],'Age':[25,39]}
df=pd.DataFrame(data)
print(f"name:{df['Name']}")
print(f"age:{df.iloc[0]}")

print(f"filtered Age:{df[df['Age']>30]}")
print(f"mean Age:{df['Age'].mean()}")
# Press the green button in the gutter to run the script.
plt.plot([1,2,3],[4,5,6])
plt.show()
#plt.scatter([1,2,3],[4,5,6])
if __name__ == '__main__':
    print('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
