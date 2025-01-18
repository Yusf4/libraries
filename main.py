
import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
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
plt.figure()
plt.plot([1,2,3],[4,5,6])
plt.show()
#plt.scatter([1,2,3],[4,5,6])
plt.figure()
plt.bar(['A','B','C'],[10,20,30])
plt.show()
X=[[1],[2],[3],[4],[5]]
y=[1,4,9,16,25]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
print("Training Features:",X_train)
print("training labels:",y_train)
print("Testing Features:",X_test)
print("testing labels:",y_test)
model=LinearRegression()
model.fit(X_train,y_train)
predictions=model.predict(X_test)
plt.figure()
plt.scatter(X,y,color="blue",label="Actual Data")
plt.plot(X,model.predict(X),color="red",label="Fitted Line")
plt.legend()
plt.xlabel("square footage")
plt.ylabel("house price")
plt.title("linear regression model fitting")
plt.show()
if __name__ == '__main__':
    print('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
