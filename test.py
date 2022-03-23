import matplotlib.pyplot as plt

train_epoch = [1,2,3,4,5]
test_epoch = [1,2,3,4,5]

train_acc = [12,13,14,12,16]
test_acc = [56,43,32,21,10]

train_losses = []
test_losses = []

plt.plot(train_epoch, train_acc)
plt.plot(train_epoch,test_acc)
plt.show()