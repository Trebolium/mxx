import pickle
import sys
import matplotlib.pyplot as plt

pickle_in=open('modelHistory/' +sys.argv[1] +'.pickle','rb')
model_history=pickle.load(pickle_in)

loss = model_history['loss']
acc = model_history['acc']
val_loss = model_history['val_loss']
val_acc = model_history['val_acc']

for x in range(len(loss)):
	print('loss: ', loss[x], 'acc: ', acc[x], 'val_loss: ', val_loss[x], 'val_acc: ', val_acc[x])

# acc = model_history['acc']
# for x in acc:

# val_loss = model_history['val_loss']
# for x in val_loss:
# 	print('val_loss: ', x)

# val_acc = model_history['val_acc']
# for x in val_acc:
# 	print('val_acc: ', x)

epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Model: ' +sys.argv[1] +', Training and validation loss')
plt.legend()

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Model: ' +sys.argv[1] +', Training and validation loss')
plt.legend()
plt.show()