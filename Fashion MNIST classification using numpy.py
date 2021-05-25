import numpy as np               # linear algebra
import pandas as pd              # data processing
import matplotlib.pyplot as plt  # for plotting


# read train and test files
# --------------------------
train_data = pd.read_csv('train.csv')  #(56000,785),  each row is a picture
test_data = pd.read_csv('test.csv')    #(14000,785)
# separate labels from data for convenience
y_train = train_data['label'].values # labels of train data in rows
x_train = train_data.drop(columns=['label']).values # each example is orginized in a row
print("train data shape: ", train_data.shape, " test data shape: ", test_data.shape, "\n")

# ==============================================================================================

# Useful Constants
#---------------------------
eps = 1e-8   # numerical stability
lambd = 0.01 # regularization
lr = 0.1   # learning rate

# ==============================================================================================
# Data Handling: Normalize train, One hot Labeling
# ----------------------------------------------------

# (1) split train data to train and validation + shuffle
# -------------------------------------------------------
from sklearn.model_selection import train_test_split
val_pct = 0.2  # set percentage for validation split
random_state = 42  # random seed for reproducibility
# Shuffle data, very important!
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_pct, random_state=random_state, stratify=y_train, shuffle=True)

print("train data shape: ", x_train.shape, " validation data shape: ", x_val.shape, "\n")

# (2) Min-Max normalization to input Function
# ---------------------------------------------------------
def normalize(X):
    # get min and max per feature (pixel)
    X_min = np.min(X[:, :], axis=0) # find the minimum value within each column by specifying axis=0:
    X_max = np.max(X[:, :], axis=0)
    return (X[:, :] - X_min)/(X_max - X_min + eps)

x_train = normalize(x_train)
x_val = normalize(x_val)

# ------------------------------------------

# One Hot Encoding for labels
# ---------------------------
digits = 10
train_examples = y_train.shape[0] # num of rows, pictures
val_examples =  y_val.shape[0]
y_train = y_train.reshape(1, train_examples)  #lables in one row
y_val = y_val.reshape(1, val_examples)
Ytrain_new = np.eye(digits)[y_train.astype('int32')]
Ytrain_new = Ytrain_new.T.reshape(digits, train_examples)  # labels one hot encoding in columns
Yval_new = np.eye(digits)[y_val.astype('int32')]
Yval_new = Yval_new.T.reshape(digits, val_examples)  # labels one hot encoding in columns
y_train = Ytrain_new.T # lables in rows again 44800*10
y_val = Yval_new.T  # 11200*10

# ==============================================================================================

# Model Functions: Activations, Loss
# ----------------------------------------------------
# (1) Hidden Layer Activation - Sigmoid
# --------------------------------------
def sigmoid(z):
    """ sigmoid activation function.
    inputs: z
    outputs: sigmoid(z)
    """
    s = 1. / (1. + np.exp(-z))
    return s

# (2) Output Layer Activation - Softmax
# --------------------------------------
def softmax(x):
    e_x = np.exp(x - np.max(x))

    return (e_x.T / (e_x.sum(axis=1)+eps)).T

# (3) Cross Entropy Loss
# --------------------------------------

def cross_entropy(y_train,y_pred):
   M = y_train.shape[0]
   P = -np.log(y_pred[range(M),np.argmax(y_train, axis=1)]+eps)
   CE_loss = np.sum(P) / M
   return CE_loss


# (4) Cross Entropy Loss With L2 Regularization
# ----------------------------------------------
def Cost_with_L2_regularization(y_train,y_pred,w1,w2,lambd):  #(A2, y_train,w1,w2,lambd):
    """
    Implement the cost function with L2 regularization.
    y_pred -- post-activation, output of forward propagation, of shape (number of examples,output size,)
    y_train -- "true" labels vector, of shape (number of examples,output size)
    Returns: cost
    """
    m = y_train.shape[0]
    cross_entropy_loss = cross_entropy(y_train,y_pred)  # gives the cross-entropy part of the cost

    L2_regularization_cost = lambd / (m * 2) * (np.sum(np.square(w1)) + np.sum(np.square(w2)))

    cost = cross_entropy_loss + L2_regularization_cost
    return cost

# ==============================================================================================
# FORWARD PASS
# ----------------
# forward pass func
# ----------------
def forward_pass(x_batch, w1, b1,w2, b2):
    # matrix-vector multiplication
    Z1 = x_batch @ w1 + b1  # Z1=x*w1+b1 64*784 784*128 = 64*128 + 1*128
    A1 = sigmoid(Z1)  # output of hidden layer 64*128
    Z2 = A1 @ w2 + b2  # Z2=w2A1+b2 64*128 128*10 z2 64*10
    # calc. probability of y_j = 1 for each input (M,)
    A2 = softmax(Z2)  # y_pred 10*1

    return Z1, A1, Z2, A2
# -----------------------------

# Calculate Accuracy
# -------------------
def accuracy(y_train, y_pred):
   # number of correct predictions divded by total number of predictions
   acc_bool = (np.argmax(y_pred, axis=1) == np.argmax(y_train, axis=1)) # convert bool to int
   acc_int=acc_bool.astype(int)
   acc = np.mean(acc_int)
   return acc
# -----------------------------


# test
# ----------------
def test(x,y,w1,b1,w2,b2):

    Z1, A1, Z2, A2 = forward_pass(x,w1,b1,w2,b2)

    # Calculate Loss
    test_loss = 0
    loss = Cost_with_L2_regularization(y,A2,w1,w2,lambd)

    # Calculate Accuracy
    acc = accuracy(y,A2)

    return loss, acc

# ----------------------------------

# ==============================================================================================
# training the model
# -------------------

# Relevant Parameters
batch_size = 128
N = x_train.shape[0] # number of rows in x_train
epochs = 50  # number of times to iterate over the entire dataset

# Initialize parameters
# initialize weights iid from a gaussian with small noise. loc = mean, scale = Standard deviation
inputLayerNeurons =  x_train.shape[1] # num of pixels parameters, columns
H = 128  # define number of nuerons in hidden layer
output_layer = 10 # 10 classes
w1 = np.random.normal(loc=0.0, scale=0.01, size=(inputLayerNeurons,H)) # 784*128
b1 = np.random.normal(loc=0.0, scale=0.01, size=(1,H))
w2 = np.random.normal(loc=0.0, scale=0.01, size=(H,output_layer))
b2 = np.random.normal(loc=0.0, scale=0.01, size=(1,output_layer))

train_losses,train_accuracy, val_losses, val_accuracy = ([] for i in range(4))  # create for empty lists

# iterations over entire dataset
for epoch in range(epochs):
    loss = 0
    batch_accuracy = [] #each epoch deserve new batch acc

    # batch iterations within each dataset iteration
    for batch_idx, idx_start in enumerate(range(0, N, batch_size)):
        idx_end = min(idx_start + batch_size, N)
        x_batch = x_train[idx_start:idx_end, :]  # take all data in the current batch
        y_batch = y_train[idx_start:idx_end]  # take relevant labels

        # forward pass, notice A2 is predicted prob
        Z1, A1, Z2, A2 = forward_pass(x_batch, w1, b1,w2, b2)

        # batch accuracy list update
        batch_acc = accuracy(y_batch,A2)
        batch_accuracy.append(batch_acc)


        # loss calculation
        batch_loss = Cost_with_L2_regularization(y_batch,A2,w1,w2,lambd)
        loss += batch_loss

        # Back Propogation
        # ----------------
        dw2 = 0
        db2 = 0
        dw1 = 0
        db1 = 0

        # Compute GD
        dZ2 = A2 - y_batch  #  gradient of the cost w.r.t Z2 ,64*10

        dw2 = (1. / batch_size) * np.matmul(A1.T, dZ2)  # 64*128-> 128*64 x 64*10 =128*10
        db2 = (1. / batch_size) * np.sum(dZ2, axis=0, keepdims=True)  # 1*10
        dA1 = np.matmul(dZ2, w2.T)  # A1 64*128  w2 128*10 64*10
        dZ1 = dA1 * sigmoid(Z1) * (1 - sigmoid(Z1))  # 64*128
        dw1 = (1. / batch_size) * np.matmul(x_batch.T, dZ1)  # w1: 784*128  x batch 64*784->784*64 z1 64*128
        db1 = (1. / batch_size) * np.sum(dZ1, axis=0, keepdims=True)  # 1*128

        # Updating the parameters with Mini-batch GD
        w2 -= lr * (dw2 + (lambd / batch_size) * w2)
        b2 -= db2 * lr
        w1 -= lr * (dw1 + (lambd / batch_size) * w1) #128X784
        b1 -= db1 * lr

    # VALIDATION
    # ------------
    # run model on validation examples
    val_loss, val_acc = test(x_val,y_val,w1,b1,w2,b2)

    # train accuracy calc
    train_acc = np.mean(batch_accuracy)

    # save for plotting
    train_losses.append(loss / batch_idx)
    train_accuracy.append(train_acc)
    val_losses.append(val_loss)
    val_accuracy.append(val_acc)


    #print('loss', loss / batch_idx, ' loss_val', val_loss) # each epoch

print('last train_losses:', train_losses[-1], ' last train_accuracy:', train_accuracy[-1], ' last val_losses:', val_losses[-1],' last val_accuracy:', val_accuracy[-1])


# ==============================================================================================
# plot loss and accuracy on train and validation sets
steps = np.arange(epochs)
fig, ax1 = plt.subplots()
ax1.set_xlabel('epochs')
ax1.set_ylabel('loss')
#ax1.set_title('test loss: %.3f, test accuracy: %.3f' % (test_loss, test_acc))
ax1.plot(steps, train_losses, label="train loss", color='red')
ax1.plot(steps, val_losses, label="val loss", color='green')

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel('accuracy')  # we already handled the x-label with ax1
ax2.plot(steps, val_accuracy, label="val acc", color='blue')
ax2.plot(steps,train_accuracy, label="train acc", color='purple')

fig.legend()
fig.tight_layout()
plt.show()

# ==============================================================================================
# Testing the NN
# ---------------
x_test = test_data.values
x_test = normalize(x_test)
Z1, A1, Z2, A2 = forward_pass(x_test, w1, b1,w2, b2)
y_pred = A2

# check performance on test set
test_acc, test_loss = test(x_test,y_pred,w1,b1,w2,b2)

# Write Predictions to CSV File
# -------------------------------
pred = np.argmax(y_pred, axis = 1) # index of max prob. in a row, meaning for example, indicates the class
np.savetxt("NN_pred.csv", pred, delimiter=",",fmt='%d') # pred is data to be saved to the csv file

# Visual Check of Predictions
# ----------------------------
x_test_arr = np.array(test_data.values)
labels = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneakr','Bag','Ankle boot']

def plot_image(X, title, ax):
    ax.imshow(X, cmap='gray')
    ax.axis('off') # hide the axes ticks
    ax.set_title(title, color = 'black', fontsize=10)


# create plot template
fig , axes = plt.subplots(8,8, figsize=(24,10))
fig.tight_layout()
axes = axes.flatten()

for i in range(64):
    image = x_test_arr[i,:].reshape(28,28)  # image
    title = str(labels[pred[i]])  # class label
    plot_image(image, title, axes[i])

plt.show()


# ======================= END ========================


