import csv
import numpy as np
import matplotlib.pyplot as plt
N = 56

data = np.zeros(shape=(111, 735))

with open('dataset_train.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile)
    bankomat_index = 0
    row_index = 0
    colon_index = 0

    for row in spamreader:
        data[bankomat_index][colon_index] = float(row[7])
        colon_index += 1
        row_index += 1
        if row_index == 735:
            row_index = 0
            colon_index = 0
            bankomat_index += 1

data_answer = np.zeros(shape=(111, 56+N))

data_answer[:, 0:N] = data[:, 735-N:]

for new_day in range(N, 56+N):
    for i in range(111):
        data_answer[i, new_day] = np.mean(data_answer[i, -(56+N)+new_day-N : -(56+N)+new_day-N+N])
data_answer = data_answer[:, N:]

answer_prediction = data_answer

y_pred_inv = answer_prediction.reshape(-1)

data_precise = np.zeros(shape=(111, 56))
with open('dataset_test.csv', newline='') as csvfile1:
    spamreader = csv.reader(csvfile1)
    bankomat_index = 0
    row_index = 0
    colon_index = 0

    for row in spamreader:
        data_precise[bankomat_index][colon_index] = float(row[7])
        colon_index += 1
        row_index += 1
        if row_index == 56:
            row_index = 0
            colon_index = 0
            bankomat_index += 1

y_test_inv = data_precise.reshape(-1)

plt.plot(data_precise[4], 'b', label="precise")
plt.plot(answer_prediction[4], 'r', label="prediction")
plt.ylabel('Value')
plt.xlabel('Time Step')


def smape(a, f):
    return (2*np.abs(f-a) / (np.abs(a) + np.abs(f))*100)

def mae(a,f):
    return np.abs(f-a)



total_smape = 0
total_mae = 0
n=len(y_test_inv)
for name in range(n):
    s_mape = smape(y_test_inv[name], y_pred_inv[name])
    m_ae = mae(y_test_inv[name], y_pred_inv[name])
    total_smape += s_mape
    total_mae += m_ae


print("SMAPE:", total_smape/n, '%')
print("MAE:", total_mae/n, '')
# print("MAPE:", total_mape/n, '%')
plt.legend()
plt.show();

