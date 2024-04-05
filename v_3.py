import csv
import numpy as np
from scipy.stats import pearsonr


class CsvHandler:
    def __init__(self, file_name):
        self.file_name = file_name

    def read_csv(self):
        data = []
        with open(self.file_name, 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=';')
            for row in csv_reader:
                data.append(row)
        return data

    def write_csv(self, data):
        with open(f'output_files/{self.file_name[:-4]}_result.csv', 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            for row in data:
                csv_writer.writerow(row)


csv_handler = CsvHandler(input() + '.csv')
data_read = csv_handler.read_csv()
input_matrix = []
for line in data_read:
    line = list(map(float, line))
    input_matrix.append(line)
trans_matrix = list(zip(*input_matrix))

for line in trans_matrix:
    print(line)

separator = '=' * 100
print(separator)

full_array = [] + trans_matrix[:-1]
for i in range(11):
    for j in range(i, 11):
        mul_array = [a * b for a, b in zip(trans_matrix[j], trans_matrix[j + 1])]
        div_array = [a / b for a, b in zip(trans_matrix[j], trans_matrix[j + 1])]
        full_array.append(mul_array)
        full_array.append(div_array)

y = trans_matrix[-1]
corrs = []
for line in full_array:
    x = line
    corr, _ = pearsonr(x, y)
    corrs.append(abs(corr))

print(corrs)
line_index = corrs.index(max(corrs))
result_line_x = full_array[4]
print(result_line_x)
print(y)
print(separator)


def linear_regression_coefficients(x, y):
    x_matrix = np.array(x).reshape((-1, 1))
    x_matrix_with_intercept = np.hstack((np.ones_like(x_matrix), x_matrix))
    coefficients = np.linalg.inv(x_matrix_with_intercept.T.dot(x_matrix_with_intercept)).dot(x_matrix_with_intercept.T.dot(y))
    return coefficients[0], coefficients[1]


b, a = linear_regression_coefficients(result_line_x, y)
print("Коэффициент 'b' (intercept):", b)
print("Коэффициент 'a' (наклон):", a)
