import numpy as np
from scipy.stats import pearsonr


class DataHandler:
    """Класс содержит методы для подготовки входных данных и расчеты."""

    @staticmethod
    def input_data_handler(data):
        """Метод транспонирует матрицу и приобразует строковые данные в числа."""

        trans_matrix = list(zip(*data))
        res_matrix = []
        for line in trans_matrix:
            line = [line[0]] + list(map(float, line[1:]))
            res_matrix.append(line)

        return res_matrix

    @staticmethod
    def add_mul_div_lines(trans_matrix):
        """Метод добавляет ряды факторных признаков путем перемножения (доп. 66 рядов) и деления (доп. 66 рядов)."""

        full_array = [] + trans_matrix[:-1]
        for i in range(11):
            for j in range(i, 11):
                mul = [f'N{i + 1}N{j + 2}'] + [a * b for a, b in zip(trans_matrix[i][1:], trans_matrix[j + 1][1:])]
                div = [f'N{i + 1}/N{j + 2}'] + [a / b for a, b in zip(trans_matrix[i][1:], trans_matrix[j + 1][1:])]
                full_array.append(mul)
                full_array.append(div)

        return full_array

    @staticmethod
    def pearson_corr_coeff(data: list, result: list) -> list:
        """Метод добавляет к данным их коэффициент корреляции Пирсона к ряду FE лаб."""

        res_data = []
        for line in data:
            corr, _ = pearsonr(line[1:], result[1:])
            line.append(corr)
            res_data.append(line)

        return res_data

    @staticmethod
    def select_best_solutions(data):
        """Метод выбирает 5 самых подходящих решений по максимальному значению коэффициента корреляции Пирсона,
        взятого по модулю."""

        return sorted(data, key=lambda x: abs(x[-1]), reverse=True)[:5]

    @classmethod
    def a_b_coeffs(cls, data, result):
        """Метод рассчитывает коэффициенты линейной регрессии."""

        def linear_regression_coefficients(x, y):
            # Преобразование списка значений x в двумерный массив (столбец)
            x_matrix = np.array(x).reshape((-1, 1))

            # Добавление столбца из единиц для коэффициента b0 (intercept)
            x_matrix_with_intercept = np.hstack((np.ones_like(x_matrix), x_matrix))

            # Вычисление коэффициентов линейной регрессии b0 и b1
            coefficients = np.linalg.inv(x_matrix_with_intercept.T.dot(x_matrix_with_intercept)).dot(
                x_matrix_with_intercept.T.dot(y))

            return coefficients[0], coefficients[1]

        result_matrix = []
        for line in data:
            b, a = linear_regression_coefficients(line[1:-1], result[1:])
            line.append((b, a))
            result_matrix.append(line)

        return result_matrix


class SolutionHandler:
    """Класс для финальной обработки данных и подготовки для записи в фаил."""

    def __init__(self, array, fe_array):
        self.name = array[0]
        self.b = array[-1][0]
        self.a = array[-1][1]
        self.corr = array[-2]
        self.intensity = array[1:-2]
        self.fe = fe_array[1:]

    def func(self):
        """Метод рассчитывает оставшиеся данные и формирует блок записи в фаил."""

        output_result = [[self.name, "Fe lab., %", "Fe calc., % ", "Delta, %", "Sigma, %"], ['=' * 54]]
        fe_calc_list = []
        delta_list = []
        sigma_list = []
        for n, i in enumerate(self.intensity):
            fe_lab = self.fe[n]
            fe_calc = self.b + self.a * i
            fe_calc_list.append(fe_calc)
            delta = abs(fe_calc - fe_lab)
            delta_list.append(delta)
            sigma = 100 * (delta / fe_lab)
            sigma_list.append(sigma)
            res = [i, fe_lab, fe_calc, delta, sigma]
            res = list(map(lambda x: round(x, 5), res))
            output_result.append(res)
        output_result.append([])
        output_result.append(['Intensity', self.name])
        output_result.append(['Correlation', round(self.corr, 5)])
        min_fe_calc = min(fe_calc_list)
        max_fe_calc = max(fe_calc_list)
        output_result.append(['Min', round(min_fe_calc, 5)])
        output_result.append(['Max', round(max_fe_calc, 5)])
        output_result.append(['Max-Min', round(max_fe_calc - min_fe_calc, 5)])
        output_result.append(['a', round(self.a, 5)])
        output_result.append(['b', round(self.b, 5)])
        delta_av = sum(delta_list) / len(delta_list)
        sigma_av = sum(sigma_list) / len(sigma_list)
        output_result.append(['Delta av., %', round(delta_av, 5)])
        output_result.append(['Sigma av., %', round(sigma_av, 5)])
        output_result.append([])

        return output_result
