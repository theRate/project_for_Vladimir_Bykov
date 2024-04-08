from csv_extension import CsvHandler
from calc_extension import DataHandler, SolutionHandler


def show_data(data: list | tuple) -> None:
    """Вспомогательная функция. Используется при отладке программы."""
    for el in data:
        print(*el)


# считываем входной фаил
csv_handler = CsvHandler(input('Введите название .csv файла (без расширения): input/'))
data_read = csv_handler.read_csv()

# обрабатываем данные
prepared_data = DataHandler.input_data_handler(data_read)
fe_list = prepared_data[-1]

# выполняем расчеты
full_data = DataHandler.add_mul_div_lines(prepared_data)
data_and_corrs = DataHandler.pearson_corr_coeff(full_data, fe_list)
data_and_corrs_sorted = DataHandler.select_best_solutions(data_and_corrs)
result_matrix = DataHandler.a_b_coeffs(data_and_corrs_sorted, fe_list)

# записываем в фаил
csv_handler.write_csv([])
for line in result_matrix:
    res = SolutionHandler(line, fe_list).func()
    csv_handler.add_data_csv(res)
print(f'Готово! Фаил {csv_handler.output_file_name}')
