import csv


class CsvHandler:
    def __init__(self, file_name):
        self.input_file_name = f'input_files/{file_name}.csv'
        self.output_file_name = f'output_files/{file_name}_result.csv'

    def read_csv(self):
        data = []
        with open(self.input_file_name, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                data.append(row)
        return data

    def write_csv(self, data):
        with open(self.output_file_name, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            for row in data:
                csv_writer.writerow(row)

    def add_data_csv(self, data):
        with open(self.output_file_name, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            for row in data:
                csv_writer.writerow(row)
