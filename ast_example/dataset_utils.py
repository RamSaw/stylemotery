from ast_example.run_exp import read_py_files
import ast

if __name__ == "__main__":
    basefolder = R"C:\Users\bms\PycharmProjects\stylemotery_code\dataset700"
    X, y, tags = read_py_files(basefolder)

    for filename in X:
        try:
            with open(filename) as file:
                ast.parse(file.read())
        except:
            print(filename)
            raise
