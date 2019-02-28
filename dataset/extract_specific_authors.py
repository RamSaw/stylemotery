import os
import re
import shutil
import sys
import subprocess


def get_author_solutions_dir(path_to_dataset, author):
    return os.path.join(path_to_dataset, author)


def get_solution_filename(author_solution_path, author, problem):
    file_name_parts = [problem.split('.')[0]] + [author] + [problem.split('.')[1]]
    file_name = '.'.join(file_name_parts)
    return os.path.join(author_solution_path, file_name)


def extract_specific_authors(path_to_dataset, authors, problems, path_to_extract="./dataset/python2/"):
    for author in authors:
        author = author.replace(' ', '.')
        author_path = get_author_solutions_dir(path_to_dataset, author)
        if not os.path.exists(author_path):
            print("NO SUCH AUTHOR: " + author_path)
            continue
        for problem in problems:
            file_path = get_solution_filename(author_path, author, problem)
            if not os.path.exists(file_path):
                print("NO SUCH PROBLEM: " + problem + " for author " + author)
                continue
            shutil.copy2(file_path, path_to_extract)
    return


def migrate_to_python3():
    return subprocess.call(['./dataset/python2to3.sh'])


def get_problems_with_extension(problems, programming_language):
    language_to_extensions = {'python': 'py', 'c++': 'cpp'}
    return [problem + '.' + language_to_extensions[programming_language] for problem in problems]


def get_authors(n_authors, n_labels):
    authors_file = os.path.join('.', 'train', 'python', str(n_authors) + "_authors.labels" + str(n_labels) + ".txt")
    with open(authors_file, "r") as f:
        f.readline()
        classes = f.readline()
        return set(re.findall(r'\'(.*?)\'', classes))


if __name__ == "__main__":
    programming_language = sys.argv[1]
    path_to_dataset_to_extract = sys.argv[2]
    authors_number = sys.argv[3]
    labels_number = sys.argv[4]
    problems_with_extension = get_problems_with_extension(sys.argv[5:], programming_language)
    authors = get_authors(authors_number, labels_number)
    extract_specific_authors(path_to_dataset_to_extract, authors, problems_with_extension)
    if programming_language == 'python':
        migrate_to_python3()
