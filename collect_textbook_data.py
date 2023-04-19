from utils import *

# provide the textbook file PDF
textbook_file = "4453/4453x_week03_01_regression.pdf"
chunk_size = 256
overlap = 0.1
out_name = 'chunk-data '+ '[' + textbook_file.split('.pdf')[0] + ']' + '.csv'

print("Gathering textbook info...")

# create a CSV with two columns, one with chapter from which text excerpt was obtained, and another with the actual text excerpt
ChapterExtractor(textbook_file,chunk_size,overlap).get_csv(out_name)