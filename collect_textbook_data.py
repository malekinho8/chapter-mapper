from utils import *
from utils import *

# provide the textbook file PDF
textbook_file = 'Xia et al. - Active Probe Atomic Force Microscopy.pdf'

print("Gathering textbook info...")

# create a CSV with two columns, one with chapter from which text excerpt was obtained, and another with the actual text excerpt
df = create_textbook_csv_from_pdf(pdf_file=textbook_file)