import  shutil
import gzip

def extract_gzip(file_path, dest):
    print('Extracting gzip')
    with open(file_path, 'rb') as f_in:
        with gzip.open(dest, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)