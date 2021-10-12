import urllib.request, shutil
import tarfile
from pathlib import Path
import requests


# url = 'https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks/blob/main/datasets/demo_coding_vs_intergenomic_seqs.tar.gz'
# file_name = './datasets/demo_coding_vs_intergenomic_seqs.tar.gz'

def download_tarfile(url, file_name, force_download=False):
    if(Path(file_name).exists() and not force_download):
        print('Using existing file')
        return
    print('downloadig')
    # with urllib.request.urlopen(url) as response, open(file_name, 'wb') as out_file:
        # shutil.copyfileobj(response, out_file)
    # with urllib.request.urlopen(url) as response:
        # with tarfile.open(fileobj=response, mode='r|gz') as file:
            # file.extractall('./datasets')

    response = requests.get(url)
    with open(file_name, 'wb') as f:
        f.write(response.content)
    
    # response = requests.get(url, stream=True)
    # if response.status_code == 200:
    #     with open(file_name, 'wb') as f:
    #         f.write(response.raw.read())

# file_path = './datasets/demo_mouse_enhancers.tar.gz'
# dest = './datasets'

def untar_file(file_path, dest):
    print('extracting')
    with tarfile.open(file_path, 'r:gz') as f:
        f.extractall(dest)