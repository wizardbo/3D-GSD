
import os
import shutil

processed_files = set()

ntu_dir='/home/fubo/shl/MS-G3D-master/data/nturgbd_raw/nturgb+d_skeletons'
ntu1_dir='/home/fubo/shl/MS-G3D-master/data/nturgbd_raw/nturgb+d_skeletons'
ob_dir='/home/fubo/shl/MS-G3D-master/data/nturgbd_raw/ntu'
txt_dir = '/home/fubo/shl/MS-G3D-master/data/nturgbd_raw/NTU_RGBD_samples_with_missing_skeletons.txt'

txt_list = []

with open(txt_dir,'r') as f:
    for name in f:
        txt_list.append(name.strip())

for name in txt_list:
    list_dir = os.listdir(ntu_dir)
    for ntu_name in list_dir:
        if ntu_name.split('.')[0] == name:
            print('1')
            sk_name = ntu_name.split('.')[0]
            shutil.copy(os.path.join(ntu_dir,ntu_name),os.path.join(ob_dir,ntu_name))


# for file_name in os.listdir(ntu_dir):
#     if file_name.endswith('.skeleton') and file_name not in processed_files:
#         with open(os.path.join(ntu_dir,file_name),'r') as f:
#             file_content=f.read()
#             if ' 0 ' in file_content or file_content.startswith('0 ') or file_content.endswith(' 0\n'):
#                 shutil.copy(os.path.join(ntu_dir, file_name), os.path.join(ntu1_dir, file_name))
#                 # shutil.copy(os.path.join(ntu_dir, file_name), ntu1_dir)
#                 processed_files.add(file_name)
#                 print(f"Processed file {file_name}")
