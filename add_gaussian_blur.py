import cv2
import shutil
from tqdm import tqdm
from pathlib import Path


root_p = Path().resolve()

dest_dir = Path(f'{root_p}/dataset/GOPRO_Large/test/x_set_17_std-2p4')
dest_dir.mkdir(exist_ok=True)
src_dir = Path(f'D:/Projects/ISETC2022/dcnn-deblur/dataset/GOPRO_Large/test/y_set')


images = list(src_dir.glob('**/*'))

for img in tqdm(images, total=len(images)):
    image = cv2.imread(str(img), cv2.IMREAD_COLOR)

    # add gaussian blurring
    blur = cv2.GaussianBlur(image, (17, 17), 2.4)
    file_dest = dest_dir / str(img.name).replace(img.suffix, '.png')
    cv2.imwrite(str(file_dest), blur)



# src_dir = Path(f'D:/Projects/ISETC2022/dcnn-deblur/GOPRO_Large/test')
#
# img_idx = 0
# for npath in src_dir.iterdir():
#     npath = npath / 'blur'
#     subdir_images = list(npath.glob('**/*'))
#     for img_f in tqdm(subdir_images):
#         dest_loc = dest_dir / f'{img_idx}.png'
#         shutil.copy(img_f, dest_loc)
#         img_idx += 1
