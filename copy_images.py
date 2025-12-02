import shutil
import os

# Create the target directory structure
target_dir = 'teeth_3d_reconstruction/seg/valid/image'
os.makedirs(target_dir, exist_ok=True)

# Copy and rename the user's images
for i in range(5):
    source = f'my_images/{i}.png'
    target = f'{target_dir}/2-{i}.png'
    if os.path.exists(source):
        shutil.copy2(source, target)
        print(f'Copied {source} to {target}')
    else:
        print(f'Source file {source} not found')

# Verify the files are there
files = os.listdir(target_dir)
print(f'Files in target directory: {files}')