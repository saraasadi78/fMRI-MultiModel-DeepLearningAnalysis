import os
import re
import random
import numpy as np
import nibabel as nib
from glob import glob
from tqdm import tqdm
import logging
import concurrent.futures

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
BASE_DIR = '/media/ekstrandlab/BackupPlus/NaturalisticDatabase_V2'
TASKS = ['500daysofsummer', 'pulpfiction', 'theshawshankredemption', 'theprestige', 'split',
         'littlemisssunshine', '12yearsaslave', 'backtothefuture', 'citizenfour', 'theusualsuspects']

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def create_average_brain_mask(output_file):
    brain_masks = glob(os.path.join(BASE_DIR, 'sub-*', 'anat', 'sub-*_T1w_mask.nii.gz'))
    brain_masks = natural_sort(brain_masks)
    n_sub = len(brain_masks)

    cmd = f"fslmaths {' -add '.join(brain_masks)} -div {n_sub} {output_file}"
    
    logging.info(f"Creating average brain mask: {output_file}")
    ret = os.system(cmd)
    if ret != 0:
        logging.error(f"Error creating average brain mask. fslmaths returned {ret}")
        return False
    return True

def load_and_process_mask(mask_file):
    load_data = nib.load(mask_file)
    mask_data = load_data.get_fdata()
    reshape_mask = mask_data.reshape(-1)
    mask_nonzero = reshape_mask > 0
    return mask_nonzero

def process_subject(sub, task, faces, mask_nonzero, saveDir, saveDirF):
    sub_num = os.path.basename(os.path.dirname(os.path.dirname(sub)))
    logging.info(f"Processing {sub_num} for task {task}")

    # Check if files already exist
    if os.path.exists(os.path.join(saveDirF, sub_num, f'{sub_num}_task-{task}_noface-complete.txt')):
        logging.info(f"Files already exist for {sub_num}, task {task}. Skipping.")
        return

    try:
        load_data = nib.load(sub)
        data = load_data.get_fdata()
    except Exception as e:
        logging.error(f"Error loading data for {sub_num}, task {task}: {str(e)}")
        return

    # Calculate the number of no face instances
    no_face_instances = sum(1 for i in range(len(faces)-1) if faces[i+1,0] - (faces[i,0] + faces[i,1]) > 10)

    # Randomly choose face onsets
    face_onsets = [i for i in range(len(faces)) if faces[i,1] > 10]
    random.shuffle(face_onsets)
    face_onsets = face_onsets[:no_face_instances]

    # Process face onsets
    for i in face_onsets:
        index = int(faces[i,0])
        new_data = data[:,:,:,index:index+10].reshape(-1, 10)[mask_nonzero]
        np.savetxt(os.path.join(saveDir, sub_num, f'{sub_num}_task-{task}_face-{i}.txt'), new_data, delimiter=',')

    # Process no-face periods
    count = 1
    for i in range(len(faces)-1):
        sumF = faces[i,0] + faces[i,1]
        diff = faces[i+1,0] - sumF
        if diff > 10:
            index = int(sumF)
            new_data = data[:,:,:,index:index+10].reshape(-1, 10)[mask_nonzero]
            np.savetxt(os.path.join(saveDirF, sub_num, f'{sub_num}_task-{task}_noface-{count}.txt'), new_data, delimiter=',')
            count += 1

    # Mark task as complete for this subject
    with open(os.path.join(saveDirF, sub_num, f'{sub_num}_task-{task}_noface-complete.txt'), 'w') as f:
        f.write('Processing complete')

def process_task(task):
    subs = glob(os.path.join(BASE_DIR, 'sub-*', 'func', f'sub-*_task-{task}_bold_preprocessedICA.nii.gz'))
    face_file = os.path.join(BASE_DIR, 'stimuli', f'stimuli-task-{task}_face-annotation.1D')
    
    if not os.path.exists(face_file):
        logging.error(f"Face annotation file not found for: {task}")
        return

    try:
        faces = np.loadtxt(face_file)
        logging.info(f"Loaded face annotations for {task}")
    except Exception as e:
        logging.error(f"Error loading face annotations for task {task}: {str(e)}")
        return

    saveDir = os.path.join(BASE_DIR, 'Sara', 'faces')
    saveDirF = os.path.join(BASE_DIR, 'Sara', 'noface')

    # Load mask
    mask_file = os.path.join(BASE_DIR, '85_subBrainMask_average.nii.gz')
    mask_nonzero = load_and_process_mask(mask_file)

    for sub in tqdm(subs, desc=f"Processing {task}"):
        sub_num = os.path.basename(os.path.dirname(os.path.dirname(sub)))
        os.makedirs(os.path.join(saveDir, sub_num), exist_ok=True)
        os.makedirs(os.path.join(saveDirF, sub_num), exist_ok=True)
        process_subject(sub, task, faces, mask_nonzero, saveDir, saveDirF)

def main():
    # Create average brain mask
    mask_file = os.path.join(BASE_DIR, '85_subBrainMask_average.nii.gz')
    if not os.path.exists(mask_file):
        if not create_average_brain_mask(mask_file):
            logging.error("Failed to create average brain mask. Exiting.")
            return

    # Process all tasks
    with concurrent.futures.ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(process_task, TASKS), total=len(TASKS), desc="Processing tasks"))

    logging.info("Processing complete.")

if __name__ == "__main__":
    main()
