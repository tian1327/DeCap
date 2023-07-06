import json
import os
import sys


def select_data(fn, count_flag):
    
    print('\n---------- Selecting data from: ', fn)

    # load the zerocap output captions file
    with open(fn) as f:
        data = json.load(f)
    
    print('Number of total videos: ', len(data))

    # collect videos with Zerocap generated captions
    # copy the vid and info to a new dict
    data_with_cap = {}
    has_zeroCapcaption_ct = 0
    for vid, info in data.items():
        if info['generated_summary'] != '_':
            has_zeroCapcaption_ct += 1
            data_with_cap[vid] = info
    print('Number of videos has Zerocap generated captions: ', has_zeroCapcaption_ct)


    if count_flag == '-1':
        count = int(len(data_with_cap) * 0.25) # top 25% or lowest 25%
    else:
        count = int(count_flag) # specific count, like 10, 20 etc.


    # sort the vid in data_with_cap by the `DeCap_caption_cider` score from low to high
    # and select the top count videos
    data_with_cap_sorted = sorted(data_with_cap.items(), key=lambda x: x[1]['DeCap_caption_cider'])
    data_selected_low = data_with_cap_sorted[:count]

    # sort the vid in data_with_cap by the `DeCap_caption_cider` score from high to low
    # and select the top count videos
    data_with_cap_sorted = sorted(data_with_cap.items(), key=lambda x: x[1]['DeCap_caption_cider'], reverse=True)
    data_selected_high = data_with_cap_sorted[:count]

    # save the selected data to a json file
    fn_high = fn.replace('.json', f'_high{count}.json')
    fn_low = fn.replace('.json', f'_low{count}.json')
    with open(fn_high, 'w') as f:
        json.dump(data_selected_high, f, indent=4)
        print('Dumped data to: ', fn_high)
    
    with open(fn_low, 'w') as f:
        json.dump(data_selected_low, f, indent=4)
        print('Dumped data to: ', fn_low)
   

if __name__ == '__main__':
    
    fn = sys.argv[1]
    count_flag = sys.argv[2]

    select_data(fn, count_flag)    