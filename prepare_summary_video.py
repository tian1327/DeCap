import json
import os

def prepare_data(fn):
    
    print('\n---------- Preparing data from: ', fn)

    # load the zerocap output captions file
    with open(fn) as f:
        data = json.load(f)
    
    print('Number of total videos: ', len(data))

    # check number of videos without Zerocap generated captions
    no_zeroCapcaption_ct = 0
    for video, info in data.items():
        if info['generated_summary'] == '_':
            no_zeroCapcaption_ct += 1
    print('Number of videos without Zerocap generated captions: ', no_zeroCapcaption_ct)

    # add slot2_path for each video, check if slot2_path exists
    no_slot2_path_ct = 0
    for vid, info in data.items():
        hasnat_path = info['path']
        slot2_path = hasnat_path.replace('/home/grads/h/hasnat.md.abdullah/open_ended_activity_analysis/zero-shot-video-to-text/data/ActivityNet_captions_dataset/', 
                                         '/slot2/open_ended_video_analytics/data/ActivityNet_200/')
        # print('hasnat_path: ', hasnat_path)
        # print('slot2_path: ', slot2_path)
        # stop

        if not os.path.exists(slot2_path):
            no_slot2_path_ct += 1
            info['slot2_path'] = ''
            info['slot2_framepath'] = ''
            info['DeCap_caption'] = '_'
        else:
            info['slot2_path'] = slot2_path
            frame_path = '/slot2/open_ended_video_analytics/data/Decap_ANet_videosummary/'+vid.split('_')[-1]
            info['slot2_framepath'] = frame_path
            info['DeCap_caption'] = 'To be generated'

    print('Number of videos without slot2_path: ', no_slot2_path_ct)

    ## dump the data to a json file

    # mkdir if not exists
    output_dir = 'ActivityNet/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print('Created directory: ', output_dir)

    # dump the data to a json file
    outfile = output_dir + fn.split('/')[-1].split('.')[0] + '_tian.json'
    with open(outfile, 'w') as f:
        json.dump(data, f, indent=4)
        print('Dumped data to: ', outfile)


if __name__ == '__main__':

    prepare_data('ActivityNet/Hasnat/zerocap_output_val_1_summary_id_caption.json')
    prepare_data('ActivityNet/Hasnat/zerocap_output_val_2_summary_id_caption.json')
    