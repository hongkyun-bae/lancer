
import sys

with open(f'./config/config_{sys.argv[6]}.py','r') as rf:
    lines = rf.readlines()

tab_str = '    '
candidate_type = tab_str + f'candidate_type = "{sys.argv[1]}"' +'\n'
loss_function = tab_str + f'loss_function = "{sys.argv[2]}"' +'\n'
model_name = f'model_name = "{sys.argv[3]}"'+ '\n'
negative_sampling_ratio = tab_str + f'negative_sampling_ratio = {sys.argv[4]}'+ '\n'
lifetime = tab_str + f'lifetime = {sys.argv[5]}'+ '\n'
data = tab_str + f'data = "{sys.argv[6]}"' + '\n'
test_data = tab_str + f'test_behaviors_file = "behaviors_{sys.argv[7]}.tsv"' + '\n'
test_filter = tab_str + f'test_filter = {sys.argv[8]}' + '\n'
history_type = tab_str + f'history_type = "{sys.argv[9]}"' + '\n'
numbering = tab_str + f'numbering = "{sys.argv[-1]}"'+ '\n'

lines[1] = model_name
lines[15] = candidate_type
lines[16] = loss_function
lines[17] = negative_sampling_ratio
lines[18] = lifetime
lines[19] = numbering
lines[20] = data
lines[21] = test_data
lines[22] = test_filter
lines[23] = history_type

with open('config.py','w') as wf:
    wf.writelines(lines)


with open('dataset_origin.py','r') as rf:
    lines = rf.readlines()

tab_str = '    '
candidate_if = tab_str +tab_str + f'if candidate_type == "{sys.argv[1]}":' +'\n'
candidate_item = tab_str +tab_str +tab_str + f'item["candidate_news"] = [self._news2dict(x) for x in row.candidate_news_{sys.argv[1]}.split()]'


lines[118] = candidate_if
lines[119] = candidate_item

with open('dataset.py','w') as wf:
    wf.writelines(lines)
