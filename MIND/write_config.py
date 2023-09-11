import sys

with open(f'config_{sys.argv[2]}.py','r') as rf:
    lines = rf.readlines()

tab_str = '    '
model_name = f'model_name = "{sys.argv[1]}"'+ '\n'
try:
    check_point = tab_str + f"checkpoint_num = '{sys.argv[6]}'" + '\n'
except:
    check_point = tab_str + f"checkpoint_num = ''" + '\n'
ns = sys.argv[3].split('_')[1]
ns = tab_str + f"negative_sampling_ratio = {ns}" + '\n'
train_type = sys.argv[3].split('_')[0]
train_type = tab_str + f"training_type = '_{train_type}'" + '\n'
test_type = tab_str + f"test_type = '_{sys.argv[4]}'" + '\n'
test_filter = tab_str + f"test_filter = {sys.argv[5]}" + '\n'

lines[1] = model_name
lines[20] = check_point
lines[21] = ns
lines[22] = train_type
lines[23] = test_type
lines[24] = test_filter

with open('config.py','w') as wf:
    wf.writelines(lines)