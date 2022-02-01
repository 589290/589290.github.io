'''
files = []
for i in range(1000000):
    files.append(open('./4.py'))
    print(i)
'''

'''
f = open('./file.txt', 'r')
file_data = f.read()
f.close()
'''

with open('./4.py') as f:
    file_data = f.read()
    
    
'''
camelot_lines = []
with open("camelot.txt") as f:
    for line in f:
        camelot_lines.append(line.strip())

print(camelot_lines)
'''

'''
def create_cast_list(filename):
    cast_list = []
    with open(filename) as f:
        for line in f:
            name = line.split(",")[0]
            cast_list.append(name)

    return cast_list

cast_list = create_cast_list('flying_circus_cast.txt')
for actor in cast_list:
    print(actor)
'''